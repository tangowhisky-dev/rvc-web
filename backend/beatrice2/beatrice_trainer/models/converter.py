# ConverterNetwork (generator).
import gc
import gzip
import json
import math
import os
import shutil
import warnings
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction
from functools import partial
from pathlib import Path
from pprint import pprint
from random import Random
from typing import BinaryIO, Literal, Optional, Union, Sequence, Iterable, Callable

import numpy as np
import pyworld
import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    assert "soundfile" in torchaudio.list_audio_backends()
except AttributeError:
    import soundfile as _sf  # noqa: F401

if not hasattr(torch.amp, "GradScaler"):
    class GradScaler(torch.cuda.amp.GradScaler):
        def __init__(self, _, *args, **kwargs):
            super().__init__(*args, **kwargs)
    torch.amp.GradScaler = GradScaler

PARAPHERNALIA_VERSION = "2.0.0-rc.0"
from ..layers.conv import CausalConv1d, WSConv1d
from ..layers.attention import ConvNeXtStack
from ..layers.vq import VectorQuantizer
from ..models.phone_extractor import PhoneExtractor
from ..models.pitch_estimator import PitchEstimator
from ..models.vocoder import Vocoder, slice_segments, compute_loudness, d4c
from ..io import dump_params, dump_layer
from ..models.reference_encoder import ReferenceEncoder

class ConverterNetwork(nn.Module):
    def __init__(
        self,
        phone_extractor: PhoneExtractor,
        pitch_estimator: PitchEstimator,
        n_speakers: int,
        pitch_bins: int,
        hidden_channels: int,
        vq_topk: int = 4,
        training_time_vq: Literal["none", "self", "random"] = "none",
        phone_noise_ratio: int = 0.5,
        floor_noise_level: float = 1e-3,
        use_reference_encoder: bool = False,
        reference_encoder_channels: int = 256,
    ):
        super().__init__()
        self.frozen_modules = {
            "phone_extractor": phone_extractor.eval().requires_grad_(False),
            "pitch_estimator": pitch_estimator.eval().requires_grad_(False),
        }
        self.pitch_bins = pitch_bins
        self.phone_noise_ratio = phone_noise_ratio
        self.floor_noise_level = floor_noise_level
        self.out_sample_rate = out_sample_rate = 24000
        self.use_reference_encoder = use_reference_encoder
        phone_channels = 128
        self.vq = VectorQuantizer(
            n_speakers=n_speakers,
            codebook_size=512,
            channels=phone_channels,
            topk=vq_topk,
            training_time_vq=training_time_vq,
        )
        self.embed_phone = nn.Conv1d(phone_channels, hidden_channels, 1)
        self.embed_phone.weight.data.normal_(0.0, math.sqrt(2.0 / (256 * 5)))
        self.embed_phone.bias.data.zero_()
        self.embed_quantized_pitch = nn.Embedding(pitch_bins, hidden_channels)
        phase = (
            torch.arange(pitch_bins, dtype=torch.float)[:, None]
            * (
                torch.arange(0, hidden_channels, 2, dtype=torch.float)
                * (-math.log(10000.0) / hidden_channels)
            ).exp_()
        )
        self.embed_quantized_pitch.weight.data[:, 0::2] = phase.sin()
        self.embed_quantized_pitch.weight.data[:, 1::2] = phase.cos_()
        self.embed_quantized_pitch.weight.data *= math.sqrt(4.0 / 5.0)
        self.embed_quantized_pitch.weight.requires_grad_(False)
        self.embed_pitch_features = nn.Conv1d(4, hidden_channels, 1)
        self.embed_pitch_features.weight.data.normal_(0.0, math.sqrt(2.0 / (4 * 5)))
        self.embed_pitch_features.bias.data.zero_()
        self.embed_speaker = nn.Embedding(n_speakers, hidden_channels)
        self.embed_speaker.weight.data.normal_(0.0, math.sqrt(2.0 / 5.0))
        self.embed_formant_shift = nn.Embedding(9, hidden_channels)
        self.embed_formant_shift.weight.data.normal_(0.0, math.sqrt(2.0 / 5.0))

        # Reference encoder — optional; enabled by use_reference_encoder=True.
        # style_projection maps [B, ref_channels, 1] → [B, hidden_channels, 1].
        # Both modules always exist (required for strict state_dict loading); when
        # use_reference_encoder=False the style_projection output is multiplied by
        # a frozen zero mask so it contributes nothing, and no ref_wav is expected.
        self.reference_encoder = ReferenceEncoder(
            style_channels=reference_encoder_channels,
        )
        self.style_projection = nn.Conv1d(reference_encoder_channels, hidden_channels, 1)
        # Zero-init style_projection so the model starts as a baseline clone
        # when fine-tuning from a checkpoint without the reference encoder.
        nn.init.zeros_(self.style_projection.weight)
        nn.init.zeros_(self.style_projection.bias)

        self.key_value_speaker_embedding_length = 384
        self.key_value_speaker_embedding_channels = 128
        self.key_value_speaker_embedding = nn.Embedding(
            n_speakers,
            self.key_value_speaker_embedding_length
            * self.key_value_speaker_embedding_channels,
        )
        self.key_value_speaker_embedding.weight.data[0].normal_()
        self.key_value_speaker_embedding.weight.data[1:] = (
            self.key_value_speaker_embedding.weight.data[0]
        )

        self.vocoder = Vocoder(
            channels=hidden_channels,
            speaker_embedding_channels=self.key_value_speaker_embedding_channels,
            hop_length=out_sample_rate // 100,
            n_pre_blocks=4,
            out_sample_rate=out_sample_rate,
        )
        self.melspectrograms = nn.ModuleList()
        for win_length, n_mels in [
            (32, 5),
            (64, 10),
            (128, 20),
            (256, 40),
            (512, 80),
            (1024, 160),
            (2048, 320),
        ]:
            self.melspectrograms.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=out_sample_rate,
                    n_fft=win_length,
                    win_length=win_length,
                    hop_length=win_length // 4,
                    n_mels=n_mels,
                    power=2,
                    norm="slaney",
                    mel_scale="slaney",
                )
            )

    def initialize_vq(self, inputs: Sequence[Iterable[torch.Tensor]]):
        collector_func = self.frozen_modules["phone_extractor"].units
        target_layer = self.frozen_modules["phone_extractor"].head

        self.vq.build_codebooks(
            collector_func,
            target_layer,
            inputs,
        )
        self.vq.enable_hook(target_layer)

    def enable_hook(self):
        target_layer = self.frozen_modules["phone_extractor"].head
        self.vq.enable_hook(target_layer)

    def _get_resampler(
        self, orig_freq, new_freq, device, cache={}
    ) -> torchaudio.transforms.Resample:
        key = orig_freq, new_freq
        if key in cache:
            return cache[key]
        resampler = torchaudio.transforms.Resample(orig_freq, new_freq).to(
            device, non_blocking=True
        )
        cache[key] = resampler
        return resampler

    def forward(
        self,
        x: torch.Tensor,
        target_speaker_id: torch.Tensor,
        formant_shift_semitone: torch.Tensor,
        pitch_shift_semitone: Optional[torch.Tensor] = None,
        slice_start_indices: Optional[torch.Tensor] = None,
        slice_segment_length: Optional[int] = None,
        return_stats: bool = False,
        ref_wav: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
        # x: [batch_size, 1, wav_length]
        # target_speaker_id: Long[batch_size]
        # formant_shift_semitone: [batch_size]
        # pitch_shift_semitone: [batch_size]
        # slice_start_indices: [batch_size]

        batch_size, _, _ = x.size()
        self.vq.set_target_speaker_ids(target_speaker_id)

        with torch.inference_mode():
            phone_extractor: PhoneExtractor = self.frozen_modules["phone_extractor"]
            pitch_estimator: PitchEstimator = self.frozen_modules["pitch_estimator"]
            # [batch_size, 1, wav_length] -> [batch_size, phone_channels, length]
            phone = phone_extractor.units(x).transpose(1, 2)

            if self.training and self.phone_noise_ratio != 0.0:
                phone *= (1.0 - self.phone_noise_ratio) / phone.square().mean(
                    1, keepdim=True
                ).sqrt_()
                noise = torch.randn_like(phone)
                noise *= (
                    self.phone_noise_ratio
                    / noise.square().mean(1, keepdim=True).sqrt_()
                )
                phone += noise
            # F.rms_norm は PyTorch >= 2.4 が必要
            phone *= (
                1.0
                / phone.square()
                .mean(1, keepdim=True)
                .add_(torch.finfo(torch.float).eps)
                .sqrt_()
            )

            # [batch_size, 1, wav_length] -> [batch_size, pitch_bins, length], [batch_size, 1, length]
            pitch, energy = pitch_estimator(x)
            # augmentation
            if self.training:
                # [batch_size, pitch_bins - 1]
                weights = pitch.softmax(1)[:, 1:, :].mean(2)
                # [batch_size]
                mean_pitch = (
                    weights
                    * torch.arange(
                        1,
                        self.embed_quantized_pitch.num_embeddings,
                        device=weights.device,
                    )
                ).sum(1) / weights.sum(1)
                mean_pitch = mean_pitch.round_().long()
                target_pitch = torch.randint_like(mean_pitch, 64, 257)
                shift = target_pitch - mean_pitch
                shift_ratio = (
                    2.0 ** (shift.float() / pitch_estimator.pitch_bins_per_octave)
                ).tolist()
                shift = []
                interval_length = 100  # 1s
                interval_zeros = torch.zeros(
                    (1, 1, interval_length * 160), device=x.device
                )
                concatenated_shifted_x = []
                offsets = [0]
                torch.backends.cudnn.benchmark = False
                for i in range(batch_size):
                    shift_ratio_i = shift_ratio[i]
                    shift_ratio_fraction_i = Fraction.from_float(
                        shift_ratio_i
                    ).limit_denominator(30)
                    shift_numer_i = shift_ratio_fraction_i.numerator
                    shift_denom_i = shift_ratio_fraction_i.denominator
                    shift_ratio_i = shift_numer_i / shift_denom_i
                    shift_i = int(
                        round(
                            math.log2(shift_ratio_i)
                            * pitch_estimator.pitch_bins_per_octave
                        )
                    )
                    shift.append(shift_i)
                    shift_ratio[i] = shift_ratio_i
                    # [1, 1, wav_length / shift_ratio]
                    with torch.amp.autocast("cuda", enabled=False):
                        shifted_x_i = self._get_resampler(
                            shift_numer_i, shift_denom_i, x.device
                        )(x[i])[None]
                    if shifted_x_i.size(2) % 160 != 0:
                        shifted_x_i = F.pad(
                            shifted_x_i,
                            (0, 160 - shifted_x_i.size(2) % 160),
                            mode="reflect",
                        )
                    assert shifted_x_i.size(2) % 160 == 0
                    offsets.append(
                        offsets[-1] + interval_length + shifted_x_i.size(2) // 160
                    )
                    concatenated_shifted_x.extend([interval_zeros, shifted_x_i])
                if offsets[-1] % 256 != 0:
                    # 長さが同じ方が何かのキャッシュが効いて早くなるようなので
                    # 適当に 256 の倍数になるようにパディングして長さのパターン数を減らす
                    concatenated_shifted_x.append(
                        torch.zeros(
                            (1, 1, (256 - offsets[-1] % 256) * 160), device=x.device
                        )
                    )
                # [batch_size, 1, sum(wav_length) + batch_size * 16000]
                concatenated_shifted_x = torch.cat(concatenated_shifted_x, dim=2)
                assert concatenated_shifted_x.size(2) % (256 * 160) == 0
                # [1, pitch_bins, length / shift_ratio], [1, 1, length / shift_ratio]
                concatenated_pitch, concatenated_energy = pitch_estimator(
                    concatenated_shifted_x
                )
                for i in range(batch_size):
                    shift_i = shift[i]
                    shift_ratio_i = shift_ratio[i]
                    left = offsets[i] + interval_length
                    right = offsets[i + 1]
                    pitch_i = concatenated_pitch[:, :, left:right]
                    energy_i = concatenated_energy[:, :, left:right]
                    pitch_i = F.interpolate(
                        pitch_i,
                        scale_factor=shift_ratio_i,
                        mode="linear",
                        align_corners=False,
                    )
                    energy_i = F.interpolate(
                        energy_i,
                        scale_factor=shift_ratio_i,
                        mode="linear",
                        align_corners=False,
                    )
                    assert pitch_i.size(2) == energy_i.size(2)
                    assert abs(pitch_i.size(2) - pitch.size(2)) <= 10
                    length = min(pitch_i.size(2), pitch.size(2))

                    if shift_i > 0:
                        pitch[i : i + 1, :1, :length] = pitch_i[:, :1, :length]
                        pitch[i : i + 1, 1:-shift_i, :length] = pitch_i[
                            :, 1 + shift_i :, :length
                        ]
                        pitch[i : i + 1, -shift_i:, :length] = -10.0
                    elif shift_i < 0:
                        pitch[i : i + 1, :1, :length] = pitch_i[:, :1, :length]
                        pitch[i : i + 1, 1 : 1 - shift_i, :length] = -10.0
                        pitch[i : i + 1, 1 - shift_i :, :length] = pitch_i[
                            :, 1:shift_i, :length
                        ]
                    energy[i : i + 1, :, :length] = energy_i[:, :, :length]
                torch.backends.cudnn.benchmark = True

            # [batch_size, pitch_bins, length] -> Long[batch_size, length], [batch_size, 3, length]
            quantized_pitch, pitch_features = pitch_estimator.sample_pitch(
                pitch, return_features=True
            )
            if pitch_shift_semitone is not None:
                quantized_pitch = torch.where(
                    quantized_pitch == 0,
                    quantized_pitch,
                    (
                        quantized_pitch
                        + (
                            pitch_shift_semitone[:, None]
                            * (pitch_estimator.pitch_bins_per_octave / 12.0)
                        )
                        .round_()
                        .long()
                    ).clamp_(1, self.pitch_bins - 1),
                )
            pitch = 55.0 * 2.0 ** (
                quantized_pitch.float() / pitch_estimator.pitch_bins_per_octave
            )
            # phone が 2.5ms 先読みしているのに対して、
            # energy は 12.5ms, pitch_features は 22.5ms 先読みしているので、
            # ずらして phone に合わせる
            energy = F.pad(energy[:, :, :-1], (1, 0), mode="reflect")
            quantized_pitch = F.pad(quantized_pitch[:, :-2], (2, 0), mode="reflect")
            pitch_features = F.pad(pitch_features[:, :, :-2], (2, 0), mode="reflect")
            # [batch_size, 1, length], [batch_size, 3, length] -> [batch_size, 4, length]
            pitch_features = torch.cat([energy, pitch_features], dim=1)
            formant_shift_indices = (
                ((formant_shift_semitone + 2.0) * 2.0).round_().long()
            )

        phone = phone.clone()
        quantized_pitch = quantized_pitch.clone()
        pitch_features = pitch_features.clone()
        formant_shift_indices = formant_shift_indices.clone()
        pitch = pitch.clone()

        # [batch_sise, hidden_channels, length]
        x = (
            self.embed_phone(phone)
            + self.embed_quantized_pitch(quantized_pitch).transpose(1, 2)
            + self.embed_pitch_features(pitch_features)
            + (
                self.embed_speaker(target_speaker_id)[:, :, None]
                + self.embed_formant_shift(formant_shift_indices)[:, :, None]
            )
        )
        # Reference encoder: compute style vector from ref_wav and add to x.
        # ref_wav is a different clip from the same speaker (leakage prevention).
        # When use_reference_encoder is False or ref_wav is None, the style
        # projection still runs but on a zero tensor — contributing nothing
        # (style_projection is zero-init; when no ref_wav, we inject zeros).
        if self.use_reference_encoder and ref_wav is not None:
            # ref_wav: [batch, T_ref] at 16 kHz
            style_vec = self.reference_encoder(ref_wav)   # [batch, ref_channels]
            # Guard against NaN/inf from random conv init or AMP fp16 overflow.
            _non_finite = (~style_vec.isfinite()).sum().item()
            if _non_finite:
                self._ref_enc_nan_count = getattr(self, "_ref_enc_nan_count", 0) + int(_non_finite)
                self._ref_enc_nan_steps = getattr(self, "_ref_enc_nan_steps", 0) + 1
                # Log on first occurrence and then every 100 steps that had NaN
                if self._ref_enc_nan_steps == 1 or self._ref_enc_nan_steps % 100 == 0:
                    print(
                        f"[ref_enc] nan_to_num activated: {int(_non_finite)} non-finite values "
                        f"in style_vec (cumulative: {self._ref_enc_nan_count} values across "
                        f"{self._ref_enc_nan_steps} steps)",
                        flush=True,
                    )
                style_vec = torch.nan_to_num(style_vec, nan=0.0, posinf=0.0, neginf=0.0)
            style_vec = style_vec.unsqueeze(-1)            # [batch, ref_channels, 1]
            x = x + self.style_projection(style_vec)      # broadcast over length
        if slice_start_indices is not None:
            assert slice_segment_length is not None
            # [batch_size, hidden_channels, length] -> [batch_size, hidden_channels, segment_length]
            x = slice_segments(x, slice_start_indices, slice_segment_length)
        x = F.silu(x, inplace=True)

        speaker_embedding = self.key_value_speaker_embedding(target_speaker_id).view(
            batch_size,
            self.key_value_speaker_embedding_length,
            self.key_value_speaker_embedding_channels,
        )

        # [batch_size, hidden_channels, segment_length] -> [batch_size, 1, segment_length * 240]
        y_g_hat, stats = self.vocoder(x, pitch, speaker_embedding)
        stats["pitch"] = pitch
        if return_stats:
            return y_g_hat, stats
        else:
            return y_g_hat

    def _normalize_melsp(self, x):
        return x.clamp(min=1e-10).log_()

    def forward_and_compute_loss(
        self,
        noisy_wavs_16k: torch.Tensor,
        target_speaker_id: torch.Tensor,
        formant_shift_semitone: torch.Tensor,
        slice_start_indices: torch.Tensor,
        slice_segment_length: int,
        y_all: torch.Tensor,
        enable_loss_ap: bool = False,
        ref_wavs_16k: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, float],
    ]:
        # noisy_wavs_16k: [batch_size, 1, wav_length]
        # target_speaker_id: Long[batch_size]
        # formant_shift_semitone: [batch_size]
        # slice_start_indices: [batch_size]
        # slice_segment_length: int
        # y_all: [batch_size, 1, wav_length]

        stats = {}
        loss_mel = 0.0
        loss_loudness = 0.0
        loudness_win_lengths = [512, 1024, 2048, 4096]

        # [batch_size, 1, wav_length] -> [batch_size, 1, wav_length * 240]
        y_hat_all, intermediates = self(
            noisy_wavs_16k,
            target_speaker_id,
            formant_shift_semitone,
            return_stats=True,
            ref_wav=ref_wavs_16k[:, 0, :] if ref_wavs_16k is not None else None,
        )
        y_hat_all = y_hat_all.detach().where(y_all == 0.0, y_hat_all)

        with torch.amp.autocast("cuda", enabled=False):
            periodic_signal = intermediates["periodic_signal"].float()
            aperiodic_signal = intermediates["aperiodic_signal"].float()
            noise_excitation = intermediates["noise_excitation"].float()
            periodic_signal = periodic_signal[:, : noise_excitation.size(1)]
            aperiodic_signal = aperiodic_signal[:, : noise_excitation.size(1)]
            y_hat_all = y_hat_all.float()
            floor_noise = torch.randn_like(y_all) * self.floor_noise_level
            y_all = y_all + floor_noise
            y_hat_all += floor_noise
            y_hat_all_truncated = y_hat_all.squeeze(1)[:, : periodic_signal.size(1)]
            y_all_truncated = y_all.squeeze(1)[:, : periodic_signal.size(1)]

            y_loudness = compute_loudness(
                y_all_truncated, self.out_sample_rate, loudness_win_lengths
            )
            y_hat_loudness = compute_loudness(
                y_hat_all_truncated, self.out_sample_rate, loudness_win_lengths
            )
            for win_length, y_loudness_i, y_hat_loudness_i in zip(
                loudness_win_lengths, y_loudness, y_hat_loudness
            ):
                loss_loudness_i = F.mse_loss(y_hat_loudness_i, y_loudness_i)
                loss_loudness += loss_loudness_i * math.sqrt(win_length)
                stats[f"loss_loudness_{win_length}"] = loss_loudness_i.item()

            for melspectrogram in self.melspectrograms:
                melsp_periodic_signal = melspectrogram(periodic_signal)
                melsp_aperiodic_signal = melspectrogram(aperiodic_signal)
                melsp_noise_excitation = melspectrogram(noise_excitation)
                # [1, n_mels, 1]
                # 1/6 ... [-0.5, 0.5] の一様乱数の平均パワー
                # 3/8 ... ハン窓をかけた時のパワー減衰
                # 0.5 ... 謎
                reference_melsp = melspectrogram.mel_scale(
                    torch.full(
                        (1, melspectrogram.n_fft // 2 + 1, 1),
                        (1 / 6) * (3 / 8) * 0.5 * melspectrogram.win_length,
                        device=noisy_wavs_16k.device,
                    )
                )
                aperiodic_ratio = melsp_aperiodic_signal / (
                    melsp_periodic_signal + melsp_aperiodic_signal + 1e-5
                )
                compensation_ratio = reference_melsp / (melsp_noise_excitation + 1e-5)

                melsp_y_hat = melspectrogram(y_hat_all_truncated)
                melsp_y_hat = melsp_y_hat * (
                    (1.0 - aperiodic_ratio) + aperiodic_ratio * compensation_ratio
                )
                y_hat_mel = self._normalize_melsp(melsp_y_hat)

                y_mel = self._normalize_melsp(melspectrogram(y_all_truncated))
                loss_mel_i = F.l1_loss(y_hat_mel, y_mel)
                loss_mel += loss_mel_i
                stats[
                    f"loss_mel_{melspectrogram.win_length}_{melspectrogram.n_mels}"
                ] = loss_mel_i.item()

            loss_mel /= len(self.melspectrograms)

            if enable_loss_ap:
                t = (
                    torch.arange(intermediates["pitch"].size(1), device=y_all.device)
                    * 0.01
                    + 0.005
                )
                y_coarse_aperiodicity, y_rms = d4c(
                    y_all.squeeze(1),
                    intermediates["pitch"],
                    t,
                    self.vocoder.out_sample_rate,
                    coarse_only=True,
                )
                y_coarse_aperiodicity = 10.0 ** (y_coarse_aperiodicity / 10.0)
                y_hat_coarse_aperiodicity, y_hat_rms = d4c(
                    y_hat_all.squeeze(1),
                    intermediates["pitch"],
                    t,
                    self.vocoder.out_sample_rate,
                    coarse_only=True,
                )
                y_hat_coarse_aperiodicity = 10.0 ** (y_hat_coarse_aperiodicity / 10.0)
                rms = torch.maximum(y_rms, y_hat_rms)
                loss_ap = F.mse_loss(
                    y_hat_coarse_aperiodicity, y_coarse_aperiodicity, reduction="none"
                )
                loss_ap *= (rms / (rms + 1e-3) * (rms > 1e-5))[:, :, None]
                loss_ap = loss_ap.mean()
            else:
                loss_ap = torch.tensor(0.0)

        # [batch_size, 1, wav_length] -> [batch_size, 1, slice_segment_length * 240]
        y_hat = slice_segments(
            y_hat_all, slice_start_indices * 240, slice_segment_length * 240
        )
        # [batch_size, 1, wav_length] -> [batch_size, 1, slice_segment_length * 240]
        y = slice_segments(y_all, slice_start_indices * 240, slice_segment_length * 240)
        return y, y_hat, y_hat_all, loss_loudness, loss_mel, loss_ap, stats

    def merge_weights(self):
        self.vocoder.merge_weights()

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.embed_phone, f)
        dump_layer(self.embed_quantized_pitch, f)
        dump_layer(self.embed_pitch_features, f)
        dump_layer(self.vocoder, f)

    def dump_reference_encoder(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        """Serialise the reference encoder + style_projection as a standalone .pt file.

        This is the only data needed at inference to encode a new reference clip.
        Called during training completion and export so inference doesn't need
        the full checkpoint for reference encoding.
        """
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump_reference_encoder(f)
            return
        import torch as _torch
        _torch.save({
            "reference_encoder": self.reference_encoder.state_dict(),
            "style_projection": self.style_projection.state_dict(),
            "use_reference_encoder": self.use_reference_encoder,
            "style_channels": self.reference_encoder.proj.out_features,
            "hidden_channels": self.style_projection.out_channels,
        }, f)

    def dump_speaker_embeddings(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump_speaker_embeddings(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_params(self.vq.codebooks, f)
        dump_layer(self.embed_speaker, f)
        dump_layer(self.embed_formant_shift, f)
        dump_layer(self.key_value_speaker_embedding, f)

    def dump_embedding_setter(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump_embedding_setter(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        self.vocoder.prenet.dump_kv(f)


# Discriminator


