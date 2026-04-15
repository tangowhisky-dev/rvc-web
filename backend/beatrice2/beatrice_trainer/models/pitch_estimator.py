# extract_pitch_features, PitchEstimator.
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
from ..layers.attention import ConvNeXtStack
from ..io import dump_layer

def extract_pitch_features(
    y: torch.Tensor,  # [..., wav_length]
    hop_length: int = 160,  # 10ms
    win_length: int = 560,  # 35ms
    max_corr_period: int = 256,  # 16ms, 62.5Hz (16000 / 256)
    corr_win_length: int = 304,  # 19ms
    instfreq_features_cutoff_bin: int = 64,  # 1828Hz (16000 * 64 / 560)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert max_corr_period + corr_win_length == win_length

    # パディングする
    padding_length = (win_length - hop_length) // 2
    y = F.pad(y, (padding_length, padding_length))

    # フレームにする
    # [..., win_length, n_frames]
    y_frames = y.unfold(-1, win_length, hop_length).transpose_(-2, -1)

    # 複素スペクトログラム
    # Complex[..., (win_length // 2 + 1), n_frames]
    spec: torch.Tensor = torch.fft.rfft(y_frames, n=win_length, dim=-2)

    # Complex[..., instfreq_features_cutoff_bin, n_frames]
    spec = spec[..., :instfreq_features_cutoff_bin, :]

    # 対数パワースペクトログラム
    log_power_spec = spec.abs().add_(1e-5).log10_()

    # 瞬時位相の時間差分
    # 時刻 0 の値は 0
    delta_spec = spec[..., :, 1:] * spec[..., :, :-1].conj()
    delta_spec /= delta_spec.abs().add_(1e-5)
    delta_spec = torch.cat(
        [torch.zeros_like(delta_spec[..., :, :1]), delta_spec], dim=-1
    )

    # [..., instfreq_features_cutoff_bin * 3, n_frames]
    instfreq_features = torch.cat(
        [log_power_spec, delta_spec.real, delta_spec.imag], dim=-2
    )

    # 自己相関
    # 元々これに 2.0 / corr_win_length を掛けて使おうと思っていたが、
    # この値は振幅の 2 乗に比例していて、NN に入力するために良い感じに分散を
    # 標準化する方法が思いつかなかったのでやめた
    flipped_y_frames = y_frames.flip((-2,))
    a = torch.fft.rfft(flipped_y_frames, n=win_length, dim=-2)
    b = torch.fft.rfft(y_frames[..., -corr_win_length:, :], n=win_length, dim=-2)
    # [..., max_corr_period, n_frames]
    corr = torch.fft.irfft(a * b, n=win_length, dim=-2)[..., corr_win_length:, :]

    # エネルギー項
    energy = flipped_y_frames.square_().cumsum_(-2)
    energy0 = energy[..., corr_win_length - 1 : corr_win_length, :]
    energy = energy[..., corr_win_length:, :] - energy[..., :-corr_win_length, :]

    # Difference function
    corr_diff = (energy0 + energy).sub_(corr.mul_(2.0))
    assert corr_diff.min() >= -1e-3, corr_diff.min()
    corr_diff.clamp_(min=0.0)  # 計算誤差対策

    # 標準化
    corr_diff *= 2.0 / corr_win_length
    corr_diff.sqrt_()

    # 変換モデルへの入力用のエネルギー
    energy = (
        (y_frames * torch.signal.windows.cosine(win_length, device=y.device)[..., None])
        .square_()
        .sum(-2, keepdim=True)
    )

    energy.clamp_(min=1e-3).log10_()  # >= -3, 振幅 1 の正弦波なら大体 2.15
    energy *= 0.5  # >= -1.5, 振幅 1 の正弦波なら大体 1.07, 1 の差は振幅で 20dB の差

    return (
        instfreq_features,  # [..., instfreq_features_cutoff_bin * 3, n_frames]
        corr_diff,  # [..., max_corr_period, n_frames]
        energy,  # [..., 1, n_frames]
    )


class PitchEstimator(nn.Module):
    def __init__(
        self,
        input_instfreq_channels: int = 192,
        input_corr_channels: int = 256,
        pitch_bins: int = 448,
        channels: int = 192,
        intermediate_channels: int = 192 * 2,
        n_blocks: int = 9,
        delay: int = 1,  # 10ms, 特徴抽出と合わせると 22.5ms
        embed_kernel_size: int = 3,
        kernel_size: int = 33,
        pitch_bins_per_octave: int = 96,
    ):
        super().__init__()
        self.pitch_bins_per_octave = pitch_bins_per_octave

        self.instfreq_embed_0 = nn.Conv1d(input_instfreq_channels, channels, 1)
        self.instfreq_embed_1 = nn.Conv1d(channels, channels, 1)
        self.corr_embed_0 = nn.Conv1d(input_corr_channels, channels, 1)
        self.corr_embed_1 = nn.Conv1d(channels, channels, 1)
        self.backbone = ConvNeXtStack(
            channels,
            channels,
            intermediate_channels,
            n_blocks,
            delay,
            embed_kernel_size,
            kernel_size,
            enable_scaling=True,
        )
        self.head = nn.Conv1d(channels, pitch_bins, 1)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # wav: [batch_size, 1, wav_length]

        # [batch_size, input_instfreq_channels, length],
        # [batch_size, input_corr_channels, length]
        with torch.amp.autocast("cuda", enabled=False):
            instfreq_features, corr_diff, energy = extract_pitch_features(
                wav.squeeze(1),
                hop_length=160,
                win_length=560,
                max_corr_period=256,
                corr_win_length=304,
                instfreq_features_cutoff_bin=64,
            )
        instfreq_features = F.gelu(
            self.instfreq_embed_0(instfreq_features), approximate="tanh"
        )
        instfreq_features = self.instfreq_embed_1(instfreq_features)
        corr_diff = F.gelu(self.corr_embed_0(corr_diff), approximate="tanh")
        corr_diff = self.corr_embed_1(corr_diff)
        # [batch_size, channels, length]
        x = F.gelu(instfreq_features + corr_diff, approximate="tanh")
        x = self.backbone(x)
        # [batch_size, pitch_bins, length]
        x = self.head(x)
        return x, energy

    def sample_pitch(
        self, pitch: torch.Tensor, band_width: int = 4, return_features: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # pitch: [batch_size, pitch_bins, length]
        # 返されるピッチの値には 0 は含まれない
        batch_size, pitch_bins, length = pitch.size()
        pitch = pitch.softmax(1)
        if return_features:
            unvoiced_proba = pitch[:, :1, :].clone()
        pitch[:, 0, :] = -100.0
        pitch = (
            pitch.transpose(1, 2).contiguous().view(batch_size * length, 1, pitch_bins)
        )
        band_pitch = F.conv1d(
            pitch,
            torch.ones((1, 1, 1), device=pitch.device).expand(1, 1, band_width),
        )
        # [batch_size * length, 1, pitch_bins - band_width + 1] -> Long[batch_size * length, 1]
        quantized_band_pitch = band_pitch.argmax(2)
        if return_features:
            # [batch_size * length, 1]
            band_proba = band_pitch.gather(2, quantized_band_pitch[:, :, None])
            # [batch_size * length, 1]
            half_pitch_band_proba = band_pitch.gather(
                2,
                (quantized_band_pitch - self.pitch_bins_per_octave).clamp_(min=1)[
                    :, :, None
                ],
            )
            half_pitch_band_proba[
                quantized_band_pitch <= self.pitch_bins_per_octave
            ] = 0.0
            half_pitch_proba = (half_pitch_band_proba / (band_proba + 1e-6)).view(
                batch_size, 1, length
            )
            # [batch_size * length, 1]
            double_pitch_band_proba = band_pitch.gather(
                2,
                (quantized_band_pitch + self.pitch_bins_per_octave).clamp_(
                    max=pitch_bins - band_width
                )[:, :, None],
            )
            double_pitch_band_proba[
                quantized_band_pitch
                > pitch_bins - band_width - self.pitch_bins_per_octave
            ] = 0.0
            double_pitch_proba = (double_pitch_band_proba / (band_proba + 1e-6)).view(
                batch_size, 1, length
            )
        # Long[1, pitch_bins]
        mask = torch.arange(pitch_bins, device=pitch.device)[None, :]
        # bool[batch_size * length, pitch_bins]
        mask = (quantized_band_pitch <= mask) & (
            mask < quantized_band_pitch + band_width
        )
        # Long[batch_size, length]
        quantized_pitch = (pitch.squeeze(1) * mask).argmax(1).view(batch_size, length)

        if return_features:
            features = torch.cat(
                [unvoiced_proba, half_pitch_proba, double_pitch_proba], dim=1
            )
            # Long[batch_size, length], [batch_size, 3, length]
            return quantized_pitch, features
        else:
            return quantized_pitch

    def merge_weights(self):
        self.backbone.merge_weights()

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.instfreq_embed_0, f)
        dump_layer(self.instfreq_embed_1, f)
        dump_layer(self.corr_embed_0, f)
        dump_layer(self.corr_embed_1, f)
        dump_layer(self.backbone, f)
        dump_layer(self.head, f)


# %% [markdown]
# ## Vocoder


# %%
