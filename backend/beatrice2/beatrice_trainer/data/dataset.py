# WavDataset and collate helpers.
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
from ..models.vocoder import slice_segments
from .augment import get_resampler, augment_audio

class WavDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_files: list[tuple[Path, int]],
        in_sample_rate: int = 16000,
        out_sample_rate: int = 24000,
        wav_length: int = 4 * 24000,  # 4s
        segment_length: int = 100,  # 1s
        noise_files: Optional[list[Union[str, bytes, os.PathLike]]] = None,
        ir_files: Optional[list[Union[str, bytes, os.PathLike]]] = None,
        augmentation_snr_candidates: list[float] = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
        augmentation_formant_shift_probability: float = 0.5,
        augmentation_formant_shift_semitone_min: float = -3.0,
        augmentation_formant_shift_semitone_max: float = 3.0,
        augmentation_reverb_probability: float = 0.5,
        augmentation_lpf_probability: float = 0.2,
        augmentation_lpf_cutoff_freq_candidates: list[float] = [
            2000.0,
            3000.0,
            4000.0,
            6000.0,
        ],
        # Reference encoder support: when True, collate() returns a ref clip per sample.
        # The ref clip is a *different* audio file from the same speaker, loaded at
        # 16 kHz (the reference encoder's expected sample rate).
        use_reference_encoder: bool = False,
        ref_clip_seconds: float = 6.0,
    ):
        self.audio_files = audio_files
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.wav_length = wav_length
        self.segment_length = segment_length
        self.noise_files = noise_files
        self.ir_files = ir_files
        self.augmentation_snr_candidates = augmentation_snr_candidates
        self.augmentation_formant_shift_probability = (
            augmentation_formant_shift_probability
        )
        self.augmentation_formant_shift_semitone_min = (
            augmentation_formant_shift_semitone_min
        )
        self.augmentation_formant_shift_semitone_max = (
            augmentation_formant_shift_semitone_max
        )
        self.augmentation_reverb_probability = augmentation_reverb_probability
        self.augmentation_lpf_probability = augmentation_lpf_probability
        self.augmentation_lpf_cutoff_freq_candidates = (
            augmentation_lpf_cutoff_freq_candidates
        )

        if (noise_files is None) is not (ir_files is None):
            raise ValueError("noise_files and ir_files must be both None or not None")

        self.in_hop_length = in_sample_rate // 100
        self.out_hop_length = out_sample_rate // 100  # 10ms 刻み

        # Reference encoder support
        self.use_reference_encoder = use_reference_encoder
        self.ref_clip_samples = int(ref_clip_seconds * in_sample_rate)  # at 16 kHz

        # Build a per-speaker file list for fast reference sampling.
        # Maps speaker_id → sorted list of dataset indices.
        if use_reference_encoder:
            from collections import defaultdict as _dd
            _spk_to_idx: dict[int, list[int]] = _dd(list)
            for i, (_, spk) in enumerate(audio_files):
                _spk_to_idx[spk].append(i)
            self._spk_to_idx: dict[int, list[int]] = dict(_spk_to_idx)
        else:
            self._spk_to_idx = {}

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, int, Optional[torch.Tensor]]:
        file, speaker_id = self.audio_files[index]
        clean_wav, sample_rate = torchaudio.load(file)
        if clean_wav.size(0) != 1:
            ch = torch.randint(0, clean_wav.size(0), ())
            clean_wav = clean_wav[ch : ch + 1]

        formant_shift_candidates = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        formant_shift = formant_shift_candidates[
            torch.randint(0, len(formant_shift_candidates), ()).item()
        ]

        resampler_fraction = Fraction(
            sample_rate / self.out_sample_rate * 2.0 ** (formant_shift / 12.0)
        ).limit_denominator(300)
        clean_wav = get_resampler(
            resampler_fraction.numerator, resampler_fraction.denominator
        )(clean_wav)

        assert clean_wav.size(0) == 1
        assert clean_wav.size(1) != 0

        clean_wav = F.pad(clean_wav, (self.wav_length, self.wav_length))

        if self.noise_files is None:
            assert False
            noisy_wav_16k = get_resampler(self.out_sample_rate, self.in_sample_rate)(
                clean_wav
            )
        else:
            clean_wav_16k = get_resampler(self.out_sample_rate, self.in_sample_rate)(
                clean_wav
            )
            noisy_wav_16k = augment_audio(
                clean_wav_16k,
                self.in_sample_rate,
                self.noise_files,
                self.ir_files,
                self.augmentation_snr_candidates,
                self.augmentation_formant_shift_probability,
                self.augmentation_formant_shift_semitone_min,
                self.augmentation_formant_shift_semitone_max,
                self.augmentation_reverb_probability,
                self.augmentation_lpf_probability,
                self.augmentation_lpf_cutoff_freq_candidates,
            )

        clean_wav = clean_wav.squeeze_(0)
        noisy_wav_16k = noisy_wav_16k.squeeze_(0)

        # 音量をランダマイズする
        amplitude = torch.rand(()).item() * 0.899 + 0.1
        factor = amplitude / clean_wav.abs().max()
        clean_wav *= factor
        noisy_wav_16k *= factor
        while noisy_wav_16k.abs().max() >= 1.0:
            clean_wav *= 0.5
            noisy_wav_16k *= 0.5

        # Load reference clip for reference encoder (different file, same speaker)
        ref_wav_16k = self._load_ref_wav(index, speaker_id)

        return clean_wav, noisy_wav_16k, speaker_id, formant_shift, ref_wav_16k, ref_wav_16k

    def _load_ref_wav(self, index: int, speaker_id: int) -> Optional[torch.Tensor]:
        """Load a reference clip at 16 kHz from a *different* file of the same speaker.

        If the speaker has only one file, uses a different segment of that file.
        Returns None if use_reference_encoder is False.
        """
        if not self.use_reference_encoder:
            return None

        same_speaker_indices = self._spk_to_idx.get(speaker_id, [index])
        # Try to find a different file index to prevent leakage
        candidates = [i for i in same_speaker_indices if i != index]
        if not candidates:
            # Single-file speaker: use the same file, different random segment
            candidates = same_speaker_indices

        ref_idx = candidates[torch.randint(0, len(candidates), ()).item()]
        ref_file, _ = self.audio_files[ref_idx]
        try:
            wav, sr = torchaudio.load(ref_file)
        except Exception:
            return None

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to 16 kHz
        if sr != self.in_sample_rate:
            wav = get_resampler(sr, self.in_sample_rate)(wav)

        wav = wav.squeeze(0)  # [T]

        # Trim or repeat to exactly ref_clip_samples
        if wav.shape[0] >= self.ref_clip_samples:
            # Random start within the file
            max_start = wav.shape[0] - self.ref_clip_samples
            start = torch.randint(0, max_start + 1, ()).item()
            wav = wav[start : start + self.ref_clip_samples]
        else:
            # Repeat to fill
            repeats = (self.ref_clip_samples + wav.shape[0] - 1) // wav.shape[0]
            wav = wav.repeat(repeats)[: self.ref_clip_samples]

        # Normalise amplitude
        peak = wav.abs().max()
        if peak > 1e-6:
            wav = wav / peak * 0.9

        return wav  # [ref_clip_samples]

    def __len__(self) -> int:
        return len(self.audio_files)

    def collate(
        self, batch: list[tuple[torch.Tensor, torch.Tensor, int, int, Optional[torch.Tensor]]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert self.wav_length % self.out_hop_length == 0
        length = self.wav_length // self.out_hop_length
        clean_wavs = []
        noisy_wavs = []
        slice_starts = []
        speaker_ids = []
        formant_shifts = []
        ref_wavs = []
        for clean_wav, noisy_wav, speaker_id, formant_shift, ref_wav in batch:
            # 発声部分をランダムに 1 箇所選ぶ
            (voiced,) = clean_wav.nonzero(as_tuple=True)
            assert voiced.numel() != 0
            center = voiced[torch.randint(0, voiced.numel(), ()).item()].item()
            # 発声部分が中央にくるように、スライス区間を選ぶ
            slice_start = center - self.segment_length * self.out_hop_length // 2
            assert slice_start >= 0
            # スライス区間が含まれるように、ランダムに wav_length の長さを切り出す
            r = torch.randint(0, length - self.segment_length + 1, ()).item()
            offset = slice_start - r * self.out_hop_length
            clean_wavs.append(clean_wav[offset : offset + self.wav_length])
            offset_in_sample_rate = int(
                round(offset * self.in_sample_rate / self.out_sample_rate)
            )
            noisy_wavs.append(
                noisy_wav[
                    offset_in_sample_rate : offset_in_sample_rate
                    + length * self.in_hop_length
                ]
            )
            slice_start = r
            slice_starts.append(slice_start)
            speaker_ids.append(speaker_id)
            formant_shifts.append(formant_shift)
            if ref_wav is not None:
                ref_wavs.append(ref_wav)
        clean_wavs = torch.stack(clean_wavs)
        noisy_wavs = torch.stack(noisy_wavs)
        slice_starts = torch.tensor(slice_starts)
        speaker_ids = torch.tensor(speaker_ids)
        formant_shifts = torch.tensor(formant_shifts)
        # ref_wavs: [batch, 1, ref_clip_samples] if all present, else None
        ref_wavs_tensor: Optional[torch.Tensor] = None
        if len(ref_wavs) == len(batch):
            ref_wavs_tensor = torch.stack(ref_wavs).unsqueeze(1)  # [B, 1, T_ref]
        return (
            clean_wavs,  # [batch_size, wav_length]
            noisy_wavs,  # [batch_size, wav_length]
            slice_starts,  # Long[batch_size]
            speaker_ids,  # Long[batch_size]
            formant_shifts,  # Long[batch_size]
            ref_wavs_tensor,  # [batch_size, 1, T_ref] or None
        )


# %% [markdown]
# ## Train

# %%
AUDIO_FILE_SUFFIXES = {
    ".wav",
    ".aif",
    ".aiff",
    ".fla",
    ".flac",
    ".oga",
    ".ogg",
    ".opus",
    ".mp3",
}


