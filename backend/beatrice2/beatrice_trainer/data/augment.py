# Training utilities: grad norm, F0, augmentation helpers.
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

def compute_grad_norm(
    model: nn.Module, return_stats: bool = False
) -> Union[float, dict[str, float]]:
    total_norm = 0.0
    stats = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm().item()
        if not math.isfinite(param_norm):
            param_norm = p.grad.data.float().norm().item()
        total_norm += param_norm * param_norm
        if return_stats:
            stats[f"grad_norm_{name}"] = param_norm
    total_norm = math.sqrt(total_norm)
    if return_stats:
        return total_norm, stats
    else:
        return total_norm


def compute_mean_f0(
    files: list[Path], method: Literal["dio", "harvest"] = "dio"
) -> float:
    sum_log_f0 = 0.0
    n_frames = 0
    for file in files:
        wav, sr = torchaudio.load(file)
        if method == "dio":
            f0, _ = pyworld.dio(wav.ravel().numpy().astype(np.float64), sr)
        elif method == "harvest":
            f0, _ = pyworld.harvest(wav.ravel().numpy().astype(np.float64), sr)
        else:
            raise ValueError(f"Invalid method: {method}")
        f0 = f0[f0 > 0]
        sum_log_f0 += float(np.log(f0).sum())
        n_frames += len(f0)
    if n_frames == 0:
        return math.nan
    mean_log_f0 = sum_log_f0 / n_frames
    return math.exp(mean_log_f0)


# %% [markdown]
# ## Dataset


# %%
def get_resampler(
    sr_before: int, sr_after: int, device="cpu", cache={}
) -> torchaudio.transforms.Resample:
    if not isinstance(device, str):
        device = str(device)
    if (sr_before, sr_after, device) not in cache:
        cache[(sr_before, sr_after, device)] = torchaudio.transforms.Resample(
            sr_before, sr_after
        ).to(device)
    return cache[(sr_before, sr_after, device)]


def convolve(signal: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    n = 1 << (signal.size(-1) + ir.size(-1) - 2).bit_length()
    res = torch.fft.irfft(torch.fft.rfft(signal, n=n) * torch.fft.rfft(ir, n=n), n=n)
    return res[..., : signal.size(-1)]


def random_formant_shift(
    wav: torch.Tensor,
    sample_rate: int,
    formant_shift_semitone_min: float = -3.0,
    formant_shift_semitone_max: float = 3.0,
) -> torch.Tensor:
    assert wav.ndim == 2
    assert wav.size(0) == 1

    device = wav.device

    hop_length = 256

    # [wav_length]
    wav_np = wav.ravel().double().cpu().numpy()
    f0, t = pyworld.dio(
        wav_np,
        sample_rate,
        f0_floor=55,
        f0_ceil=1400,
        frame_period=hop_length * 1000 / sample_rate,
    )
    f0 = pyworld.stonemask(wav_np, f0, t, sample_rate)
    world_sp = pyworld.cheaptrick(wav_np, f0, t, sample_rate)
    world_sp = (
        torch.from_numpy(world_sp).float().to(device).sqrt_()[None]
    )  # [1, length, n_fft // 2 + 1]

    n_fft = win_length = (world_sp.size(2) - 1) * 2

    window = torch.hann_window(win_length, device=device)

    # [1, n_fft // 2 + 1, length]
    stft_sp = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    assert world_sp.size(1) == stft_sp.size(2), (world_sp.size(), stft_sp.size())
    assert world_sp.size(2) == stft_sp.size(1), (world_sp.size(), stft_sp.size())

    shift_semitones = (
        torch.rand(()).item()
        * (formant_shift_semitone_max - formant_shift_semitone_min)
        + formant_shift_semitone_min
    )
    shift_ratio = 2.0 ** (shift_semitones / 12.0)
    shifted_world_sp = F.interpolate(
        world_sp, scale_factor=shift_ratio, mode="linear", align_corners=True
    )

    if shifted_world_sp.size(2) > n_fft // 2 + 1:
        shifted_world_sp = shifted_world_sp[:, :, : n_fft // 2 + 1]
    elif shifted_world_sp.size(2) < n_fft // 2 + 1:
        shifted_world_sp = F.pad(
            shifted_world_sp, (0, n_fft // 2 + 1 - shifted_world_sp.size(2))
        )

    ratio = ((shifted_world_sp + 1e-5) / (world_sp + 1e-5)).clamp(0.1, 10.0)
    stft_sp *= ratio.transpose(-2, -1)  # [1, n_fft // 2 + 1, length]

    out = torch.istft(
        stft_sp,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=wav.size(-1),
    )

    return out


def random_filter(audio: torch.Tensor) -> torch.Tensor:
    assert audio.ndim == 2
    ab = torch.rand(audio.size(0), 6) * 0.75 - 0.375
    a, b = ab[:, :3], ab[:, 3:]
    a[:, 0] = 1.0
    b[:, 0] = 1.0
    audio = torchaudio.functional.lfilter(audio, a, b, clamp=False)
    return audio


def get_noise(
    n_samples: int, sample_rate: float, files: list[Union[str, bytes, os.PathLike]]
) -> torch.Tensor:
    resample_augmentation_candidates = [0.9, 0.95, 1.0, 1.05, 1.1]
    wavs = []
    current_length = 0
    while current_length < n_samples:
        idx_files = torch.randint(0, len(files), ())
        file = files[idx_files]
        wav, sr = torchaudio.load(file)
        assert wav.size(0) == 1
        augmented_sample_rate = int(
            round(
                sample_rate
                * resample_augmentation_candidates[
                    torch.randint(0, len(resample_augmentation_candidates), ())
                ]
            )
        )
        resampler = get_resampler(sr, augmented_sample_rate)
        wav = resampler(wav)
        wav = random_filter(wav)
        wav *= 0.99 / (wav.abs().max() + 1e-5)
        wavs.append(wav)
        current_length += wav.size(1)
    start = torch.randint(0, current_length - n_samples + 1, ())
    wav = torch.cat(wavs, dim=1)[:, start : start + n_samples]
    assert wav.size() == (1, n_samples), wav.size()
    return wav


def get_butterworth_lpf(
    cutoff_freq: float, sample_rate: int, cache={}
) -> tuple[torch.Tensor, torch.Tensor]:
    if (cutoff_freq, sample_rate) not in cache:
        q = math.sqrt(0.5)
        omega = math.tau * cutoff_freq / sample_rate
        cos_omega = math.cos(omega)
        alpha = math.sin(omega) / (2.0 * q)
        b1 = (1.0 - cos_omega) / (1.0 + alpha)
        b0 = b1 * 0.5
        a1 = -2.0 * cos_omega / (1.0 + alpha)
        a2 = (1.0 - alpha) / (1.0 + alpha)
        cache[(cutoff_freq, sample_rate)] = (
            torch.tensor([b0, b1, b0]),
            torch.tensor([1.0, a1, a2]),
        )
    return cache[(cutoff_freq, sample_rate)]


def augment_audio(
    clean: torch.Tensor,
    sample_rate: int,
    noise_files: list[Union[str, bytes, os.PathLike]],
    ir_files: list[Union[str, bytes, os.PathLike]],
    snr_candidates: list[float] = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
    formant_shift_probability: float = 0.5,
    formant_shift_semitone_min: float = -3.0,
    formant_shift_semitone_max: float = 3.0,
    reverb_probability: float = 0.5,
    lpf_probability: float = 0.2,
    lpf_cutoff_freq_candidates: list[float] = [2000.0, 3000.0, 4000.0, 6000.0],
) -> torch.Tensor:
    # [1, wav_length]
    assert clean.size(0) == 1
    n_samples = clean.size(1)

    original_clean_rms = clean.square().mean().sqrt_()

    # clean をフォルマントシフトする
    if torch.rand(()) < formant_shift_probability:
        clean = random_formant_shift(
            clean, sample_rate, formant_shift_semitone_min, formant_shift_semitone_max
        )

    # noise を取得して clean と concat する
    noise = get_noise(n_samples, sample_rate, noise_files)
    signals = torch.cat([clean, noise])

    # clean, noise に異なるランダムフィルタをかける
    signals = random_filter(signals)

    # clean, noise にリバーブをかける
    if torch.rand(()) < reverb_probability:
        ir_file = ir_files[torch.randint(0, len(ir_files), ())]
        ir, sr = torchaudio.load(ir_file)
        assert ir.size() == (2, sr), ir.size()
        assert sr == sample_rate, (sr, sample_rate)
        signals = convolve(signals, ir)

    # clean, noise に同じ LPF をかける
    if torch.rand(()) < lpf_probability:
        if signals.abs().max() > 0.8:
            signals /= signals.abs().max() * 1.25
        cutoff_freq = lpf_cutoff_freq_candidates[
            torch.randint(0, len(lpf_cutoff_freq_candidates), ())
        ]
        b, a = get_butterworth_lpf(cutoff_freq, sample_rate)
        signals = torchaudio.functional.lfilter(signals, a, b, clamp=False)

    # clean の音量を合わせる
    clean, noise = signals
    clean_rms = clean.square().mean().sqrt_()
    clean *= original_clean_rms / clean_rms

    if len(snr_candidates) >= 1:
        # clean, noise の音量をピークを重視して取る
        clean_level = clean.square().square_().mean().sqrt_().sqrt_()
        noise_level = noise.square().square_().mean().sqrt_().sqrt_()
        # SNR
        snr = snr_candidates[torch.randint(0, len(snr_candidates), ())]
        # noisy を生成
        noisy = clean + noise * (
            0.1 ** (snr / 20.0) * clean_level / (noise_level + 1e-5)
        )

    return noisy


