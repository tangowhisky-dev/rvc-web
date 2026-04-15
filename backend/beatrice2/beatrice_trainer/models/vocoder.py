# D4C synthesis utilities, Vocoder, compute_loudness, slice_segments.
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
from ..io import dump_layer

def overlap_add(
    ir_amp: torch.Tensor,
    ir_phase: torch.Tensor,
    window: torch.Tensor,
    pitch: torch.Tensor,
    hop_length: int = 240,
    delay: int = 0,
    sr: float = 24000.0,
) -> torch.Tensor:
    batch_size, ir_length, length = ir_amp.size()
    ir_length = (ir_length - 1) * 2
    assert ir_phase.size() == ir_amp.size()
    assert window.size() == (ir_length,), (window.size(), ir_amp.size())
    assert pitch.size() == (batch_size, length * hop_length)
    assert 0 <= delay < ir_length, (delay, ir_length)
    # 正規化角周波数 [2π rad]
    normalized_freq = pitch / sr
    # 初期位相 [2π rad] をランダムに設定
    normalized_freq[:, 0] = torch.rand(batch_size, device=pitch.device)
    with torch.amp.autocast("cuda", enabled=False):
        phase = (normalized_freq.double().cumsum_(1) % 1.0).float()
    # 重ねる箇所を求める
    # [n_pitchmarks], [n_pitchmarks]
    indices0, indices1 = torch.nonzero(phase[:, :-1] > phase[:, 1:], as_tuple=True)
    # 重ねる箇所の小数部分 (位相の遅れ) を求める
    numer = 1.0 - phase[indices0, indices1]
    # [n_pitchmarks]
    fractional_part = numer / (numer + phase[indices0, indices1 + 1])
    # 重ねる値を求める
    # Complex[n_pitchmarks, ir_length / 2 + 1]
    ir_amp = ir_amp[indices0, :, indices1 // hop_length]
    ir_phase = ir_phase[indices0, :, indices1 // hop_length]
    # 位相遅れの量 [rad]
    # [n_pitchmarks, ir_length / 2 + 1]
    delay_phase = (
        torch.arange(ir_length // 2 + 1, device=pitch.device, dtype=torch.float32)[
            None, :
        ]
        * (-math.tau / ir_length)
        * fractional_part[:, None]
    )
    # Complex[n_pitchmarks, ir_length / 2 + 1]
    spec = torch.polar(ir_amp, ir_phase + delay_phase)
    # [n_pitchmarks, ir_length]
    ir = torch.fft.irfft(spec, n=ir_length, dim=1)
    ir *= window

    # 加算する値をサンプル単位にばらす
    # [n_pitchmarks * ir_length]
    ir = ir.ravel()
    # Long[n_pitchmarks * ir_length]
    indices0 = indices0[:, None].expand(-1, ir_length).ravel()
    # Long[n_pitchmarks * ir_length]
    indices1 = (
        indices1[:, None] + torch.arange(ir_length, device=pitch.device)
    ).ravel()

    # overlap-add する
    overlap_added_signal = torch.zeros(
        (batch_size, length * hop_length + ir_length), device=pitch.device
    )
    overlap_added_signal.index_put_((indices0, indices1), ir, accumulate=True)
    overlap_added_signal = overlap_added_signal[:, delay : -ir_length + delay]

    return overlap_added_signal


def generate_noise(
    aperiodicity: torch.Tensor, delay: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    # aperiodicity: [batch_size, hop_length, length]
    batch_size, hop_length, length = aperiodicity.size()
    excitation = torch.rand(
        batch_size, (length + 1) * hop_length, device=aperiodicity.device
    )
    excitation -= 0.5
    n_fft = 2 * hop_length
    # 矩形窓で分析
    # Complex[batch_size, hop_length + 1, length]
    noise = torch.stft(
        excitation,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.ones(n_fft, device=excitation.device),
        center=False,
        return_complex=True,
    )
    assert noise.size(2) == aperiodicity.size(2)
    noise[:, 0, :] = 0.0
    noise[:, 1:, :] *= aperiodicity
    # ハン窓で合成
    # torch.istft は最適合成窓が使われるので使えないことに注意
    # [batch_size, 2 * hop_length, length]
    noise = torch.fft.irfft(noise, n=2 * hop_length, dim=1)
    noise *= torch.hann_window(2 * hop_length, device=noise.device)[None, :, None]
    # [batch_size, (length + 1) * hop_length]
    noise = F.fold(
        noise,
        (1, (length + 1) * hop_length),
        (1, 2 * hop_length),
        stride=(1, hop_length),
    ).squeeze_((1, 2))

    assert delay < hop_length
    noise = noise[:, delay : -hop_length + delay]
    excitation = excitation[:, delay : -hop_length + delay]
    return noise, excitation  # [batch_size, length * hop_length]


D4C_PREVENT_ZERO_DIVISION = True  # False にすると本家の処理


def interp(x: torch.Tensor, y: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    # x が単調増加で等間隔と仮定
    # 外挿は起こらないと仮定
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    xi = torch.as_tensor(xi)
    if xi.ndim < y.ndim:
        diff_ndim = y.ndim - xi.ndim
        xi = xi.view(tuple([1] * diff_ndim) + xi.size())
    if xi.size()[:-1] != y.size()[:-1]:
        xi = xi.expand(y.size()[:-1] + (xi.size(-1),))
    assert (x.min(-1).values == x[..., 0]).all()
    assert (x.max(-1).values == x[..., -1]).all()
    assert (xi.min(-1).values >= x[..., 0]).all()
    assert (xi.max(-1).values <= x[..., -1]).all()
    delta_x = (x[..., -1].double() - x[..., 0].double()) / (x.size(-1) - 1.0)
    delta_x = delta_x.to(x.dtype)
    xi = (xi - x[..., :1]).div_(delta_x[..., None])
    xi_base = xi.floor()
    xi_fraction = xi.sub_(xi_base)
    xi_base = xi_base.long()
    delta_y = y.diff(dim=-1, append=y[..., -1:])
    yi = y.gather(-1, xi_base) + delta_y.gather(-1, xi_base) * xi_fraction
    return yi


def linear_smoothing(
    group_delay: torch.Tensor, sr: int, n_fft: int, width: torch.Tensor
) -> torch.Tensor:
    group_delay = torch.as_tensor(group_delay)
    assert group_delay.size(-1) == n_fft // 2 + 1
    width = torch.as_tensor(width)
    boundary = (width.max() * n_fft / sr).long() + 1

    dtype = group_delay.dtype
    device = group_delay.device
    fft_resolution = sr / n_fft
    mirroring_freq_axis = (
        torch.arange(-boundary, n_fft // 2 + 1 + boundary, dtype=dtype, device=device)
        .add(0.5)
        .mul(fft_resolution)
    )
    if group_delay.ndim == 1:
        mirroring_spec = F.pad(
            group_delay[None], (boundary, boundary), mode="reflect"
        ).squeeze_(0)
    elif group_delay.ndim >= 4:
        shape = group_delay.size()
        mirroring_spec = F.pad(
            group_delay.view(math.prod(shape[:-1]), group_delay.size(-1)),
            (boundary, boundary),
            mode="reflect",
        ).view(shape[:-1] + (shape[-1] + 2 * boundary,))
    else:
        mirroring_spec = F.pad(group_delay, (boundary, boundary), mode="reflect")
    mirroring_segment = mirroring_spec.mul(fft_resolution).cumsum_(-1)
    center_freq = torch.arange(n_fft // 2 + 1, dtype=dtype, device=device).mul_(
        fft_resolution
    )
    low_freq = center_freq - width[..., None] * 0.5
    high_freq = center_freq + width[..., None] * 0.5
    levels = interp(
        mirroring_freq_axis, mirroring_segment, torch.cat([low_freq, high_freq], dim=-1)
    )
    low_levels, high_levels = levels.split([n_fft // 2 + 1] * 2, dim=-1)
    smoothed = (high_levels - low_levels).div_(width[..., None])
    return smoothed


def dc_correction(
    spec: torch.Tensor, sr: int, n_fft: int, f0: torch.Tensor
) -> torch.Tensor:
    spec = torch.as_tensor(spec)
    f0 = torch.as_tensor(f0)
    dtype = spec.dtype
    device = spec.device

    upper_limit = 2 + (f0 * (n_fft / sr)).long()
    max_upper_limit = upper_limit.max()
    upper_limit_mask = (
        torch.arange(max_upper_limit - 1, device=device) < (upper_limit - 1)[..., None]
    )
    low_freq_axis = torch.arange(max_upper_limit + 1, dtype=dtype, device=device) * (
        sr / n_fft
    )
    low_freq_replica = interp(
        f0[..., None] - low_freq_axis.flip(-1),
        spec[..., : max_upper_limit + 1].flip(-1),
        low_freq_axis[..., : max_upper_limit - 1] * upper_limit_mask,
    )
    output = spec.clone()
    output[..., : max_upper_limit - 1] += low_freq_replica * upper_limit_mask
    return output


def nuttall(n: int, device: torch.types.Device) -> torch.Tensor:
    t = torch.linspace(0, math.tau, n, device=device)
    coefs = torch.tensor([0.355768, -0.487396, 0.144232, -0.012604], device=device)
    terms = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
    cos_matrix = (terms[:, None] * t).cos_()  # [4, n]
    window = coefs.matmul(cos_matrix)
    return window


def get_windowed_waveform(
    x: torch.Tensor,
    sr: int,
    f0: torch.Tensor,
    position: torch.Tensor,
    half_window_length_ratio: float,
    window_type: Literal["hann", "blackman"],
    n_fft: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(x)
    f0 = torch.as_tensor(f0)
    position = torch.as_tensor(position)

    current_sample = position * sr
    # [...]
    half_window_length = (half_window_length_ratio * sr / f0).add_(0.5).long()
    # [..., fft_size]
    base_index = -half_window_length[..., None] + torch.arange(n_fft, device=x.device)
    base_index_mask = base_index <= half_window_length[..., None]
    # [..., fft_size]
    safe_index = ((current_sample + 0.501).long()[..., None] + base_index).clamp_(
        0, x.size(-1) - 1
    )
    # [..., fft_size]
    time_axis = base_index.to(x.dtype).div_(half_window_length_ratio)
    # [...]
    normalized_f0 = math.pi / sr * f0
    # [..., fft_size]
    phase = time_axis.mul_(normalized_f0[..., None])

    if window_type == "hann":
        window = phase.cos_().mul_(0.5).add_(0.5)
    elif window_type == "blackman":
        window = phase.mul(2.0).cos_().mul_(0.08).add_(phase.cos().mul_(0.5)).add_(0.42)
    else:
        assert False
    window *= base_index_mask

    prefix_shape = tuple(
        max(x_size, i_size) for x_size, i_size in zip(x.size(), safe_index.size())
    )[:-1]
    waveform = (
        x.expand(prefix_shape + (-1,))
        .gather(-1, safe_index.expand(prefix_shape + (-1,)))
        .mul_(window)
    )
    if not D4C_PREVENT_ZERO_DIVISION:
        waveform += torch.randn_like(window).mul_(1e-12)
    waveform *= base_index_mask
    waveform -= window * waveform.sum(-1, keepdim=True).div_(
        window.sum(-1, keepdim=True)
    )
    return waveform, window


def get_centroid(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    x = torch.as_tensor(x)
    if D4C_PREVENT_ZERO_DIVISION:
        x = x / x.norm(dim=-1, keepdim=True).clamp(min=6e-8)
    else:
        x = x / x.norm(dim=-1, keepdim=True)
    spec0 = torch.fft.rfft(x, n_fft)
    spec1 = torch.fft.rfft(
        x * torch.arange(1, x.size(-1) + 1, dtype=x.dtype, device=x.device).div_(n_fft),
        n_fft,
    )
    centroid = spec0.real * spec1.real + spec0.imag * spec1.imag
    return centroid


def get_static_centroid(
    x: torch.Tensor, sr: int, f0: torch.Tensor, position: torch.Tensor, n_fft: int
) -> torch.Tensor:
    """First step: calculation of temporally static parameters on basis of group delay"""
    x1, _ = get_windowed_waveform(
        x, sr, f0, position + 0.25 / f0, 2.0, "blackman", n_fft
    )
    x2, _ = get_windowed_waveform(
        x, sr, f0, position - 0.25 / f0, 2.0, "blackman", n_fft
    )
    centroid1 = get_centroid(x1, n_fft)
    centroid2 = get_centroid(x2, n_fft)
    return dc_correction(centroid1 + centroid2, sr, n_fft, f0)


def get_smoothed_power_spec(
    x: torch.Tensor, sr: int, f0: torch.Tensor, position: torch.Tensor, n_fft: int
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(x)
    f0 = torch.as_tensor(f0)
    x, window = get_windowed_waveform(x, sr, f0, position, 2.0, "hann", n_fft)
    window_weight = window.square().sum(-1, keepdim=True)
    rms = x.square().sum(-1, keepdim=True).div_(window_weight).sqrt_()
    if D4C_PREVENT_ZERO_DIVISION:
        x = x / (rms * math.sqrt(n_fft)).clamp_(min=6e-8)
    smoothed_power_spec = torch.fft.rfft(x, n_fft).abs().square_()
    smoothed_power_spec = dc_correction(smoothed_power_spec, sr, n_fft, f0)
    smoothed_power_spec = linear_smoothing(smoothed_power_spec, sr, n_fft, f0)
    return smoothed_power_spec, rms.detach().squeeze(-1)


def get_static_group_delay(
    static_centroid: torch.Tensor,
    smoothed_power_spec: torch.Tensor,
    sr: int,
    f0: torch.Tensor,
    n_fft: int,
) -> torch.Tensor:
    """Second step: calculation of parameter shaping"""
    if D4C_PREVENT_ZERO_DIVISION:
        smoothed_power_spec = smoothed_power_spec.clamp(min=6e-8)
    static_group_delay = static_centroid / smoothed_power_spec  # t_g
    static_group_delay = linear_smoothing(
        static_group_delay, sr, n_fft, f0 * 0.5
    )  # t_gs
    smoothed_group_delay = linear_smoothing(static_group_delay, sr, n_fft, f0)  # t_gb
    static_group_delay = static_group_delay - smoothed_group_delay  # t_D
    return static_group_delay


def get_coarse_aperiodicity(
    group_delay: torch.Tensor,
    sr: int,
    n_fft: int,
    freq_interval: int,
    n_aperiodicities: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """Third step: estimation of band-aperiodicity"""
    group_delay = torch.as_tensor(group_delay)
    window = torch.as_tensor(window)
    boundary = int(round(n_fft * 8 / window.size(-1)))
    half_window_length = window.size(-1) // 2
    coarse_aperiodicity = torch.empty(
        group_delay.size()[:-1] + (n_aperiodicities,),
        dtype=group_delay.dtype,
        device=group_delay.device,
    )
    for i in range(n_aperiodicities):
        center = freq_interval * (i + 1) * n_fft // sr
        segment = (
            group_delay[
                ..., center - half_window_length : center + half_window_length + 1
            ]
            * window
        )
        power_spec: torch.Tensor = torch.fft.rfft(segment, n_fft).abs().square_()
        cumulative_power_spec = power_spec.sort(-1).values.cumsum_(-1)
        if D4C_PREVENT_ZERO_DIVISION:
            cumulative_power_spec.clamp_(min=6e-8)
        coarse_aperiodicity[..., i] = (
            cumulative_power_spec[..., n_fft // 2 - boundary - 1]
            / cumulative_power_spec[..., -1]
        )
    coarse_aperiodicity.log10_().mul_(10.0)
    return coarse_aperiodicity


def d4c_love_train(
    x: torch.Tensor, sr: int, f0: torch.Tensor, position: torch.Tensor, threshold: float
) -> int:
    x = torch.as_tensor(x)
    position = torch.as_tensor(position)
    f0: torch.Tensor = torch.as_tensor(f0)
    vuv = f0 != 0
    lowest_f0 = 40
    f0 = f0.clamp(min=lowest_f0)
    n_fft = 1 << (3 * sr // lowest_f0).bit_length()
    boundary0 = (100 * n_fft - 1) // sr + 1
    boundary1 = (4000 * n_fft - 1) // sr + 1
    boundary2 = (7900 * n_fft - 1) // sr + 1
    waveform, _ = get_windowed_waveform(x, sr, f0, position, 1.5, "blackman", n_fft)
    power_spec = torch.fft.rfft(waveform, n_fft).abs().square_()
    power_spec[..., : boundary0 + 1] = 0.0
    cumulative_spec = power_spec.cumsum_(-1)
    vuv = vuv & (
        cumulative_spec[..., boundary1] > threshold * cumulative_spec[..., boundary2]
    )
    return vuv


def d4c_general_body(
    x: torch.Tensor,
    sr: int,
    f0: torch.Tensor,
    freq_interval: int,
    position: torch.Tensor,
    n_fft: int,
    n_aperiodicities: int,
    window: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    static_centroid = get_static_centroid(x, sr, f0, position, n_fft)
    smoothed_power_spec, rms = get_smoothed_power_spec(x, sr, f0, position, n_fft)
    static_group_delay = get_static_group_delay(
        static_centroid, smoothed_power_spec, sr, f0, n_fft
    )
    coarse_aperiodicity = get_coarse_aperiodicity(
        static_group_delay, sr, n_fft, freq_interval, n_aperiodicities, window
    )
    coarse_aperiodicity.add_((f0[..., None] - 100.0).div_(50.0)).clamp_(max=0.0)
    return coarse_aperiodicity, rms


def d4c(
    x: torch.Tensor,
    f0: torch.Tensor,
    t: torch.Tensor,
    sr: int,
    threshold: float = 0.85,
    n_fft_spec: Optional[int] = None,
    coarse_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/tuanad121/Python-WORLD/blob/master/world/d4c.py"""
    FLOOR_F0 = 71
    FLOOR_F0_D4C = 47
    UPPER_LIMIT = 15000
    FREQ_INTERVAL = 3000

    assert sr == int(sr)
    sr = int(sr)
    assert sr % 2 == 0
    x = torch.as_tensor(x)
    f0 = torch.as_tensor(f0)
    temporal_positions = torch.as_tensor(t)

    n_fft_d4c = 1 << (4 * sr // FLOOR_F0_D4C).bit_length()
    if n_fft_spec is None:
        n_fft_spec = 1 << (3 * sr // FLOOR_F0).bit_length()
    n_aperiodicities = min(UPPER_LIMIT, sr // 2 - FREQ_INTERVAL) // FREQ_INTERVAL
    assert n_aperiodicities >= 1
    window_length = FREQ_INTERVAL * n_fft_d4c // sr * 2 + 1
    window = nuttall(window_length, device=x.device)
    freq_axis = torch.arange(n_fft_spec // 2 + 1, device=x.device) * (sr / n_fft_spec)

    coarse_aperiodicity, rms = d4c_general_body(
        x[..., None, :],
        sr,
        f0.clamp(min=FLOOR_F0_D4C),
        FREQ_INTERVAL,
        temporal_positions,
        n_fft_d4c,
        n_aperiodicities,
        window,
    )
    if coarse_only:
        return coarse_aperiodicity, rms

    even_coarse_axis = (
        torch.arange(n_aperiodicities + 3, device=x.device) * FREQ_INTERVAL
    )
    assert even_coarse_axis[-2] <= sr // 2 < even_coarse_axis[-1], sr
    coarse_axis_low = (
        torch.arange(n_aperiodicities + 1, dtype=torch.float, device=x.device)
        * FREQ_INTERVAL
    )
    aperiodicity_low = interp(
        coarse_axis_low,
        F.pad(coarse_aperiodicity, (1, 0), value=-60.0),
        freq_axis[freq_axis < n_aperiodicities * FREQ_INTERVAL],
    )
    coarse_axis_high = torch.tensor(
        [n_aperiodicities * FREQ_INTERVAL, sr * 0.5], device=x.device
    )
    aperiodicity_high = interp(
        coarse_axis_high,
        F.pad(coarse_aperiodicity[..., -1:], (0, 1), value=-1e-12),
        freq_axis[freq_axis >= n_aperiodicities * FREQ_INTERVAL],
    )
    aperiodicity = torch.cat([aperiodicity_low, aperiodicity_high], -1)
    aperiodicity = 10.0 ** (aperiodicity / 20.0)
    vuv = d4c_love_train(x[..., None, :], sr, f0, temporal_positions, threshold)
    aperiodicity = torch.where(vuv[..., None], aperiodicity, 1 - 1e-12)

    return aperiodicity, coarse_aperiodicity


class Vocoder(nn.Module):
    def __init__(
        self,
        channels: int,
        speaker_embedding_channels: int = 128,
        hop_length: int = 240,
        n_pre_blocks: int = 4,
        out_sample_rate: float = 24000.0,
    ):
        super().__init__()
        self.hop_length = hop_length
        self.out_sample_rate = out_sample_rate

        self.prenet = ConvNeXtStack(
            in_channels=channels,
            channels=channels,
            intermediate_channels=channels * 2,
            n_blocks=n_pre_blocks,
            delay=2,  # 20ms 遅延
            embed_kernel_size=7,
            kernel_size=33,
            enable_scaling=True,
            use_mha=True,
            cross_attention=True,
            kv_channels=speaker_embedding_channels,
        )
        self.ir_generator = ConvNeXtStack(
            in_channels=channels,
            channels=channels,
            intermediate_channels=channels * 2,
            n_blocks=2,
            delay=0,
            embed_kernel_size=3,
            kernel_size=33,
            use_weight_standardization=True,
            enable_scaling=True,
        )
        self.ir_generator_post = WSConv1d(channels, 512, 1)
        self.register_buffer("ir_scale", torch.tensor(1.0))
        self.ir_window = nn.Parameter(torch.ones(512))
        self.aperiodicity_generator = ConvNeXtStack(
            in_channels=channels,
            channels=channels,
            intermediate_channels=channels * 2,
            n_blocks=1,
            delay=0,
            embed_kernel_size=3,
            kernel_size=33,
            use_weight_standardization=True,
            enable_scaling=True,
        )
        self.aperiodicity_generator_post = WSConv1d(channels, hop_length, 1, bias=False)
        self.register_buffer("aperiodicity_scale", torch.tensor(0.005))
        self.post_filter_generator = ConvNeXtStack(
            in_channels=channels,
            channels=channels,
            intermediate_channels=channels * 2,
            n_blocks=1,
            delay=0,
            embed_kernel_size=3,
            kernel_size=33,
            use_weight_standardization=True,
            enable_scaling=True,
        )
        self.post_filter_generator_post = WSConv1d(channels, 512, 1, bias=False)
        self.register_buffer("post_filter_scale", torch.tensor(0.01))

    def forward(
        self, x: torch.Tensor, pitch: torch.Tensor, speaker_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # x: [batch_size, channels, length]
        # pitch: [batch_size, length]
        # speaker_embedding: [batch_size, speaker_embedding_length, speaker_embedding_channels]
        batch_size, _, length = x.size()

        x = self.prenet(x, speaker_embedding)
        ir = self.ir_generator(x)
        ir = F.silu(ir, inplace=True)
        # [batch_size, 512, length]
        ir = self.ir_generator_post(ir)
        ir *= self.ir_scale
        ir_amp = ir[:, : ir.size(1) // 2 + 1, :].exp()
        ir_phase = F.pad(ir[:, ir.size(1) // 2 + 1 :, :], (0, 0, 1, 1))
        ir_phase[:, 1::2, :] += math.pi
        # TODO: 直流成分が正の値しか取れないのを修正する

        # 最近傍補間
        # [batch_size, length * hop_length]
        pitch = torch.repeat_interleave(pitch, self.hop_length, dim=1)

        # [batch_size, length * hop_length]
        periodic_signal = overlap_add(
            ir_amp,
            ir_phase,
            self.ir_window,
            pitch,
            self.hop_length,
            delay=0,
            sr=self.out_sample_rate,
        )

        aperiodicity = self.aperiodicity_generator(x)
        aperiodicity = F.silu(aperiodicity, inplace=True)
        # [batch_size, hop_length, length]
        aperiodicity = self.aperiodicity_generator_post(aperiodicity)
        aperiodicity *= self.aperiodicity_scale
        # [batch_size, length * hop_length], [batch_size, length * hop_length]
        aperiodic_signal, noise_excitation = generate_noise(aperiodicity, delay=0)

        post_filter = self.post_filter_generator(x)
        post_filter = F.silu(post_filter, inplace=True)
        # [batch_size, 512, length]
        post_filter = self.post_filter_generator_post(post_filter)
        post_filter *= self.post_filter_scale
        post_filter[:, 0, :] += 1.0
        # [batch_size, length, 512]
        post_filter = post_filter.transpose(1, 2)
        with torch.amp.autocast("cuda", enabled=False):
            periodic_signal = periodic_signal.float()
            aperiodic_signal = aperiodic_signal.float()
            post_filter = post_filter.float()
            post_filter = torch.fft.rfft(post_filter, n=768)

            # [batch_size, length, 768]
            periodic_signal = torch.fft.irfft(
                torch.fft.rfft(
                    periodic_signal.view(batch_size, length, self.hop_length), n=768
                )
                * post_filter,
                n=768,
            )
            aperiodic_signal = torch.fft.irfft(
                torch.fft.rfft(
                    aperiodic_signal.view(batch_size, length, self.hop_length), n=768
                )
                * post_filter,
                n=768,
            )
            periodic_signal = F.fold(
                periodic_signal.transpose(1, 2),
                (1, (length - 1) * self.hop_length + 768),
                (1, 768),
                stride=(1, self.hop_length),
            ).squeeze_((1, 2))
            aperiodic_signal = F.fold(
                aperiodic_signal.transpose(1, 2),
                (1, (length - 1) * self.hop_length + 768),
                (1, 768),
                stride=(1, self.hop_length),
            ).squeeze_((1, 2))
        periodic_signal = periodic_signal[:, 120 : 120 + length * self.hop_length]
        aperiodic_signal = aperiodic_signal[:, 120 : 120 + length * self.hop_length]
        noise_excitation = noise_excitation[:, 120:]

        # TODO: compensation の正確さが怪しくなってくる。今も本当に必要なのか？

        # [batch_size, 1, length * hop_length]
        y_g_hat = (periodic_signal + aperiodic_signal)[:, None, :]

        return y_g_hat, {
            "periodic_signal": periodic_signal.detach(),
            "aperiodic_signal": aperiodic_signal.detach(),
            "noise_excitation": noise_excitation.detach(),
        }

    def merge_weights(self):
        self.prenet.merge_weights()
        self.ir_generator.merge_weights()
        self.ir_generator_post.merge_weights()
        self.aperiodicity_generator.merge_weights()
        self.aperiodicity_generator_post.merge_weights()
        self.ir_generator_post.weight.data *= self.ir_scale
        self.ir_generator_post.bias.data *= self.ir_scale
        self.ir_scale.fill_(1.0)
        self.aperiodicity_generator_post.weight.data *= self.aperiodicity_scale
        self.aperiodicity_scale.fill_(1.0)
        self.post_filter_generator.merge_weights()
        self.post_filter_generator_post.merge_weights()
        self.post_filter_generator_post.weight.data *= self.post_filter_scale
        self.post_filter_scale.fill_(1.0)

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.prenet, f)
        dump_layer(self.ir_generator, f)
        dump_layer(self.ir_generator_post, f)
        dump_layer(self.ir_window, f)
        dump_layer(self.aperiodicity_generator, f)
        dump_layer(self.aperiodicity_generator_post, f)
        dump_layer(self.post_filter_generator, f)
        dump_layer(self.post_filter_generator_post, f)


def compute_loudness(
    x: torch.Tensor, sr: int, win_lengths: list[int]
) -> list[torch.Tensor]:
    # x: [batch_size, wav_length]
    assert x.ndim == 2
    n_fft = 2048
    chunk_length = n_fft // 2
    n_taps = chunk_length + 1

    results = []
    with torch.amp.autocast("cuda", enabled=False):
        if not hasattr(compute_loudness, "filter"):
            compute_loudness.filter = {}
        if sr not in compute_loudness.filter:
            ir = torch.zeros(n_taps, device=x.device, dtype=torch.double)
            ir[0] = 0.5
            ir = torchaudio.functional.treble_biquad(
                ir, sr, 4.0, 1500.0, 1.0 / math.sqrt(2)
            )
            ir = torchaudio.functional.highpass_biquad(ir, sr, 38.0, 0.5)
            ir *= 2.0
            compute_loudness.filter[sr] = torch.fft.rfft(ir, n=n_fft).to(
                torch.complex64
            )

        x = x.float()
        wav_length = x.size(-1)
        if wav_length % chunk_length != 0:
            x = F.pad(x, (0, chunk_length - wav_length % chunk_length))
        padded_wav_length = x.size(-1)
        x = x.view(x.size()[:-1] + (padded_wav_length // chunk_length, chunk_length))
        x = torch.fft.irfft(
            torch.fft.rfft(x, n=n_fft) * compute_loudness.filter[sr],
            n=n_fft,
        )
        x = F.fold(
            x.transpose(-2, -1),
            (1, padded_wav_length + chunk_length),
            (1, n_fft),
            stride=(1, chunk_length),
        ).squeeze_((-3, -2))[..., :wav_length]

        x.square_()
        for win_length in win_lengths:
            hop_length = win_length // 4
            # [..., n_frames]
            energy = (
                x.unfold(-1, win_length, hop_length)
                .matmul(torch.hann_window(win_length, device=x.device))
                .add_(win_length / 4.0 * 1e-5)
                .log10_()
            )
            # フィルタリング後の波形が振幅 1 の正弦波なら大体 log10(win_length/4), 1 の差は 10dB の差
            results.append(energy)
    return results


def slice_segments(
    x: torch.Tensor, start_indices: torch.Tensor, segment_length: int
) -> torch.Tensor:
    batch_size, channels, _ = x.size()
    # [batch_size, 1, segment_size]
    indices = start_indices[:, None, None] + torch.arange(
        segment_length, device=start_indices.device
    )
    # [batch_size, channels, segment_size]
    indices = indices.expand(batch_size, channels, segment_length)
    return x.gather(2, indices)


