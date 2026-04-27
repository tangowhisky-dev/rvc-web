"""F0 transformation utilities for voice conversion.

Three post-processing transforms that can be layered on top of each other,
applied to voiced F0 frames after pitch extraction:

  1. Affine normalization (mean + std)  — already in _patched_get_f0 / B2 patch
  2. Histogram equalization (HistEQ)    — maps full distribution shape
  3. Velocity normalization             — scales intonation dynamics (causal)
  4. Range soft-clip                    — confines pitch to target speaking range

All functions operate in log-Hz space (natural-log of Hz).
RVC consumers work with Hz float arrays; B2 consumers work with integer bin arrays.

HistEQ storage format
---------------------
f0_hist: list[float] of length N_HIST_BINS
  f0_hist[i] = fraction of voiced frames whose log(F0) falls in bucket i
  Bucket i covers log-Hz range [LO + i*BW, LO + (i+1)*BW)
  where LO = log(F0_HIST_LO_HZ), BW = (log(F0_HIST_HI_HZ) - LO) / N_HIST_BINS
  The array is a density (sums to ~1 after normalisation).  An inverse-CDF lookup
  table is pre-built from the cumulative sum at inference time.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

F0_HIST_LO_HZ  = 50.0      # Lower bound for histogram (Hz)
F0_HIST_HI_HZ  = 1400.0    # Upper bound for histogram (Hz)
N_HIST_BINS    = 256        # Number of histogram buckets
_LOG_LO        = math.log(F0_HIST_LO_HZ)
_LOG_HI        = math.log(F0_HIST_HI_HZ)
_BW            = (_LOG_HI - _LOG_LO) / N_HIST_BINS  # bucket width in log-Hz

# Beatrice 2 bin space
B2_BPO         = 96         # bins per octave
B2_ANCHOR_HZ   = 55.0       # anchor frequency in Hz
B2_MAX_BIN     = 447        # maximum voiced bin index


# ---------------------------------------------------------------------------
# Statistics computation (shared by compute_speaker_f0 and input_f0)
# ---------------------------------------------------------------------------

def compute_f0_statistics(log_f0: np.ndarray) -> dict:
    """Compute full F0 statistics from an array of log-Hz voiced frames.

    Parameters
    ----------
    log_f0 : np.ndarray
        1-D array of log(Hz) values for voiced frames only (no zeros).

    Returns
    -------
    dict with keys:
        mean_f0, std_f0, p5_f0, p25_f0, p50_f0, p75_f0, p95_f0,
        vel_std, f0_hist (list[float] of length N_HIST_BINS)
    """
    arr = np.asarray(log_f0, dtype=np.float64)
    if len(arr) < 2:
        raise ValueError("Need at least 2 voiced frames to compute F0 statistics")

    mean_log  = float(arr.mean())
    std_log   = float(arr.std())
    mean_f0   = math.exp(mean_log)

    percs = np.percentile(arr, [5, 25, 50, 75, 95])
    p5_f0, p25_f0, p50_f0, p75_f0, p95_f0 = (math.exp(float(p)) for p in percs)

    # Velocity std: std of frame-to-frame log-Hz differences
    # Only consecutive voiced frames (assumes arr is in temporal order — per-file concat).
    vel_std = float(np.diff(arr).std()) if len(arr) > 1 else 0.0

    # 256-bucket histogram (density — sums to ~1)
    counts, _ = np.histogram(arr, bins=N_HIST_BINS,
                             range=(_LOG_LO, _LOG_HI))
    total = counts.sum()
    hist = (counts / total).tolist() if total > 0 else [0.0] * N_HIST_BINS

    return {
        "mean_f0":  round(mean_f0, 4),
        "std_f0":   round(std_log, 6),
        "p5_f0":    round(p5_f0, 4),
        "p25_f0":   round(p25_f0, 4),
        "p50_f0":   round(p50_f0, 4),
        "p75_f0":   round(p75_f0, 4),
        "p95_f0":   round(p95_f0, 4),
        "vel_std":  round(vel_std, 6),
        "f0_hist":  hist,
    }


# ---------------------------------------------------------------------------
# Inverse CDF helpers for HistEQ
# ---------------------------------------------------------------------------

def build_inverse_cdf(hist: list[float]) -> np.ndarray:
    """Build a 256-sample inverse CDF lookup table from a histogram density.

    The table maps uniform quantile rank (0..255 / 255) → log-Hz value,
    ready for np.interp lookup.

    Parameters
    ----------
    hist : list[float] len N_HIST_BINS
        Density histogram (should sum to ~1).

    Returns
    -------
    np.ndarray, shape (N_HIST_BINS,), dtype float64
        inverse_cdf[i] is the log-Hz value at quantile rank i/(N_HIST_BINS-1).
    """
    counts = np.asarray(hist, dtype=np.float64)
    cdf = np.cumsum(counts)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    # Map each histogram bucket to its center log-Hz
    bucket_centers = _LOG_LO + (_BW * (np.arange(N_HIST_BINS) + 0.5))
    # Build inverse CDF: for each uniform quantile point, find log-Hz
    quantile_points = np.linspace(0.0, 1.0, N_HIST_BINS)
    inv_cdf = np.interp(quantile_points, cdf, bucket_centers)
    return inv_cdf


def source_cdf_from_log_f0(log_f0: np.ndarray) -> np.ndarray:
    """Build the CDF of a source file's voiced F0 array.

    Returns the same shape as build_inverse_cdf: the quantile → log-Hz mapping.
    This is used at inference time for HistEQ.
    """
    counts, _ = np.histogram(log_f0, bins=N_HIST_BINS, range=(_LOG_LO, _LOG_HI))
    total = counts.sum()
    if total == 0:
        return np.linspace(_LOG_LO, _LOG_HI, N_HIST_BINS)
    hist = counts / total
    return build_inverse_cdf(hist.tolist())


# ---------------------------------------------------------------------------
# Transform: Histogram equalization
# ---------------------------------------------------------------------------

def histeq_transform(
    f0_hz: np.ndarray,
    src_inv_cdf: np.ndarray,
    tgt_inv_cdf: np.ndarray,
) -> np.ndarray:
    """Apply histogram equalization to an F0 array (Hz, voiced frames only).

    Maps each voiced frame to its rank in the source CDF, then looks up the
    target CDF at that rank.  Monotone transform — preserves pitch contour shape.

    Parameters
    ----------
    f0_hz : np.ndarray
        F0 values in Hz.  Zeros treated as unvoiced (pass through unchanged).
    src_inv_cdf : np.ndarray, shape (N_HIST_BINS,)
        Inverse CDF of source speaker (from source_cdf_from_log_f0).
    tgt_inv_cdf : np.ndarray, shape (N_HIST_BINS,)
        Inverse CDF of target speaker (from build_inverse_cdf).

    Returns
    -------
    np.ndarray
        Transformed F0 in Hz, same shape as input.
    """
    out = f0_hz.copy().astype(np.float64)
    voiced = out > 0
    if not voiced.any():
        return out.astype(np.float32)

    log_f = np.log(out[voiced])
    quantile_points = np.linspace(0.0, 1.0, N_HIST_BINS)

    # Build source CDF (for mapping each log_f to a rank)
    # src_inv_cdf maps quantile → log_f_src, so we invert: log_f_src → quantile
    rank = np.interp(log_f, src_inv_cdf, quantile_points)

    # Map rank to target log-Hz via target inverse CDF
    log_f_out = np.interp(rank, quantile_points, tgt_inv_cdf)
    out[voiced] = np.exp(log_f_out)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Transform: Velocity normalization
# ---------------------------------------------------------------------------

class VelocityNormalizer:
    """Causal velocity normalization — maintains running state across blocks.

    Scales frame-to-frame log-Hz increments by (vel_std_tgt / vel_std_src),
    then re-centers to the running mean of the input (affine mean preserved).

    Designed for realtime use: processes one block at a time.
    Resets on a silence gap longer than `gap_tolerance` consecutive frames.
    """

    def __init__(
        self,
        vel_ratio: float,          # σ_vel_t / σ_vel_s
        gap_tolerance: int = 5,    # frames of silence before resetting accumulator
    ) -> None:
        self.vel_ratio = float(vel_ratio)
        self.gap_tolerance = gap_tolerance
        self._prev_log_in: Optional[float] = None   # last voiced input frame
        self._prev_log_out: Optional[float] = None  # last voiced output frame
        self._silence_count: int = 0
        self._sum_in: float = 0.0
        self._sum_out: float = 0.0
        self._n: int = 0

    def reset(self) -> None:
        self._prev_log_in = None
        self._prev_log_out = None
        self._silence_count = 0
        self._sum_in = 0.0
        self._sum_out = 0.0
        self._n = 0

    def process(self, f0_hz: np.ndarray) -> np.ndarray:
        """Process one block of F0 (Hz).  Zeros = unvoiced, pass through.

        Returns transformed F0 array (Hz), same shape as input.
        """
        if abs(self.vel_ratio - 1.0) < 1e-4:
            return f0_hz  # identity — skip compute

        out = f0_hz.copy().astype(np.float64)
        for i in range(len(out)):
            hz = out[i]
            if hz <= 0.0:
                self._silence_count += 1
                if self._silence_count >= self.gap_tolerance:
                    # Long silence — reset accumulator so next phrase starts fresh
                    self._prev_log_in = None
                    self._prev_log_out = None
                continue

            self._silence_count = 0
            log_in = math.log(hz)
            self._sum_in += log_in
            self._n += 1

            if self._prev_log_in is None:
                # First voiced frame or after a gap reset: no delta, passthrough
                log_out = log_in
            else:
                delta = log_in - self._prev_log_in
                log_out = self._prev_log_out + delta * self.vel_ratio  # type: ignore[operator]

            self._prev_log_in = log_in
            self._prev_log_out = log_out
            self._sum_out += log_out
            out[i] = math.exp(log_out)

        # Re-center: shift output mean back to input mean (preserves affine mean)
        if self._n > 0:
            drift = (self._sum_out - self._sum_in) / self._n
            if abs(drift) > 1e-6:
                voiced_mask = out > 0
                if voiced_mask.any():
                    out[voiced_mask] *= math.exp(-drift)
                    # Update accumulators to reflect the correction
                    self._sum_out -= drift * self._n

        return out.astype(np.float32)


class VelocityNormalizerBins:
    """Same as VelocityNormalizer but operates on B2 bin integers.

    Bins are linear in log-Hz so velocity scaling in bin space is equivalent.
    """

    def __init__(self, vel_ratio: float, gap_tolerance: int = 5) -> None:
        self.vel_ratio = float(vel_ratio)
        self.gap_tolerance = gap_tolerance
        self._prev_bin_in: Optional[float] = None
        self._prev_bin_out: Optional[float] = None
        self._silence_count: int = 0
        self._sum_in: float = 0.0
        self._sum_out: float = 0.0
        self._n: int = 0

    def reset(self) -> None:
        self._prev_bin_in = None
        self._prev_bin_out = None
        self._silence_count = 0
        self._sum_in = 0.0
        self._sum_out = 0.0
        self._n = 0

    def process_tensor(self, qp: "torch.Tensor") -> "torch.Tensor":
        """Process a Long tensor of bin indices.  0 = unvoiced."""
        import torch
        if abs(self.vel_ratio - 1.0) < 1e-4:
            return qp

        arr = qp.cpu().numpy().astype(np.float64).flatten()
        out = arr.copy()
        for i in range(len(out)):
            b = arr[i]
            if b <= 0:
                self._silence_count += 1
                if self._silence_count >= self.gap_tolerance:
                    self._prev_bin_in = None
                    self._prev_bin_out = None
                continue
            self._silence_count = 0
            self._sum_in += b
            self._n += 1

            if self._prev_bin_in is None:
                b_out = b
            else:
                delta = b - self._prev_bin_in
                b_out = self._prev_bin_out + delta * self.vel_ratio  # type: ignore[operator]

            self._prev_bin_in = b
            self._prev_bin_out = b_out
            self._sum_out += b_out
            out[i] = b_out

        # Re-center
        if self._n > 0:
            drift = (self._sum_out - self._sum_in) / self._n
            if abs(drift) > 0.5:
                voiced = out > 0
                out[voiced] -= drift
                self._sum_out -= drift * self._n

        result = np.round(out).astype(np.int64)
        result = np.clip(result, 0, B2_MAX_BIN)
        return torch.from_numpy(result).reshape(qp.shape).to(qp.device)


# ---------------------------------------------------------------------------
# Transform: Range soft-clip
# ---------------------------------------------------------------------------

def soft_clip_f0(
    f0_hz: np.ndarray,
    p5_hz: float,
    p95_hz: float,
    hardness: float = 2.0,
) -> np.ndarray:
    """Soft-clip F0 to the target's speaking range [p5_hz, p95_hz].

    Uses tanh compression: frames well within the range are almost unchanged;
    frames far outside are pulled asymptotically toward the boundary.

    Parameters
    ----------
    hardness : float
        Controls steepness of the squash (higher = closer to hard clip).
        2.0 = gentle (< 1 semitone error within range), 4.0 = firm.
    """
    out = f0_hz.copy().astype(np.float64)
    voiced = out > 0
    if not voiced.any():
        return out.astype(np.float32)

    log_lo    = math.log(max(p5_hz, 20.0))
    log_hi    = math.log(max(p95_hz, p5_hz * 1.01))
    center    = (log_lo + log_hi) / 2.0
    half_r    = (log_hi - log_lo) / 2.0

    if half_r < 1e-6:
        return out.astype(np.float32)

    log_f  = np.log(out[voiced])
    norm   = (log_f - center) / half_r
    clipped = np.tanh(norm * hardness) / math.tanh(hardness)
    out[voiced] = np.exp(center + clipped * half_r)
    return out.astype(np.float32)


def soft_clip_bins(
    qp: "torch.Tensor",
    p5_hz: float,
    p95_hz: float,
    hardness: float = 2.0,
) -> "torch.Tensor":
    """Soft-clip B2 bin indices to the target's speaking range."""
    import torch

    log_lo = math.log(max(p5_hz, 20.0))
    log_hi = math.log(max(p95_hz, p5_hz * 1.01))
    # Convert Hz bounds to B2 bins
    bin_lo = math.log2(math.exp(log_lo) / B2_ANCHOR_HZ) * B2_BPO
    bin_hi = math.log2(math.exp(log_hi) / B2_ANCHOR_HZ) * B2_BPO
    center    = (bin_lo + bin_hi) / 2.0
    half_r    = (bin_hi - bin_lo) / 2.0

    if half_r < 0.5:
        return qp

    voiced = qp > 0
    if not voiced.any():
        return qp

    qp_f  = qp.float()
    norm  = (qp_f - center) / half_r
    # tanh(hardness) constant
    th    = math.tanh(hardness)
    clipped = torch.tanh(norm * hardness) / th
    out = center + clipped * half_r
    # Round to nearest bin, clamp, keep unvoiced as 0
    out_long = torch.round(out).long().clamp(1, B2_MAX_BIN)
    return torch.where(voiced, out_long, qp)


# ---------------------------------------------------------------------------
# Full pipeline helper: apply_f0_prior
# ---------------------------------------------------------------------------

def apply_f0_prior_hz(
    f0_hz: np.ndarray,
    params: dict,
    vel_state: Optional[VelocityNormalizer] = None,
) -> np.ndarray:
    """Apply the full F0 prior transform pipeline to an Hz array.

    Transform order:
      1. Affine (mean + std)  — applied before this call in the existing patch
      2. HistEQ               — if src_hist and tgt_hist are present
      3. Velocity norm        — if vel_state is provided
      4. Soft-clip            — if p5_tgt and p95_tgt are present

    Parameters
    ----------
    f0_hz : np.ndarray   F0 in Hz (0 = unvoiced)
    params : dict        F0 normalization params dict (from form or worker state)
    vel_state : VelocityNormalizer | None  Persistent velocity state (None for offline)

    The params dict may contain:
        src_mean, src_std, tgt_mean, tgt_std  — affine (handled upstream)
        src_hist: list[float]     — source histogram for HistEQ (offline only)
        tgt_hist: list[float]     — target histogram for HistEQ (offline only)
        vel_ratio: float          — σ_vel_t / σ_vel_s  (for velocity norm)
        p5_tgt: float             — target 5th percentile Hz (soft-clip floor)
        p95_tgt: float            — target 95th percentile Hz (soft-clip ceiling)
    """
    result = f0_hz.copy()

    # HistEQ (offline only — requires pre-computed src CDF)
    src_hist = params.get("src_hist")
    tgt_hist = params.get("tgt_hist")
    if src_hist is not None and tgt_hist is not None:
        src_inv_cdf = build_inverse_cdf(src_hist)
        tgt_inv_cdf = build_inverse_cdf(tgt_hist)
        result = histeq_transform(result, src_inv_cdf, tgt_inv_cdf)

    # Velocity normalization
    vel_ratio = params.get("vel_ratio")
    if vel_ratio is not None and abs(float(vel_ratio) - 1.0) > 1e-4:
        if vel_state is None:
            vel_state = VelocityNormalizer(float(vel_ratio))
        result = vel_state.process(result)

    # Soft-clip
    p5_tgt  = params.get("p5_tgt")
    p95_tgt = params.get("p95_tgt")
    if p5_tgt is not None and p95_tgt is not None and p95_tgt > p5_tgt:
        result = soft_clip_f0(result, float(p5_tgt), float(p95_tgt))

    return result


def apply_f0_prior_bins(
    qp: "torch.Tensor",
    params: dict,
    vel_state: Optional[VelocityNormalizerBins] = None,
) -> "torch.Tensor":
    """Apply the F0 prior pipeline to B2 bin integers.

    Applies:
      1. Affine in bin space (handled upstream by the norm patch)
      2. Velocity norm (if vel_ratio present)
      3. Soft-clip in bin space (if p5_tgt / p95_tgt present)

    Note: HistEQ is not applied in bin space (requires conversion to Hz and back;
    currently not implemented — the affine norm in bin space subsumes the mean/std
    component and soft-clip bounds the range).
    """
    result = qp

    # Velocity normalization
    vel_ratio = params.get("vel_ratio")
    if vel_ratio is not None and abs(float(vel_ratio) - 1.0) > 1e-4:
        if vel_state is None:
            vel_state = VelocityNormalizerBins(float(vel_ratio))
        result = vel_state.process_tensor(result)

    # Soft-clip
    p5_tgt  = params.get("p5_tgt")
    p95_tgt = params.get("p95_tgt")
    if p5_tgt is not None and p95_tgt is not None and p95_tgt > p5_tgt:
        result = soft_clip_bins(result, float(p5_tgt), float(p95_tgt))

    return result
