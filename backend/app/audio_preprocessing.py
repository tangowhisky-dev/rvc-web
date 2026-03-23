"""Audio preprocessing utilities — noise removal, RMS measurement, and normalization.

Uses noisereduce (spectral gating) which runs entirely on CPU/numpy.
torchaudio handles I/O and is MPS-aware for fast resampling where needed.

Public API
----------
get_duration(path)                          -> float | None   seconds
measure_speech_rms(path)                    -> float          RMS over voiced frames
normalize_rms_inplace(path, target_rms)     -> float          actual RMS after normalization
remove_noise(in_path, out_path, ...)        -> None

RMS normalization strategy
--------------------------
``measure_speech_rms`` computes the RMS over frames where the signal exceeds
a voiced-speech threshold (0.005 linear, ≈ −46 dBFS).  Using only voiced
frames prevents long silences from pulling the mean down and making the
normalization target inconsistent.

``normalize_rms_inplace`` applies a single linear gain so the speech-weighted
RMS matches ``target_rms``, then writes back to the same file.  A headroom
guard (``_MAX_PEAK``) prevents clipping.

The stored ``profile_rms`` value on a profile is the speech-weighted RMS of
the *first* uploaded audio file for that profile.  Every subsequent file is
normalized to match it before being written to disk, so the training corpus
is level-consistent.  At inference time the realtime worker applies the same
gain to the microphone input so the model always sees input at the level it
was trained on.
"""

import logging
import os
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger("rvc_web.audio")

# Voiced-frame threshold: frames below this RMS are excluded from the speech-
# weighted RMS calculation (silence + very quiet room noise).
_VOICED_THRESH = 0.005

# Headroom limit: never let the normalized peak exceed this value to avoid
# clipping after applying the RMS gain.
_MAX_PEAK = 0.95

# Frame size used for the voiced-frame RMS scan (25 ms at the file's native SR).
_FRAME_MS = 25


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

def get_duration(path: str) -> Optional[float]:
    """Return audio duration in seconds using soundfile (fast header-only read).

    Returns None if the file cannot be opened.
    """
    try:
        info = sf.info(path)
        return round(info.duration, 3)
    except Exception:
        # Fall back to torchaudio for formats soundfile can't handle (MP3)
        try:
            import torchaudio  # noqa: PLC0415
            info = torchaudio.info(path)
            return round(info.num_frames / info.sample_rate, 3)
        except Exception as exc:
            logger.warning("get_duration failed for %s: %s", path, exc)
            return None


# ---------------------------------------------------------------------------
# RMS measurement
# ---------------------------------------------------------------------------

def measure_speech_rms(path: str) -> float:
    """Return the speech-weighted RMS of the audio file at ``path``.

    Only voiced frames (frame RMS ≥ ``_VOICED_THRESH``) are included in the
    average.  If no voiced frames are found the overall RMS is returned as a
    fallback so the function never returns 0.

    Args:
        path: Path to any audio file readable by soundfile (WAV, FLAC, OGG, MP3).

    Returns:
        Speech-weighted RMS as a float in the range (0, ∞).

    Raises:
        RuntimeError: if the file cannot be loaded.
    """
    try:
        audio, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception as exc:
        raise RuntimeError(f"Could not load audio for RMS measurement: {exc}") from exc

    mono = audio.mean(axis=1)  # [N]
    frame = max(1, int(_FRAME_MS / 1000 * sr))

    voiced_rms_sq = []
    for i in range(0, len(mono) - frame, frame):
        seg = mono[i : i + frame]
        seg_rms = float(np.sqrt(np.mean(seg ** 2)))
        if seg_rms >= _VOICED_THRESH:
            voiced_rms_sq.append(seg_rms ** 2)

    if voiced_rms_sq:
        return float(np.sqrt(np.mean(voiced_rms_sq)))

    # Fallback: overall RMS (file may be uniformly quiet)
    overall = float(np.sqrt(np.mean(mono ** 2)))
    return max(overall, 1e-6)


# ---------------------------------------------------------------------------
# RMS normalization
# ---------------------------------------------------------------------------

def normalize_rms_inplace(path: str, target_rms: float) -> float:
    """Normalize ``path`` so its speech-weighted RMS matches ``target_rms``.

    Reads the file, computes the current speech RMS, applies a linear gain,
    guards against clipping, then writes back in-place as 16-bit PCM WAV.

    Args:
        path:       WAV file to normalize (modified in-place).
        target_rms: Desired speech-weighted RMS (from ``measure_speech_rms``).

    Returns:
        The actual speech-weighted RMS of the file after normalization.

    Raises:
        RuntimeError: if loading or writing fails.
    """
    if target_rms <= 0:
        raise ValueError(f"target_rms must be positive, got {target_rms}")

    try:
        audio, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception as exc:
        raise RuntimeError(f"Could not load audio for RMS normalization: {exc}") from exc

    mono = audio.mean(axis=1)

    current_rms = measure_speech_rms(path)
    gain = target_rms / current_rms

    # Headroom guard: scale down gain if it would clip
    peak_after = float(np.abs(mono).max()) * gain
    if peak_after > _MAX_PEAK:
        gain = gain * (_MAX_PEAK / peak_after)

    normalized = mono * gain

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, normalized, sr, subtype="PCM_16", format="WAV")

    actual_rms = measure_speech_rms(path)
    logger.info(
        "rms_normalize: %s  current=%.5f  target=%.5f  gain=%.3f×  actual=%.5f",
        os.path.basename(path),
        current_rms,
        target_rms,
        gain,
        actual_rms,
    )
    return actual_rms


# ---------------------------------------------------------------------------
# Noise removal
# ---------------------------------------------------------------------------

def remove_noise(
    input_path: str,
    output_path: str,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> None:
    """Read audio from input_path, slice [start_sec, end_sec], apply noise
    reduction, and write the result as 16-bit WAV to output_path.

    Algorithm: noisereduce spectral gating (Jansson 2017-style).
    Runs on CPU only — MPS not needed for this numpy/scipy pipeline.

    The first 0.5 s of the slice is used as the noise profile if the slice
    starts at 0, otherwise a 0.5 s window from the beginning of the full
    file is used.

    Note: this function does NOT apply RMS normalization.  Call
    ``normalize_rms_inplace`` on the output if level consistency is needed.

    Args:
        input_path:  Source audio file (any format soundfile can read: MP3, WAV, FLAC, OGG).
        output_path: Destination WAV file path.
        start_sec:   Start of the selected segment in seconds.
        end_sec:     End of the selected segment, or None for end of file.

    Raises:
        RuntimeError: if loading, processing, or saving fails.
    """
    import noisereduce as nr  # noqa: PLC0415

    # --- load via soundfile (handles MP3/WAV/FLAC/OGG without torchcodec) ---
    try:
        raw, sr = sf.read(input_path, always_2d=True, dtype="float32")  # [N, C]
    except Exception as exc:
        raise RuntimeError(f"Could not load audio: {exc}") from exc

    total_frames = raw.shape[0]

    # --- slice ---
    start_frame = int(round(start_sec * sr))
    end_frame   = int(round(end_sec   * sr)) if end_sec is not None else total_frames
    start_frame = max(0, min(start_frame, total_frames))
    end_frame   = max(start_frame, min(end_frame, total_frames))

    sliced = raw[start_frame:end_frame, :]        # [N', C]

    # --- to numpy mono float32 ---
    audio_np = sliced.mean(axis=1)                # [N']

    # --- build noise profile from first 0.5 s of the file ---
    noise_frames = min(int(0.5 * sr), total_frames)
    noise_profile = raw[:noise_frames, :].mean(axis=1)

    # --- noise reduction ---
    try:
        cleaned = nr.reduce_noise(
            y=audio_np,
            sr=sr,
            y_noise=noise_profile,
            prop_decrease=0.80,   # aggressive but preserves speech well
            stationary=False,     # non-stationary: adapt per frame
            n_fft=2048,
            n_jobs=1,             # single-threaded to avoid MPS/OpenMP conflict
        )
    except Exception as exc:
        raise RuntimeError(f"noisereduce failed: {exc}") from exc

    # --- normalise to -1 dB headroom then save as 16-bit WAV ---
    peak = np.abs(cleaned).max()
    if peak > 0:
        cleaned = cleaned / peak * 0.891  # -1 dBFS

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, cleaned, sr, subtype="PCM_16", format="WAV")

    logger.info(
        "noise removal complete: %s -> %s (%.1f s)",
        os.path.basename(input_path),
        os.path.basename(output_path),
        len(cleaned) / sr,
    )
