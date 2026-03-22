"""Audio preprocessing utilities — noise removal and duration extraction.

Uses noisereduce (spectral gating) which runs entirely on CPU/numpy.
torchaudio handles I/O and is MPS-aware for fast resampling where needed.

Public API
----------
get_duration(path)          -> float seconds
remove_noise(in_path, out_path, start_sec, end_sec) -> None
"""

import logging
import os
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger("rvc_web.audio")


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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, cleaned, sr, subtype="PCM_16")

    logger.info(
        "noise removal complete: %s -> %s (%.1f s)",
        os.path.basename(input_path),
        os.path.basename(output_path),
        len(cleaned) / sr,
    )
