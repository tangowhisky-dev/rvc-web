"""Audio preprocessing utilities — noise removal, RMS measurement, and normalization.

Uses noisereduce (spectral gating) which runs entirely on CPU/numpy.
torchaudio handles I/O and is MPS-aware for fast resampling where needed.

Public API
----------
get_duration(path)                          -> float | None   seconds
measure_speech_rms(path)                    -> float          RMS over voiced frames
normalize_rms_inplace(path, target_rms)     -> float          actual RMS after normalization
apply_dc_offset_removal(audio)              -> np.ndarray     removes DC bias in-place
apply_highpass_80hz(audio, sr)              -> np.ndarray     80 Hz highpass (SOS butterworth)
trim_silence(audio, sr, ...)                -> np.ndarray     strip leading/trailing silence
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

Silence trimming
----------------
``trim_silence`` uses a simple frame-energy gate to strip leading and trailing
silence.  The threshold is set slightly below the voiced-speech RMS threshold
(0.003 linear, ≈ −50 dBFS) to be conservative — it's better to keep a little
silence than to accidentally cut the beginning of a word.  A minimum of 100ms
of silence is kept at each end (``_TRIM_PAD_MS``) so word onsets aren't
hard-clipped.

DC offset removal
-----------------
RMS-normalizing after DC offset removal is more accurate: a DC bias shifts the
mean of the waveform, making the true AC (speech) amplitude appear smaller than
it is.  ``apply_dc_offset_removal`` subtracts the waveform mean from every
sample — a free, lossless operation that also removes low-frequency rumble that
DC-offset introduces into the frequency domain.

High-pass filter
----------------
``apply_highpass_80hz`` is a 4th-order Butterworth highpass at 80 Hz.  It
removes:
  - Mic rumble and handling noise (typically 20–60 Hz)
  - HVAC / room tone below 80 Hz
  - The ~50/60 Hz mains hum that appears even in treated rooms
RMVPE's minimum F0 range starts at 50 Hz but in practice F0 estimation for
speech starts at 70–80 Hz (bass male voices).  Everything below 80 Hz in
speech audio is either fundamental-only on the very deepest bass voices, or
noise.  Removing it improves RMVPE F0 accuracy on frames where the model might
otherwise lock onto a low-frequency noise component instead of the true F0.

Processing order in the upload pipeline
----------------------------------------
  1. Clip to selected segment
  2. DC offset removal       (mean subtraction — instant, no filtering artefacts)
  3. High-pass at 80 Hz      (removes rumble before RMS measurement)
  4. Trim silence             (strip > 500ms leading/trailing silence)
  5. Noise removal            (spectral gating — optional, triggered by user)
  6. RMS normalization        (normalize to profile_rms — or measure and store)
"""

import logging
import os
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

logger = logging.getLogger("rvc_web.audio")

# Voiced-frame threshold: frames below this RMS are excluded from the speech-
# weighted RMS calculation (silence + very quiet room noise).
_VOICED_THRESH = 0.005

# Headroom limit: never let the normalized peak exceed this value to avoid
# clipping after applying the RMS gain.
_MAX_PEAK = 0.95

# Frame size used for the voiced-frame RMS scan (25 ms at the file's native SR).
_FRAME_MS = 25

# Silence trimming: strip leading/trailing silence longer than this threshold.
# Slightly below _VOICED_THRESH to be conservative — we'd rather keep a little
# silence than accidentally cut the beginning of a word.
_TRIM_THRESH = 0.003
_TRIM_PAD_MS = 100    # always keep at least 100ms of silence at each end
_TRIM_MIN_MS = 500    # only strip if the silence run is longer than this

# High-pass filter order and cutoff
_HP_ORDER = 4
_HP_CUTOFF_HZ = 80.0


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
# DC offset removal
# ---------------------------------------------------------------------------

def apply_dc_offset_removal(audio: np.ndarray) -> np.ndarray:
    """Remove DC bias from a mono float32 audio array.

    Subtracts the mean of the entire signal.  This is a lossless, free
    operation — no filtering artefacts, no latency.  It should always be
    applied before RMS measurement and highpass filtering.

    A DC offset shifts the waveform's mean away from zero.  It can come from:
      - Cheap electret microphones with a DC bias on the capsule
      - ADC offsets in consumer-grade interfaces
      - Improperly AC-coupled signal paths

    Args:
        audio: Mono float32 numpy array.

    Returns:
        Copy of the array with the mean subtracted.
    """
    mean_val = float(audio.mean())
    if abs(mean_val) < 1e-6:
        return audio  # negligible offset — skip the copy
    result = audio - mean_val
    logger.debug("dc_offset_removal: removed offset %.6f", mean_val)
    return result


# ---------------------------------------------------------------------------
# High-pass filter at 80 Hz
# ---------------------------------------------------------------------------

def apply_highpass_80hz(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply a 4th-order Butterworth highpass at 80 Hz.

    Removes microphone rumble, handling noise, mains hum (50/60 Hz), and HVAC
    noise below 80 Hz.  Also improves RMVPE F0 accuracy by preventing the pitch
    extractor from latching onto sub-80 Hz noise instead of the true voice F0.

    The filter is applied with ``sosfilt`` (second-order sections) which is
    more numerically stable than the direct-form transfer function at low
    cutoff-to-samplerate ratios.

    Args:
        audio: Mono float32 numpy array.
        sr:    Sample rate of the audio in Hz.

    Returns:
        Highpass-filtered copy of the array (float32).
    """
    nyquist = sr / 2.0
    if _HP_CUTOFF_HZ >= nyquist:
        logger.warning("highpass cutoff %.0f Hz exceeds Nyquist %.0f Hz — skipping", _HP_CUTOFF_HZ, nyquist)
        return audio
    sos = butter(_HP_ORDER, _HP_CUTOFF_HZ / nyquist, btype="highpass", output="sos")
    filtered = sosfilt(sos, audio).astype(np.float32)
    logger.debug("highpass_80hz: applied %.0f Hz highpass at sr=%d", _HP_CUTOFF_HZ, sr)
    return filtered


# ---------------------------------------------------------------------------
# Silence trimming
# ---------------------------------------------------------------------------

def trim_silence(
    audio: np.ndarray,
    sr: int,
    thresh: float = _TRIM_THRESH,
    min_silence_ms: float = _TRIM_MIN_MS,
    pad_ms: float = _TRIM_PAD_MS,
) -> np.ndarray:
    """Strip leading and trailing silence from a mono audio array.

    Only removes silence runs longer than ``min_silence_ms`` (default 500ms).
    Keeps ``pad_ms`` (default 100ms) of silence at each end even after trimming,
    so word onsets are not hard-clipped.

    Algorithm:
      1. Divide the signal into 25ms frames.
      2. Compute the RMS of each frame.
      3. Find the first and last frames that exceed ``thresh``.
      4. Back off ``pad_ms`` from each boundary to retain natural onset/offset.
      5. Only trim if the silence run is ≥ ``min_silence_ms``.

    Args:
        audio:          Mono float32 numpy array.
        sr:             Sample rate in Hz.
        thresh:         Frame RMS threshold for voiced vs silent (default 0.003).
        min_silence_ms: Minimum silence run to trim (default 500ms).
        pad_ms:         Padding kept at each end after trimming (default 100ms).

    Returns:
        Trimmed array.  Returns the original if no trim is needed.
    """
    if len(audio) == 0:
        return audio

    frame = max(1, int(_FRAME_MS / 1000 * sr))
    pad_frames = max(1, int(pad_ms / 1000 * sr / frame))
    min_frames = max(1, int(min_silence_ms / 1000 * sr / frame))

    n_frames = len(audio) // frame
    if n_frames == 0:
        return audio

    rms_per_frame = np.array([
        float(np.sqrt(np.mean(audio[i*frame:(i+1)*frame] ** 2)))
        for i in range(n_frames)
    ])
    voiced = rms_per_frame >= thresh

    if not voiced.any():
        # Entirely silent — return as-is (don't discard everything)
        return audio

    first_voiced = int(np.argmax(voiced))
    last_voiced  = int(len(voiced) - 1 - np.argmax(voiced[::-1]))

    # Leading silence
    if first_voiced >= min_frames:
        start_frame = max(0, first_voiced - pad_frames)
        start_sample = start_frame * frame
        logger.debug(
            "trim_silence: removed %.0fms leading silence (kept %.0fms pad)",
            first_voiced * _FRAME_MS, pad_frames * _FRAME_MS,
        )
    else:
        start_sample = 0

    # Trailing silence
    trailing_silent_frames = (n_frames - 1) - last_voiced
    if trailing_silent_frames >= min_frames:
        end_frame = min(n_frames, last_voiced + 1 + pad_frames)
        end_sample = end_frame * frame
        logger.debug(
            "trim_silence: removed %.0fms trailing silence (kept %.0fms pad)",
            trailing_silent_frames * _FRAME_MS, pad_frames * _FRAME_MS,
        )
    else:
        end_sample = len(audio)

    return audio[start_sample:end_sample]


# ---------------------------------------------------------------------------
# Combined preprocessing pipeline (used by the upload flow)
# ---------------------------------------------------------------------------

def preprocess_audio_inplace(path: str) -> dict:
    """Apply DC removal → highpass → silence trim to an audio file in-place.

    This is the canonical preprocessing step applied automatically after every
    upload/clip operation, before noise removal and RMS normalization.

    Returns a dict with metadata about what was done:
        {
          "dc_offset_removed": float,    # mean value subtracted
          "highpass_applied": bool,
          "trimmed_ms": float,           # total ms removed (leading + trailing)
          "duration_before_s": float,
          "duration_after_s": float,
        }

    Raises:
        RuntimeError: if the file cannot be loaded or written.
    """
    try:
        audio, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception as exc:
        raise RuntimeError(f"Could not load audio for preprocessing: {exc}") from exc

    mono = audio.mean(axis=1)
    duration_before = len(mono) / sr

    # 1. DC offset
    mean_before = float(mono.mean())
    mono = apply_dc_offset_removal(mono)

    # 2. Highpass 80 Hz
    mono = apply_highpass_80hz(mono, sr)

    # 3. Silence trim
    n_before = len(mono)
    mono = trim_silence(mono, sr)
    n_after = len(mono)
    trimmed_ms = (n_before - n_after) / sr * 1000

    duration_after = len(mono) / sr

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, mono, sr, subtype="PCM_16", format="WAV")

    logger.info(
        "preprocess_audio: %s  dc=%.5f  hp=80Hz  trimmed=%.0fms  dur: %.1f→%.1fs",
        os.path.basename(path), mean_before, trimmed_ms, duration_before, duration_after,
    )
    return {
        "dc_offset_removed": mean_before,
        "highpass_applied": True,
        "trimmed_ms": trimmed_ms,
        "duration_before_s": round(duration_before, 3),
        "duration_after_s": round(duration_after, 3),
    }


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
