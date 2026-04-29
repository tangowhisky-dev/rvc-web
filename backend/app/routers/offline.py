"""Offline (file-based) voice conversion router.

Endpoints:
  POST  /api/offline/convert            — convert an uploaded audio file, stream SSE progress
  GET   /api/offline/result/{job_id}    — download the converted file
  GET   /api/offline/input/{job_id}     — download the original input file
  POST  /api/offline/analyze/{job_id}   — run ECAPA analysis on input+output temp files
  GET   /api/offline/profiles           — list trained profiles available for inference

Input and output files are saved to {PROJECT_ROOT}/temp/ as:
  temp/{job_id}_input.{ext}
  temp/{job_id}_output.{ext}

These are swept on startup and shutdown via start.sh / stop.sh.
"""

import asyncio
import io
import logging
import os
import sys
import time
import uuid
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.app.db import get_db
from backend.app.routers.profiles import ProfileOut
import aiosqlite
from backend.app import db

# MIME types for common audio formats
_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
}



# ---------------------------------------------------------------------------
# Temp directory helpers
# ---------------------------------------------------------------------------

def _temp_dir() -> str:
    """Return {PROJECT_ROOT}/temp/, creating it if needed."""
    root = os.environ.get("PROJECT_ROOT", "")
    if not root:
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
    d = os.path.join(root, "temp")
    os.makedirs(d, exist_ok=True)
    return d


def _input_path(job_id: str, ext: str) -> str:
    return os.path.join(_temp_dir(), f"{job_id}_input{ext}")


def _output_path(job_id: str, ext: str) -> str:
    return os.path.join(_temp_dir(), f"{job_id}_output{ext}")



def _write_audio(path: str, audio: np.ndarray, sr: int, ext: str) -> None:
    """Write audio to *path* in the format implied by *ext*.

    Offline inference always outputs WAV PCM-16 (out_ext=".wav"), matching
    the realtime save format. Non-WAV formats are left as a fallback in case
    the caller ever changes out_ext, but pydub/ffmpeg are not required.
    """
    ext = ext.lower()
    fmt_map = {
        ".wav": "WAV",
        ".flac": "FLAC",
        ".ogg": "OGG",
        ".aiff": "AIFF",
        ".aif": "AIFF",
    }
    if ext in fmt_map:
        sf.write(path, audio, sr, format=fmt_map[ext], subtype="PCM_16")
    else:
        # Unsupported extension — fall back to WAV at the same path.
        # pydub is no longer a dependency; if lossy output is ever needed,
        # use torchaudio.save() with the appropriate encoder.
        import warnings
        warnings.warn(f"Unsupported audio extension '{ext}', writing as WAV instead.")
        sf.write(path, audio, sr, format="WAV", subtype="PCM_16")


logger = logging.getLogger("rvc_web.offline")

router = APIRouter(prefix="/api/offline", tags=["offline"])

# ---------------------------------------------------------------------------
# In-memory job store (results held until downloaded or server restart)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}  # job_id → {status, input_path, result_path, error, sample_rate, fmt}


# ---------------------------------------------------------------------------
# Profile list — re-uses same DB query as training page
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Inference helpers — mirrors _realtime_worker.py block pipeline exactly
# ---------------------------------------------------------------------------

_SR_16K = 16000
_SR_48K = 48000
_WINDOW = 160
_ZC = _SR_48K // 100  # 480
_BLOCK_48K = 9600  # 200ms blocks — same as realtime
_SOLA_BUF_MS_DEFAULT = 20  # ms — matches realtime worker default
_SOLA_SEARCH_48K = _ZC  # 480
_EXTRA_TIME = 2.0  # 2s context window


def _compute_block_params(block_48k: int = _BLOCK_48K,
                          sola_buf_ms: int = _SOLA_BUF_MS_DEFAULT) -> dict:
    sola_buf_ms_clamped = max(0, min(sola_buf_ms, 50))
    sola_buf_48k = int(sola_buf_ms_clamped * _SR_48K / 1000)
    block_16k = int(block_48k * _SR_16K / _SR_48K)
    f0_min = block_16k + 800
    f0_extractor_frame = 5120 * ((f0_min - 1) // 5120 + 1) - _WINDOW
    extra_16k = (int(_EXTRA_TIME * _SR_16K) // _WINDOW) * _WINDOW
    total_16k_samples = extra_16k + f0_extractor_frame
    p_len = total_16k_samples // _WINDOW
    total_out_48k = block_48k + sola_buf_48k + _SOLA_SEARCH_48K
    return_length_frames = total_out_48k // _ZC
    skip_head_frames = p_len - return_length_frames
    return {
        "block_48k": block_48k,
        "block_16k": block_16k,
        "f0_extractor_frame": f0_extractor_frame,
        "total_16k_samples": total_16k_samples,
        "p_len": p_len,
        "return_length_frames": return_length_frames,
        "skip_head_frames": skip_head_frames,
        "sola_buf_48k": sola_buf_48k,
        "sola_search_48k": _SOLA_SEARCH_48K,
    }


def _apply_rnnoise(audio_48k: np.ndarray) -> np.ndarray:
    """Denoise a float32 mono 48kHz array with pyrnnoise.

    Processes in 480-sample (10ms) frames.  Returns float32 in the same range.
    Fails silently — returns input unchanged if pyrnnoise is unavailable.
    """
    try:
        from pyrnnoise.rnnoise import create, destroy, process_mono_frame, FRAME_SIZE
    except Exception:
        return audio_48k

    state = create()
    try:
        pad = (-len(audio_48k)) % FRAME_SIZE
        pcm = np.concatenate([audio_48k, np.zeros(pad, dtype=np.float32)]) if pad else audio_48k
        frames_out = []
        for i in range(0, len(pcm), FRAME_SIZE):
            frame_out, _ = process_mono_frame(state, pcm[i:i + FRAME_SIZE])
            frames_out.append(frame_out.astype(np.float32) / 32767.0)
        return np.concatenate(frames_out)[:len(audio_48k)]
    finally:
        destroy(state)


def _run_offline_inference(
    audio_16k: np.ndarray,  # float32, mono, 16 kHz
    audio_tgt: np.ndarray,  # float32, mono, tgt_sr  (for SOLA output alignment)
    model_path: str,
    index_path: str,
    pitch: float,
    index_rate: float,
    protect: float,
    progress_cb,  # callable(fraction: float)
    output_gain: float = 1.0,  # post-inference volume multiplier (1.0 = no change)
    noise_reduction: bool = False,  # apply pyrnnoise to input before inference
    sola_crossfade_ms: int = _SOLA_BUF_MS_DEFAULT,  # SOLA crossfade buffer length in ms
    f0_norm_params: "dict | None" = None,
    # f0_norm_params keys: src_mean, src_std, tgt_mean, tgt_std  (all floats)
    # When set, applies mean-std log-F0 normalisation instead of a global semitone shift.
    # pitch parameter is ignored when f0_norm_params is provided.
    reference_audio_bytes: "bytes | None" = None,  # raw audio bytes for reference encoder
) -> np.ndarray:
    """Run chunked RVC inference on a full audio array.

    Mirrors the realtime worker block loop exactly so that inference quality
    is identical between offline and realtime modes.

    Returns float32 mono audio at tgt_sr (32 kHz for our models).
    """
    project_root = os.environ.get("PROJECT_ROOT", "")
    rvc_pkg_dir = os.path.join(project_root, "backend", "rvc")
    if rvc_pkg_dir not in sys.path:
        sys.path.insert(0, rvc_pkg_dir)

    try:
        from configs.config import Config

        Config.use_insecure_load()
    except Exception:
        pass

    from infer.lib.rtrvc import RVC
    from torchaudio.transforms import Resample

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # fp16 on CUDA only — MPS and CPU require fp32
    is_half = device.startswith("cuda")

    rvc = RVC(
        key=0 if f0_norm_params else pitch,
        formant=0,
        pth_path=model_path,
        index_path=index_path,
        index_rate=index_rate,
        device=device,
        is_half=is_half,
    )
    tgt_sr = rvc.tgt_sr

    # Reference encoder: encode reference clip once, or use profile mean
    if reference_audio_bytes and rvc.has_reference_encoder:
        try:
            import io as _io
            import torchaudio as _ta
            _ref_wav, _ref_sr = _ta.load(_io.BytesIO(reference_audio_bytes))
            if _ref_wav.shape[0] > 1:
                _ref_wav = _ref_wav.mean(0, keepdim=True)
            if _ref_sr != 16000:
                _ref_wav = _ta.transforms.Resample(_ref_sr, 16000)(_ref_wav)
            rvc.set_reference(_ref_wav.squeeze(0).numpy())
        except Exception as _re:
            logger.warning("offline RVC: failed to encode reference audio: %s", _re)

    # Resamplers
    resample_16_tgt = Resample(
        orig_freq=_SR_16K, new_freq=tgt_sr, dtype=torch.float32
    ).to(device)
    resample_tgt_16 = Resample(
        orig_freq=tgt_sr, new_freq=_SR_16K, dtype=torch.float32
    ).to(device)

    bp = _compute_block_params(sola_buf_ms=sola_crossfade_ms)
    block_16k = bp["block_16k"]
    total_16k_samples = bp["total_16k_samples"]
    skip_head = bp["skip_head_frames"]
    return_length = bp["return_length_frames"]
    sola_buf_48k = bp["sola_buf_48k"]
    sola_search_48k = bp["sola_search_48k"]
    block_tgt = int(bp["block_48k"] * tgt_sr / _SR_48K)
    sola_buf_tgt = int(sola_buf_48k * tgt_sr / _SR_48K)
    sola_search_tgt = int(sola_search_48k * tgt_sr / _SR_48K)

    # Patch _get_f0: head-padding + F0 normalization pipeline (affine → HistEQ → vel → soft-clip)
    _orig_get_f0 = rvc._get_f0
    p_len_expected = bp["p_len"]
    _f0_norm = f0_norm_params  # captured in closure
    # Offline: build a one-shot VelocityNormalizer here (processes all blocks sequentially)
    _vel_state_offline = None
    if _f0_norm is not None:
        vel_ratio = _f0_norm.get("vel_ratio")
        if vel_ratio is not None and abs(float(vel_ratio) - 1.0) > 1e-4:
            from backend.app.f0_transform import VelocityNormalizer as _VN
            _vel_state_offline = _VN(float(vel_ratio))

    def _patched_get_f0(x, f0_up_key, filter_radius=None, method="rmvpe"):
        c, f = _orig_get_f0(x, f0_up_key, filter_radius=filter_radius, method=method)
        if c.shape[-1] < p_len_expected:
            pad = p_len_expected - c.shape[-1]
            c = torch.cat([torch.zeros(pad, dtype=c.dtype, device=c.device), c])
            f = torch.cat([torch.zeros(pad, dtype=f.dtype, device=f.device), f])
        if _f0_norm is not None:
            import math as _m
            from backend.app.f0_transform import apply_f0_prior_hz
            src_mean = float(_f0_norm["src_mean"])
            src_std  = float(_f0_norm["src_std"])
            tgt_mean = float(_f0_norm["tgt_mean"])
            tgt_std  = float(_f0_norm["tgt_std"])
            ratio = (tgt_std / src_std) if src_std > 1e-6 else 1.0
            log_mu_s = _m.log(src_mean) if src_mean > 0 else 0.0
            log_mu_t = _m.log(tgt_mean) if tgt_mean > 0 else 0.0
            # Step 1: Affine normalization in log-Hz space
            f_np = f.cpu().float().numpy()
            voiced_mask = f_np > 0
            if voiced_mask.any():
                log_f = np.log(np.where(voiced_mask, f_np, 1.0))
                log_f_norm = log_mu_t + ratio * (log_f - log_mu_s)
                f_np = np.where(voiced_mask, np.exp(log_f_norm), 0.0).astype(np.float32)
            # Steps 2-4: HistEQ + velocity norm + soft-clip (from f0_transform module)
            f_np = apply_f0_prior_hz(f_np, _f0_norm, vel_state=_vel_state_offline)
            f = torch.from_numpy(f_np).to(f.device)
            # Recompute coarse mel-bin from normalized Hz
            f_np_c = f.cpu().numpy()
            coarse = np.zeros_like(f_np_c, dtype=np.int64)
            voiced_c = f_np_c > 0
            coarse[voiced_c] = np.clip(
                np.round(np.log2(f_np_c[voiced_c] / 32.7) * (255.0 / 7.5)),
                1, 255,
            ).astype(np.int64)
            c = torch.from_numpy(coarse).to(c.device)
        return c, f

    # FAISS fix
    if hasattr(rvc, "index"):
        import faiss  # noqa: F401

        _orig_search = rvc.index.search

        def _safe_search(npy, k):
            return _orig_search(np.ascontiguousarray(npy, dtype=np.float32), k=k)

        rvc.index.search = _safe_search

    # Warmup
    _silence = torch.zeros(total_16k_samples, device=device)
    rvc.infer(_silence, block_16k, skip_head, return_length, "rmvpe", protect)
    del _silence

    # ── pyrnnoise denoising (whole-file, before block loop) ──────────────────
    # pyrnnoise operates at 48kHz.  Upsample → denoise → downsample back to 16k.
    # Done here (not per-block) so the GRU context spans the full file naturally.
    # audio_16k_orig is kept for RMS matching so denoising doesn't affect level.
    audio_16k_orig = audio_16k
    if noise_reduction:
        _resamp_up   = torch.nn.functional.interpolate(
            torch.from_numpy(audio_16k)[None, None, :],
            scale_factor=3.0, mode="linear", align_corners=False,
        ).squeeze().numpy()
        _denoised_48k = _apply_rnnoise(_resamp_up)
        _resamp_down = torch.nn.functional.interpolate(
            torch.from_numpy(_denoised_48k)[None, None, :],
            size=len(audio_16k), mode="linear", align_corners=False,
        ).squeeze().numpy().astype(np.float32)
        audio_16k = _resamp_down

    rolling_buf = torch.zeros(total_16k_samples, device=device)

    # SOLA state — sin² fade windows, same as realtime worker
    fade_in_win = (
        torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, sola_buf_tgt, device=device))
        ** 2
    )
    fade_out_win = 1.0 - fade_in_win
    sola_buf = torch.zeros(sola_buf_tgt, device=device)
    output_parts: list[np.ndarray] = []

    # Pad input at end so last block is full
    pad_16k = np.zeros(block_16k, dtype=np.float32)
    audio_padded = np.concatenate([audio_16k, pad_16k])
    n_blocks = max(1, int(np.ceil(len(audio_16k) / block_16k)))

    for block_idx in range(n_blocks):
        off = block_idx * block_16k
        chunk_np = audio_padded[off : off + block_16k]
        chunk_t = torch.from_numpy(chunk_np).to(device)

        # Shift rolling buffer left, append new block
        rolling_buf[:-block_16k] = rolling_buf[block_16k:].clone()
        rolling_buf[-block_16k:] = chunk_t

        infer_out = rvc.infer(
            rolling_buf, block_16k, skip_head, return_length, "rmvpe", protect
        )
        infer_tgt = infer_out

        # SOLA: find best alignment in search window
        if infer_tgt.shape[0] >= sola_buf_tgt + sola_search_tgt:
            search_region = infer_tgt[: sola_buf_tgt + sola_search_tgt]
            # cross-correlate sola_buf against search_region
            if sola_buf_tgt > 0 and search_region.shape[0] > sola_buf_tgt:
                try:
                    cor_nom = torch.nn.functional.conv1d(
                        search_region[None, None, :],
                        sola_buf[None, None, :],
                        padding=0,
                    ).squeeze()
                    cor_den = (
                        torch.nn.functional.conv1d(
                            search_region[None, None, :] ** 2,
                            torch.ones(1, 1, sola_buf_tgt, device=device),
                            padding=0,
                        ).squeeze()
                        + 1e-8
                    )
                    sola_offset = int(torch.argmax(cor_nom / cor_den).item())
                except Exception:
                    sola_offset = 0
            else:
                sola_offset = 0

            # Extract the aligned output block
            out_start = sola_offset
            out_block = infer_tgt[out_start : out_start + block_tgt + sola_buf_tgt]
            if out_block.shape[0] < block_tgt + sola_buf_tgt:
                out_block = torch.nn.functional.pad(
                    out_block, (0, block_tgt + sola_buf_tgt - out_block.shape[0])
                )

            # Crossfade leading sola_buf_tgt samples with saved buf
            out_block[:sola_buf_tgt] = (
                sola_buf * fade_out_win + out_block[:sola_buf_tgt] * fade_in_win
            )
            sola_buf = out_block[block_tgt : block_tgt + sola_buf_tgt].clone()
            block_out = out_block[:block_tgt]
            output_parts.append(block_out.cpu().numpy())
        else:
            # infer_out too short (end of file) — use as-is, update sola_buf from tail
            tail = infer_tgt
            if tail.shape[0] > sola_buf_tgt:
                sola_buf[:] = tail[-sola_buf_tgt:]
            block_out = tail
            output_parts.append(block_out.cpu().numpy())

        progress_cb((block_idx + 1) / n_blocks)

    if output_parts:
        result = np.concatenate(output_parts).astype(np.float32)
    else:
        result = np.zeros(0, dtype=np.float32)

    # Trim to original length at tgt_sr
    total_16k = len(audio_16k_orig)
    expected_len = int(np.ceil(total_16k * tgt_sr / _SR_16K))
    result = result[:expected_len]

    # RMS matching — corrects the model's inherent output level difference so
    # the output matches input loudness.  This is not a gain addition; the model
    # naturally outputs at a different RMS than the input (measured ~3.8× higher).
    # Without this the output is significantly louder than the source file.
    if len(result) > 0 and len(audio_16k) > 0:
        in_rms = float(np.sqrt(np.mean(audio_16k_orig**2)) + 1e-8)
        out_rms = float(np.sqrt(np.mean(result**2)) + 1e-8)
        rms_scale = in_rms / out_rms
        rms_scale = float(np.clip(rms_scale, 0.1, 10.0))
        result = result * rms_scale

    if output_gain != 1.0 and len(result) > 0:
        result = result * output_gain

    # ── Explicit GPU cleanup ──────────────────────────────────────────────────
    # RVC loads synthesizer + embedder + RMVPE onto GPU.  Without explicit
    # deletion and cache flush, every offline call leaks all three models on
    # the GPU for the lifetime of the process, exhausting VRAM across calls.
    # Use try/finally so cleanup also runs when an exception is raised above.
    try:
        return result, tgt_sr
    finally:
        del rvc
        del resample_16_tgt
        del resample_tgt_16
        del rolling_buf
        del sola_buf
        del fade_in_win
        del fade_out_win
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# SSE convert endpoint
# ---------------------------------------------------------------------------


@router.post("/convert")
async def convert_audio(
    profile_id: str = Form(...),
    pitch: float = Form(0.0),
    index_rate: float = Form(0.50),
    protect: float = Form(0.33),
    use_best: bool = Form(False),
    noise_reduction: bool = Form(False),
    sola_crossfade_ms: int = Form(_SOLA_BUF_MS_DEFAULT),
    pitch_shift_semitones: float = Form(0.0),
    formant_shift_semitones: float = Form(0.0),
    # F0 normalization — all four must be non-zero to activate normalization.
    f0_src_mean: float = Form(0.0),
    f0_src_std:  float = Form(0.0),
    f0_tgt_mean: float = Form(0.0),
    f0_tgt_std:  float = Form(0.0),
    # Extended F0 prior — velocity normalization
    f0_vel_src: float = Form(0.0),
    f0_vel_tgt: float = Form(0.0),
    # Extended F0 prior — soft-clip range
    f0_p5_tgt:  float = Form(0.0),
    f0_p95_tgt: float = Form(0.0),
    # Extended F0 prior — histogram equalization
    f0_src_hist: str = Form(""),
    f0_tgt_hist: str = Form(""),
    # Reference encoder — optional reference audio clip (5–10 s, 16 kHz or any SR)
    # When provided, the engine encodes it to a style vector that conditions synthesis.
    # When absent, the engine uses the profile's pre-computed mean style vector.
    reference_audio: Optional[UploadFile] = File(default=None),
    file: UploadFile = File(...),
):
    """Convert an uploaded audio file using a trained profile.

    Streams Server-Sent Events with progress updates:
      data: {"type": "progress", "fraction": 0.0..1.0}
      data: {"type": "done", "job_id": "...", "duration_s": 12.3}
      data: {"type": "error", "message": "..."}
    """
    # Load profile
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, name, profile_dir, pipeline, model_path FROM profiles WHERE id = ?",
            (profile_id,),
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")

    profile_pipeline = str(row["pipeline"]) if row["pipeline"] else "rvc"
    pdir = row["profile_dir"]
    model_path = None
    index_path = None
    if pdir and os.path.isdir(pdir):
        if profile_pipeline == "beatrice2":
            # Beatrice 2: prefer checkpoint_best when use_best is set, else latest
            b2_dir = os.path.join(pdir, "beatrice2_out")
            if use_best:
                best_ckpt = os.path.join(b2_dir, "checkpoint_best.pt.gz")
                latest_ckpt = os.path.join(b2_dir, "checkpoint_latest.pt.gz")
                b2_ckpt = best_ckpt if os.path.exists(best_ckpt) else latest_ckpt
            else:
                b2_ckpt = os.path.join(b2_dir, "checkpoint_latest.pt.gz")
            if os.path.exists(b2_ckpt):
                model_path = b2_ckpt
        else:
            # Resolve model: prefer model_best.pth when use_best is set and file
            # exists; fall back to model_infer.pth (latest) otherwise.
            if use_best:
                best_candidate = os.path.join(pdir, "model_best.pth")
                mp = best_candidate if os.path.exists(best_candidate) else os.path.join(pdir, "model_infer.pth")
            else:
                mp = os.path.join(pdir, "model_infer.pth")
            if os.path.exists(mp):
                model_path = mp
            import glob

            matches = glob.glob(os.path.join(pdir, "*.index"))
            if matches:
                index_path = matches[0]

    if not model_path:
        raise HTTPException(status_code=400, detail="Profile has no trained model yet")
    if profile_pipeline != "beatrice2" and not index_path:
        raise HTTPException(status_code=400, detail="Profile has no FAISS index yet")

    # Read input audio
    audio_bytes = await file.read()
    # Read reference audio bytes now (before the async generator starts;
    # UploadFile handle is not available inside the streaming generator).
    reference_audio_bytes: Optional[bytes] = None
    if reference_audio is not None:
        reference_audio_bytes = await reference_audio.read()
        if not reference_audio_bytes:
            reference_audio_bytes = None
    orig_filename = file.filename or "input.wav"
    orig_ext = os.path.splitext(orig_filename)[1].lower() or ".wav"
    # Always output WAV to preserve full quality regardless of input format.
    out_ext = ".wav"

    try:
        buf = io.BytesIO(audio_bytes)
        audio_data, orig_sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read audio file: {e}")

    # Mono + resample to 16k for inference
    if audio_data.ndim == 2:
        audio_mono = audio_data.mean(axis=1)
    else:
        audio_mono = audio_data

    if orig_sr != _SR_16K:
        resamp = torch.from_numpy(audio_mono).unsqueeze(0)
        from torchaudio.transforms import Resample as _R

        resamp = _R(orig_freq=orig_sr, new_freq=_SR_16K)(resamp).squeeze(0)
        audio_16k = resamp.numpy()
    else:
        audio_16k = audio_mono

    job_id = uuid.uuid4().hex

    # Save input to temp/ for later analysis
    in_path = _input_path(job_id, orig_ext)
    with open(in_path, "wb") as _f:
        _f.write(audio_bytes)

    _jobs[job_id] = {"status": "running", "input_path": in_path, "profile_id": profile_id}

    import json as _json

    async def _event_stream():
        loop = asyncio.get_running_loop()
        progress_queue: asyncio.Queue = asyncio.Queue()

        def _progress(fraction: float):
            loop.call_soon_threadsafe(progress_queue.put_nowait, fraction)

        # ---- Beatrice 2 inference ----
        if profile_pipeline == "beatrice2":
            b2_checkpoint = row["model_path"] if row["model_path"] else None
            if not b2_checkpoint or not os.path.exists(b2_checkpoint):
                # Try default location
                if pdir:
                    b2_checkpoint = os.path.join(pdir, "beatrice2_out", "checkpoint_latest.pt.gz")
            if not b2_checkpoint or not os.path.exists(b2_checkpoint):
                yield 'data: {"type":"error","message":"No Beatrice 2 checkpoint found. Train first."}\n\n'
                return

            def _run_b2():
                try:
                    from backend.beatrice2.inference import get_or_load_engine
                    import torch
                    import numpy as np
                    import json as _json
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    engine = get_or_load_engine(profile_id, b2_checkpoint, device)
                    _progress(0.1)

                    # Reference encoder: encode the reference audio clip if supplied.
                    # Done once here; the engine caches it for all subsequent chunks.
                    if reference_audio_bytes is not None and engine.has_reference_encoder:
                        try:
                            import io as _io
                            import torchaudio as _ta
                            _ref_wav, _ref_sr = _ta.load(_io.BytesIO(reference_audio_bytes))
                            if _ref_wav.shape[0] > 1:
                                _ref_wav = _ref_wav.mean(0, keepdim=True)
                            if _ref_sr != 16000:
                                _ref_wav = _ta.transforms.Resample(_ref_sr, 16000)(_ref_wav)
                            engine.set_reference(_ref_wav.squeeze(0).numpy())
                            logger.info("offline B2: reference encoder set from uploaded clip (%d samples)", _ref_wav.shape[-1])
                        except Exception as _re:
                            logger.warning("offline B2: failed to encode reference audio: %s", _re)

                    # Build norm params dict if all four stats are present
                    _norm = None
                    if (f0_src_mean > 0 and f0_tgt_mean > 0
                            and f0_src_std > 1e-6 and f0_tgt_std > 1e-6):
                        _norm = {
                            "src_mean": f0_src_mean,
                            "src_std":  f0_src_std,
                            "tgt_mean": f0_tgt_mean,
                            "tgt_std":  f0_tgt_std,
                        }
                        if f0_vel_src > 1e-6 and f0_vel_tgt > 1e-6:
                            _norm["vel_ratio"] = f0_vel_tgt / f0_vel_src
                        if f0_p5_tgt > 0 and f0_p95_tgt > f0_p5_tgt:
                            _norm["p5_tgt"]  = f0_p5_tgt
                            _norm["p95_tgt"] = f0_p95_tgt
                        if f0_src_hist and f0_tgt_hist:
                            try:
                                _norm["src_hist"] = _json.loads(f0_src_hist)
                                _norm["tgt_hist"] = _json.loads(f0_tgt_hist)
                            except Exception:
                                pass

                    result = engine.convert_offline(
                        audio_16k,
                        target_speaker_id=0,
                        pitch_shift_semitones=float(pitch_shift_semitones),
                        formant_shift_semitones=float(formant_shift_semitones),
                        f0_norm_params=_norm,
                    )
                    # Clear session reference after offline use (stateless per call)
                    engine.set_reference(None)
                    # Peak normalise to 0.95 to prevent clipping on save
                    peak = np.abs(result).max()
                    if peak > 0.95:
                        result = result * (0.95 / peak)
                    _progress(0.9)
                    tgt_sr = engine.OUT_SR
                    out_path = _output_path(job_id, out_ext)
                    _write_audio(out_path, result, tgt_sr, out_ext)
                    _jobs[job_id] = {
                        "status": "done",
                        "input_path": in_path,
                        "result_path": out_path,
                        "sample_rate": tgt_sr,
                        "orig_filename": orig_filename,
                        "pipeline": "beatrice2",
                    }
                    loop.call_soon_threadsafe(progress_queue.put_nowait, "done")
                except Exception as exc:
                    logger.exception("beatrice2 offline inference failed")
                    _jobs[job_id] = {"status": "error", "error": str(exc), "input_path": in_path}
                    loop.call_soon_threadsafe(progress_queue.put_nowait, f"error:{exc}")

            loop.run_in_executor(None, _run_b2)
        else:
            # ---- RVC inference ----
            def _run():
                try:
                    import json as _json
                    # Build F0 normalization params if all four values are provided
                    _norm = None
                    if (f0_src_mean > 0 and f0_tgt_mean > 0
                            and f0_src_std > 1e-6 and f0_tgt_std > 1e-6):
                        _norm = {
                            "src_mean": f0_src_mean,
                            "src_std":  f0_src_std,
                            "tgt_mean": f0_tgt_mean,
                            "tgt_std":  f0_tgt_std,
                        }
                        # Velocity normalization
                        if f0_vel_src > 1e-6 and f0_vel_tgt > 1e-6:
                            _norm["vel_ratio"] = f0_vel_tgt / f0_vel_src
                        # Soft-clip
                        if f0_p5_tgt > 0 and f0_p95_tgt > f0_p5_tgt:
                            _norm["p5_tgt"]  = f0_p5_tgt
                            _norm["p95_tgt"] = f0_p95_tgt
                        # HistEQ (offline only — full file available)
                        if f0_src_hist and f0_tgt_hist:
                            try:
                                _norm["src_hist"] = _json.loads(f0_src_hist)
                                _norm["tgt_hist"] = _json.loads(f0_tgt_hist)
                            except Exception:
                                pass
                    result_audio, tgt_sr = _run_offline_inference(
                        audio_16k=audio_16k,
                        audio_tgt=audio_16k,  # unused placeholder
                        model_path=model_path,
                        index_path=index_path,
                        pitch=pitch,
                        index_rate=index_rate,
                        protect=protect,
                        progress_cb=_progress,
                        noise_reduction=noise_reduction,
                        sola_crossfade_ms=sola_crossfade_ms,
                        f0_norm_params=_norm,
                        reference_audio_bytes=reference_audio_bytes,
                    )
                    # Write result to temp file preserving original format
                    out_path = _output_path(job_id, out_ext)
                    _write_audio(out_path, result_audio, tgt_sr, out_ext)
                    _jobs[job_id] = {
                        "status": "done",
                        "input_path": in_path,
                        "result_path": out_path,
                        "sample_rate": tgt_sr,
                        "orig_filename": orig_filename,
                        "pipeline": "rvc",
                    }
                    loop.call_soon_threadsafe(progress_queue.put_nowait, "done")
                except Exception as exc:
                    logger.exception("offline inference failed")
                    _jobs[job_id] = {"status": "error", "error": str(exc), "input_path": in_path}
                    loop.call_soon_threadsafe(progress_queue.put_nowait, f"error:{exc}")

            # Run inference in thread pool (it's CPU/GPU intensive, blocks event loop)
            loop.run_in_executor(None, _run)

        while True:
            try:
                item = await asyncio.wait_for(progress_queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                yield 'data: {"type":"error","message":"timeout"}\n\n'
                return

            if isinstance(item, float):
                yield f"data: {_json.dumps({'type': 'progress', 'fraction': round(item, 3)})}\n\n"
            elif item == "done":
                job = _jobs[job_id]
                yield f"data: {_json.dumps({'type': 'done', 'job_id': job_id, 'result_path': job.get('result_path', '')})}\n\n"
                return
            elif isinstance(item, str) and item.startswith("error:"):
                msg = item[6:]
                yield f"data: {_json.dumps({'type': 'error', 'message': msg})}\n\n"
                return

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Result download
# ---------------------------------------------------------------------------


@router.get("/result/{job_id}")
async def download_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")

    result_path = job["result_path"]
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    orig = job.get("orig_filename", "output.wav")
    stem = os.path.splitext(orig)[0]
    ext = os.path.splitext(result_path)[1].lower() or ".wav"
    suffix = "_b2" if job.get("pipeline") == "beatrice2" else "_rvc"
    download_name = f"{stem}{suffix}{ext}"
    mime = _AUDIO_MIME.get(ext, "application/octet-stream")

    return FileResponse(
        result_path,
        media_type=mime,
        filename=download_name,
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )


# ---------------------------------------------------------------------------
# Input audio streaming (for waveform on analysis page)
# ---------------------------------------------------------------------------

@router.get("/input/{job_id}")
async def stream_input(job_id: str):
    """Stream the original input audio for a completed job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    in_path = job.get("input_path")
    if not in_path or not os.path.exists(in_path):
        raise HTTPException(status_code=404, detail="Input file not found or expired")

    ext = os.path.splitext(in_path)[1].lower() or ".wav"
    mime = _AUDIO_MIME.get(ext, "audio/wav")
    return FileResponse(in_path, media_type=mime, headers={"Cache-Control": "no-cache"})


@router.post("/speaker_f0/{profile_id}/compute")
async def compute_speaker_f0(profile_id: str):
    """Compute full F0 statistics from profile audio files and save to speaker_f0.json.

    Works for both RVC (audio in profile_dir/audio/) and Beatrice 2
    (audio in profile_dir/beatrice2_data/<speaker>/). Uses pyworld DIO.

    Returns full stats: mean_f0, std_f0, p5_f0, p25_f0, p50_f0, p75_f0, p95_f0,
    vel_std, voiced_rate, n_frames, f0_hist (256-bucket CDF histogram), method.
    """
    import json as _json
    import numpy as _np
    import torch as _torch
    import torchaudio as _torchaudio
    from backend.app.f0_transform import compute_f0_statistics

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT profile_dir, pipeline FROM profiles WHERE id = ?", (profile_id,)
        )
        row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")

    pdir = row["profile_dir"]
    pipeline = row["pipeline"] or "rvc"
    if not pdir or not os.path.isdir(pdir):
        raise HTTPException(status_code=404, detail="Profile directory not found")

    # Collect audio files
    EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    audio_files: list[str] = []
    if pipeline == "beatrice2":
        data_dir = os.path.join(pdir, "beatrice2_data")
        if os.path.isdir(data_dir):
            for root, _, files in os.walk(data_dir):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in EXTS and "_originals" not in root:
                        audio_files.append(os.path.join(root, f))
    if not audio_files:
        # RVC or fallback: profile_dir/audio/
        audio_dir = os.path.join(pdir, "audio")
        if os.path.isdir(audio_dir):
            audio_files = sorted([
                os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
                if os.path.splitext(f)[1].lower() in EXTS
            ])

    if not audio_files:
        raise HTTPException(status_code=422, detail="No audio files found in profile")

    def _compute():
        import pyworld as _pyworld
        all_log_f0: list[float] = []
        total_frames = 0
        for af in audio_files:
            try:
                wav, sr = _torchaudio.load(af)
                mono = wav.mean(0).numpy().astype(_np.float64)
                if sr != 16000:
                    mono = _torchaudio.functional.resample(
                        _torch.from_numpy(mono), sr, 16000
                    ).numpy().astype(_np.float64)
                    sr = 16000
                f0, _ = _pyworld.dio(mono, sr, f0_floor=50.0, f0_ceil=800.0)
                total_frames += len(f0)
                voiced = f0[f0 > 0]
                if len(voiced):
                    all_log_f0.extend(_np.log(voiced).tolist())
            except Exception:
                pass
        if not all_log_f0:
            raise ValueError("No voiced frames found in audio files")
        stats = compute_f0_statistics(_np.array(all_log_f0, dtype=_np.float64))
        voiced_rate = len(all_log_f0) / total_frames if total_frames > 0 else 0.0
        stats["voiced_rate"] = round(voiced_rate, 4)
        stats["n_frames"] = len(all_log_f0)
        stats["method"] = "dio"
        return stats

    loop = asyncio.get_event_loop()
    try:
        payload = await loop.run_in_executor(None, _compute)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Save to appropriate location
    if pipeline == "beatrice2":
        out_dir = os.path.join(pdir, "beatrice2_out")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "speaker_f0.json")
    else:
        save_path = os.path.join(pdir, "speaker_f0.json")

    import json as _json
    with open(save_path, "w") as f:
        _json.dump(payload, f)

    return payload


@router.get("/speaker_f0/{profile_id}")
async def get_speaker_f0(profile_id: str):
    """Return the profile's stored mean F0 (Hz) for auto pitch-shift computation.

    Looks for speaker_f0.json in:
      - profile_dir/speaker_f0.json          (RVC)
      - profile_dir/beatrice2_out/speaker_f0.json  (Beatrice 2)
    Returns {"mean_f0": float, "method": str} or 404 if not computed yet.
    """
    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT profile_dir, pipeline FROM profiles WHERE id = ?", (profile_id,)
        )
        row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")

    pdir = row["profile_dir"]
    pipeline = row["pipeline"] or "rvc"

    candidates = [
        os.path.join(pdir, "speaker_f0.json"),
        os.path.join(pdir, "beatrice2_out", "speaker_f0.json"),
    ]
    import json as _json
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return _json.load(f)

    raise HTTPException(status_code=404, detail="speaker_f0.json not found — train first")


@router.get("/reference/{job_id}")
async def stream_reference(job_id: str):
    """Stream the profile reference audio for a completed job (for waveform preview)."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    ref_path = job.get("ref_path")
    if not ref_path or not os.path.exists(ref_path):
        raise HTTPException(status_code=404, detail="Reference audio not available")

    ext = os.path.splitext(ref_path)[1].lower() or ".wav"
    mime = _AUDIO_MIME.get(ext, "audio/wav")
    return FileResponse(ref_path, media_type=mime, headers={"Cache-Control": "no-cache"})

class _AnalyzeJobResponse(dict):
    pass


@router.post("/analyze/{job_id}")
async def analyze_job(job_id: str, profile_id: Optional[str] = None):
    """Run ECAPA-TDNN speaker-embedding comparison on a completed job.

    File A = profile reference audio (first cleaned audio file, or sample_path fallback).
    File B = converted output file.
    profile_id may be passed as a query param to override what's stored in _jobs.
    Returns CompareResponse-shaped JSON.
    """
    from backend.app.voice_analysis import analyze_pair

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job not done (status: {job['status']})")

    out_path = job.get("result_path")
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="Output temp file not found or expired")

    # Resolve reference audio from profile — prefer query param, fall back to _jobs
    resolved_profile_id = profile_id or job.get("profile_id")
    ref_path: Optional[str] = None
    if resolved_profile_id:
        async with aiosqlite.connect(db.DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            # Prefer a cleaned audio file
            cursor = await conn.execute(
                "SELECT file_path FROM audio_files WHERE profile_id = ? AND is_cleaned = 1 ORDER BY created_at ASC LIMIT 1",
                (resolved_profile_id,),
            )
            row = await cursor.fetchone()
            if row and os.path.exists(row["file_path"]):
                ref_path = row["file_path"]
            else:
                # Fallback: any audio file for this profile
                cursor = await conn.execute(
                    "SELECT file_path FROM audio_files WHERE profile_id = ? ORDER BY created_at ASC LIMIT 1",
                    (resolved_profile_id,),
                )
                row = await cursor.fetchone()
                if row and os.path.exists(row["file_path"]):
                    ref_path = row["file_path"]
                else:
                    # Last resort: sample_path on profile row
                    cursor = await conn.execute(
                        "SELECT sample_path FROM profiles WHERE id = ?",
                        (resolved_profile_id,),
                    )
                    row = await cursor.fetchone()
                    if row and row["sample_path"] and os.path.exists(row["sample_path"]):
                        ref_path = row["sample_path"]

    if not ref_path:
        raise HTTPException(
            status_code=404,
            detail="No reference audio found for this profile. Upload training audio first.",
        )

    # Cache for waveform preview endpoint
    job["ref_path"] = ref_path

    device = os.environ.get("RVC_DEVICE", "cpu")

    try:
        result = analyze_pair(path_a=ref_path, path_b=out_path, device=device)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Analysis failed for job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return result


# ---------------------------------------------------------------------------
# POST /api/offline/input_f0  — compute mean F0 of an uploaded audio file
# ---------------------------------------------------------------------------

@router.post("/input_f0")
async def compute_input_f0(file: UploadFile = File(...)):
    """Compute full F0 statistics for an uploaded audio file using pyworld DIO.

    Mirrors the same algorithm used for profile F0 computation so both sides
    of the normalization formula are on equal footing.

    Returns full stats: mean_f0, std_f0, p5_f0, p25_f0, p50_f0, p75_f0, p95_f0,
    vel_std, voiced_rate, n_frames, f0_hist (256-bucket CDF), method.
    Raises 422 if no voiced frames are found.
    """
    import numpy as _np
    import torch as _torch
    import torchaudio as _torchaudio
    import pyworld as _pyworld
    from backend.app.f0_transform import compute_f0_statistics

    audio_bytes = await file.read()
    try:
        buf = io.BytesIO(audio_bytes)
        audio_data, orig_sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read audio file: {e}")

    if audio_data.ndim == 2:
        audio_mono = audio_data.mean(axis=1)
    else:
        audio_mono = audio_data

    if orig_sr != _SR_16K:
        t = _torch.from_numpy(audio_mono).unsqueeze(0)
        from torchaudio.transforms import Resample as _R
        t = _R(orig_freq=orig_sr, new_freq=_SR_16K)(t).squeeze(0)
        audio_mono = t.numpy()

    def _run():
        mono64 = audio_mono.astype(_np.float64)
        f0, _ = _pyworld.dio(mono64, _SR_16K, f0_floor=50.0, f0_ceil=800.0)
        voiced = f0[f0 > 0]
        if len(voiced) == 0:
            raise ValueError("No voiced frames found — is the file speech?")
        log_v = _np.log(voiced)
        stats = compute_f0_statistics(log_v)
        voiced_rate = len(voiced) / len(f0) if len(f0) > 0 else 0.0
        stats["voiced_rate"] = round(voiced_rate, 4)
        stats["n_frames"] = int(len(voiced))
        stats["method"] = "dio"
        return stats

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result

