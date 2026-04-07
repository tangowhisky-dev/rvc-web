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

# Formats soundfile can write natively (no ffmpeg needed)
_SF_WRITE_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


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

    Uses soundfile for lossless formats; pydub+ffmpeg for everything else
    (MP3, M4A, AAC, …).
    """
    ext = ext.lower()
    if ext in _SF_WRITE_EXTS:
        fmt_map = {
            ".wav": "WAV",
            ".flac": "FLAC",
            ".ogg": "OGG",
            ".aiff": "AIFF",
            ".aif": "AIFF",
        }
        sf.write(path, audio, sr, format=fmt_map.get(ext, "WAV"))
    else:
        # Write to a temporary WAV first, then convert with pydub
        from pydub import AudioSegment
        import tempfile as _tmp

        with _tmp.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            wav_path = t.name
        try:
            sf.write(wav_path, audio, sr, format="WAV")
            seg = AudioSegment.from_wav(wav_path)
            fmt_map = {".mp3": "mp3", ".m4a": "mp4", ".aac": "adts"}
            fmt = fmt_map.get(ext, ext.lstrip("."))
            seg.export(path, format=fmt)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


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
        key=pitch,
        formant=0,
        pth_path=model_path,
        index_path=index_path,
        index_rate=index_rate,
        device=device,
        is_half=is_half,
    )
    tgt_sr = rvc.tgt_sr

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

    # Patch _get_f0 for head-padding (same as realtime worker)
    _orig_get_f0 = rvc._get_f0
    p_len_expected = bp["p_len"]

    def _patched_get_f0(x, f0_up_key, filter_radius=None, method="rmvpe"):
        c, f = _orig_get_f0(x, f0_up_key, filter_radius=filter_radius, method=method)
        if c.shape[-1] < p_len_expected:
            pad = p_len_expected - c.shape[-1]
            c = torch.cat([torch.zeros(pad, dtype=c.dtype, device=c.device), c])
            f = torch.cat([torch.zeros(pad, dtype=f.dtype, device=f.device), f])
        return c, f

    rvc._get_f0 = _patched_get_f0

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
            "SELECT id, name, profile_dir FROM profiles WHERE id = ?", (profile_id,)
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")

    pdir = row["profile_dir"]
    model_path = None
    index_path = None
    if pdir and os.path.isdir(pdir):
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
    if not index_path:
        raise HTTPException(status_code=400, detail="Profile has no FAISS index yet")

    # Read input audio
    audio_bytes = await file.read()
    orig_filename = file.filename or "input.wav"
    orig_ext = os.path.splitext(orig_filename)[1].lower() or ".wav"

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

    _jobs[job_id] = {"status": "running", "input_path": in_path}

    import json as _json

    async def _event_stream():
        loop = asyncio.get_running_loop()
        progress_queue: asyncio.Queue = asyncio.Queue()

        def _progress(fraction: float):
            loop.call_soon_threadsafe(progress_queue.put_nowait, fraction)

        def _run():
            try:
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
                )
                # Write result to temp file preserving original format
                out_path = _output_path(job_id, orig_ext)
                _write_audio(out_path, result_audio, tgt_sr, orig_ext)
                _jobs[job_id] = {
                    "status": "done",
                    "input_path": in_path,
                    "result_path": out_path,
                    "sample_rate": tgt_sr,
                    "orig_filename": orig_filename,
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
    download_name = f"{stem}_rvc{ext}"
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


# ---------------------------------------------------------------------------
# ECAPA analysis on already-converted job
# ---------------------------------------------------------------------------

class _AnalyzeJobResponse(dict):
    pass


@router.post("/analyze/{job_id}")
async def analyze_job(job_id: str):
    """Run ECAPA-TDNN speaker-embedding comparison on a completed job.

    Both input and output files are already on disk in temp/ — no upload needed.
    Returns CompareResponse-shaped JSON.
    """
    from backend.app.voice_analysis import analyze_pair

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job not done (status: {job['status']})")

    in_path = job.get("input_path")
    out_path = job.get("result_path")

    if not in_path or not os.path.exists(in_path):
        raise HTTPException(status_code=404, detail="Input temp file not found or expired")
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="Output temp file not found or expired")

    device = os.environ.get("RVC_DEVICE", "cpu")

    try:
        result = analyze_pair(path_a=in_path, path_b=out_path, device=device)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Analysis failed for job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return result

