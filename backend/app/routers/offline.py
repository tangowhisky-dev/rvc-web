"""Offline (file-based) voice conversion router.

Endpoints:
  POST  /api/offline/convert   — convert an uploaded audio file, stream SSE progress
  GET   /api/offline/result/{job_id}  — download the converted file
  GET   /api/offline/profiles  — list trained profiles available for inference

Uses the same RVC pipeline as the realtime worker (rtrvc.RVC) so inference
quality is identical and can be used to diagnose cloning quality.
"""

import asyncio
import io
import logging
import os
import sys
import tempfile
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

logger = logging.getLogger("rvc_web.offline")

router = APIRouter(prefix="/api/offline", tags=["offline"])

# ---------------------------------------------------------------------------
# In-memory job store (results held until downloaded or server restart)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}  # job_id → {status, result_path, error, sample_rate, fmt}


# ---------------------------------------------------------------------------
# Profile list — re-uses same DB query as training page
# ---------------------------------------------------------------------------

class ProfileOut(BaseModel):
    id: str
    name: str
    model_path: Optional[str]
    index_path: Optional[str]
    total_epochs_trained: int


@router.get("/profiles", response_model=list[ProfileOut])
async def list_profiles():
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT id, name, profile_dir, checkpoint_path, total_epochs_trained
               FROM profiles
               WHERE status != 'untrained'
               ORDER BY created_at DESC"""
        )
        rows = await cursor.fetchall()

    out = []
    for row in rows:
        pdir = row["profile_dir"]
        # model_infer.pth is the inference-ready small model
        model_path = None
        index_path = None
        if pdir and os.path.isdir(pdir):
            mp = os.path.join(pdir, "model_infer.pth")
            if os.path.exists(mp):
                model_path = mp
            # find index
            import glob
            matches = glob.glob(os.path.join(pdir, "*.index"))
            if matches:
                index_path = matches[0]
        out.append(ProfileOut(
            id=row["id"],
            name=row["name"],
            model_path=model_path,
            index_path=index_path,
            total_epochs_trained=int(row["total_epochs_trained"] or 0),
        ))
    return out


# ---------------------------------------------------------------------------
# Inference helpers — mirrors _realtime_worker.py block pipeline exactly
# ---------------------------------------------------------------------------

_SR_16K = 16000
_SR_48K = 48000
_WINDOW = 160
_ZC     = _SR_48K // 100       # 480
_BLOCK_48K   = 9600             # 200ms blocks — same as realtime
_SOLA_BUF_48K   = 2 * _ZC      # 960
_SOLA_SEARCH_48K = _ZC          # 480
_EXTRA_TIME = 2.0               # 2s context window


def _compute_block_params(block_48k: int = _BLOCK_48K) -> dict:
    block_16k = int(block_48k * _SR_16K / _SR_48K)
    f0_min = block_16k + 800
    f0_extractor_frame = 5120 * ((f0_min - 1) // 5120 + 1) - _WINDOW
    extra_16k = (int(_EXTRA_TIME * _SR_16K) // _WINDOW) * _WINDOW
    total_16k_samples = extra_16k + f0_extractor_frame
    p_len = total_16k_samples // _WINDOW
    total_out_48k = block_48k + _SOLA_BUF_48K + _SOLA_SEARCH_48K
    return_length_frames = total_out_48k // _ZC
    skip_head_frames = p_len - return_length_frames
    return {
        "block_48k":            block_48k,
        "block_16k":            block_16k,
        "f0_extractor_frame":   f0_extractor_frame,
        "total_16k_samples":    total_16k_samples,
        "p_len":                p_len,
        "return_length_frames": return_length_frames,
        "skip_head_frames":     skip_head_frames,
        "sola_buf_48k":         _SOLA_BUF_48K,
        "sola_search_48k":      _SOLA_SEARCH_48K,
    }


def _run_offline_inference(
    audio_16k: np.ndarray,     # float32, mono, 16 kHz
    audio_tgt: np.ndarray,     # float32, mono, tgt_sr  (for SOLA output alignment)
    model_path: str,
    index_path: str,
    pitch: float,
    index_rate: float,
    protect: float,
    progress_cb,               # callable(fraction: float)
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

    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")

    rvc = RVC(
        key=pitch,
        formant=0,
        pth_path=model_path,
        index_path=index_path,
        index_rate=index_rate,
        device=device,
        is_half=False,
    )
    tgt_sr = rvc.tgt_sr

    # Resamplers
    resample_16_tgt = Resample(orig_freq=_SR_16K, new_freq=tgt_sr,
                               dtype=torch.float32).to(device)
    resample_tgt_16 = Resample(orig_freq=tgt_sr, new_freq=_SR_16K,
                               dtype=torch.float32).to(device)

    bp = _compute_block_params()
    block_16k          = bp["block_16k"]
    total_16k_samples  = bp["total_16k_samples"]
    skip_head          = bp["skip_head_frames"]
    return_length      = bp["return_length_frames"]
    sola_buf_48k       = bp["sola_buf_48k"]
    sola_search_48k    = bp["sola_search_48k"]
    block_tgt          = int(bp["block_48k"] * tgt_sr / _SR_48K)
    sola_buf_tgt       = int(sola_buf_48k   * tgt_sr / _SR_48K)
    sola_search_tgt    = int(sola_search_48k * tgt_sr / _SR_48K)

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

    rolling_buf = torch.zeros(total_16k_samples, device=device)

    # SOLA state
    fade_in_win  = torch.linspace(0.0, 1.0, sola_buf_tgt, device=device)
    fade_out_win = 1.0 - fade_in_win
    sola_buf     = torch.zeros(sola_buf_tgt, device=device)
    output_parts: list[np.ndarray] = []

    # Pad input at end so last block is full
    pad_16k = np.zeros(block_16k, dtype=np.float32)
    audio_padded = np.concatenate([audio_16k, pad_16k])
    n_blocks = max(1, int(np.ceil(len(audio_16k) / block_16k)))

    for block_idx in range(n_blocks):
        off = block_idx * block_16k
        chunk_np = audio_padded[off: off + block_16k]
        chunk_t  = torch.from_numpy(chunk_np).to(device)

        # Shift rolling buffer left, append new block
        rolling_buf[:-block_16k] = rolling_buf[block_16k:].clone()
        rolling_buf[-block_16k:] = chunk_t

        infer_out = rvc.infer(rolling_buf, block_16k, skip_head, return_length, "rmvpe", protect)
        # infer_out is at tgt_sr already (rtrvc handles resampling internally)
        # But rtrvc returns tgt_sr samples: return_length * window * (tgt_sr/16k)
        infer_tgt = infer_out  # shape: (return_length * window * tgt_sr/16k,)

        # SOLA: find best alignment in search window
        if infer_tgt.shape[0] >= sola_buf_tgt + sola_search_tgt:
            search_region = infer_tgt[:sola_buf_tgt + sola_search_tgt]
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
                        ).squeeze() + 1e-8
                    )
                    sola_offset = int(torch.argmax(cor_nom / cor_den).item())
                except Exception:
                    sola_offset = 0
            else:
                sola_offset = 0

            # Extract the aligned output block
            out_start = sola_offset
            out_block = infer_tgt[out_start: out_start + block_tgt + sola_buf_tgt]
            if out_block.shape[0] < block_tgt + sola_buf_tgt:
                out_block = torch.nn.functional.pad(
                    out_block, (0, block_tgt + sola_buf_tgt - out_block.shape[0])
                )

            # Crossfade leading sola_buf_tgt samples with saved buf
            out_block[:sola_buf_tgt] = (
                sola_buf * fade_out_win + out_block[:sola_buf_tgt] * fade_in_win
            )
            sola_buf = out_block[block_tgt: block_tgt + sola_buf_tgt].clone()
            output_parts.append(out_block[:block_tgt].cpu().numpy())
        else:
            # infer_out too short (end of file) — just use as-is
            output_parts.append(infer_tgt.cpu().numpy())

        progress_cb((block_idx + 1) / n_blocks)

    if output_parts:
        result = np.concatenate(output_parts).astype(np.float32)
    else:
        result = np.zeros(0, dtype=np.float32)

    # Trim to original length at tgt_sr
    total_16k = len(audio_16k)
    expected_len = int(np.ceil(total_16k * tgt_sr / _SR_16K))
    result = result[:expected_len]

    return result, tgt_sr


# ---------------------------------------------------------------------------
# SSE convert endpoint
# ---------------------------------------------------------------------------

@router.post("/convert")
async def convert_audio(
    profile_id: str = Form(...),
    pitch: float = Form(0.0),
    index_rate: float = Form(0.75),
    protect: float = Form(0.33),
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
            "SELECT id, name, profile_dir FROM profiles WHERE id = ?",
            (profile_id,)
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")

    pdir = row["profile_dir"]
    model_path = None
    index_path = None
    if pdir and os.path.isdir(pdir):
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
    _jobs[job_id] = {"status": "running"}

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
                )
                # Write result to temp file in original format
                tmp = tempfile.NamedTemporaryFile(
                    suffix=orig_ext, delete=False, prefix=f"rvc_offline_{job_id}_"
                )
                tmp.close()
                sf.write(tmp.name, result_audio, tgt_sr)
                _jobs[job_id] = {
                    "status": "done",
                    "result_path": tmp.name,
                    "sample_rate": tgt_sr,
                    "orig_filename": orig_filename,
                }
                loop.call_soon_threadsafe(progress_queue.put_nowait, "done")
            except Exception as exc:
                logger.exception("offline inference failed")
                _jobs[job_id] = {"status": "error", "error": str(exc)}
                loop.call_soon_threadsafe(progress_queue.put_nowait, f"error:{exc}")

        # Run inference in thread pool (it's CPU/GPU intensive, blocks event loop)
        loop.run_in_executor(None, _run)

        while True:
            try:
                item = await asyncio.wait_for(progress_queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                yield "data: {\"type\":\"error\",\"message\":\"timeout\"}\n\n"
                return

            if isinstance(item, float):
                yield f"data: {_json.dumps({'type': 'progress', 'fraction': round(item, 3)})}\n\n"
            elif item == "done":
                job = _jobs[job_id]
                yield f"data: {_json.dumps({'type': 'done', 'job_id': job_id})}\n\n"
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
    ext  = os.path.splitext(result_path)[1] or ".wav"
    download_name = f"{stem}_rvc{ext}"

    return FileResponse(
        result_path,
        media_type="audio/wav",
        filename=download_name,
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )
