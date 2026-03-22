"""Training REST + WebSocket router.

Endpoints:
  GET    /api/training/hardware           — hardware info + batch size sweet spot
  POST   /api/training/start              — start a training job
  POST   /api/training/cancel             — cancel a running training job
  GET    /api/training/status/{profile_id} — poll job status
  WS     /ws/training/{profile_id}        — live log stream
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.app.db import get_db
from backend.app.training import manager

logger = logging.getLogger("rvc_web.training")

router = APIRouter(prefix="/api/training", tags=["training"])

# Separate router for WebSocket (no prefix — WS path is /ws/training/{profile_id})
ws_router = APIRouter(tags=["training"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class StartTrainingRequest(BaseModel):
    profile_id: str
    epochs: int = 200
    save_every: int = 10
    batch_size: int = 8


class CancelTrainingRequest(BaseModel):
    profile_id: str


class StartTrainingResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    phase: str
    progress_pct: int
    status: str
    error: Optional[str] = None


class HardwareInfo(BaseModel):
    total_ram_gb: float
    available_ram_gb: float
    ram_used_pct: float
    cpu_cores: int
    mps_available: bool
    mps_allocated_mb: float
    sweet_spot_batch_size: int
    max_safe_batch_size: int


# ---------------------------------------------------------------------------
# GET /api/training/hardware
# ---------------------------------------------------------------------------

@router.get("/hardware", response_model=HardwareInfo)
async def get_hardware() -> HardwareInfo:
    """Return live hardware metrics and recommended batch size sweet spot.

    Sweet spot heuristic for Apple Silicon MPS:
      - RVC v2 training with 48k keeps roughly 1.5 GB per batch in unified mem
        (model params ~500 MB + feature cache + gradients + activations)
      - Reserve 40% of total RAM for OS + frontend + backend processes
      - sweet_spot = floor((total_ram * 0.60) / 1.5)  clamped to [1, 32]
    """
    import psutil  # noqa: PLC0415
    import torch   # noqa: PLC0415

    mem = psutil.virtual_memory()
    total_gb = mem.total / 1024 ** 3
    avail_gb = mem.available / 1024 ** 3

    mps_available = torch.backends.mps.is_available()
    mps_alloc_mb = 0.0
    if mps_available:
        try:
            mps_alloc_mb = torch.mps.current_allocated_memory() / 1024 ** 2
        except Exception:
            pass

    # Sweet spot: 60% of total RAM headroom, 1.5 GB per batch step
    mem_for_training_gb = total_gb * 0.60
    sweet_spot = max(1, min(32, int(mem_for_training_gb / 1.5)))

    # Max safe: use currently *available* RAM
    max_safe = max(1, min(32, int(avail_gb * 0.80 / 1.5)))

    return HardwareInfo(
        total_ram_gb=round(total_gb, 1),
        available_ram_gb=round(avail_gb, 1),
        ram_used_pct=round(mem.percent, 1),
        cpu_cores=psutil.cpu_count(logical=False) or 1,
        mps_available=mps_available,
        mps_allocated_mb=round(mps_alloc_mb, 1),
        sweet_spot_batch_size=sweet_spot,
        max_safe_batch_size=max_safe,
    )


# ---------------------------------------------------------------------------
# POST /api/training/start
# ---------------------------------------------------------------------------

@router.post("/start", response_model=StartTrainingResponse)
async def start_training(request: StartTrainingRequest) -> StartTrainingResponse:
    """Start a training job for the given profile.

    Args:
        request: StartTrainingRequest with profile_id, epochs, save_every.

    Returns:
        StartTrainingResponse with the new job_id.

    Raises:
        HTTPException(404): Profile not found.
        HTTPException(400): Profile has no sample_path or RVC_ROOT is not set.
        HTTPException(409): A training job is already running for this profile.
    """
    # Look up profile
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, name, status, sample_path, batch_size FROM profiles WHERE id = ?",
            (request.profile_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {request.profile_id}")

    if row["sample_path"] is None:
        raise HTTPException(status_code=400, detail="Profile has no sample_path")

    rvc_root = os.environ.get("RVC_ROOT", "")
    if not rvc_root:
        raise HTTPException(status_code=400, detail="RVC_ROOT not set")
    rvc_root = os.path.abspath(rvc_root)

    # Use batch_size from request; fall back to profile's stored value
    batch_size = request.batch_size
    if row["batch_size"] is not None:
        batch_size = row["batch_size"]

    sample_dir = os.path.dirname(os.path.abspath(row["sample_path"]))

    try:
        job = manager.start_job(
            request.profile_id,
            sample_dir,
            rvc_root,
            total_epoch=request.epochs,
            save_every=request.save_every,
            batch_size=batch_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    logger.info(
        json.dumps({
            "event": "training_start",
            "profile_id": request.profile_id,
            "job_id": job.job_id,
        })
    )

    return StartTrainingResponse(job_id=job.job_id)


# ---------------------------------------------------------------------------
# POST /api/training/cancel
# ---------------------------------------------------------------------------

@router.post("/cancel")
async def cancel_training(request: CancelTrainingRequest) -> dict:
    """Cancel a running training job.

    Args:
        request: CancelTrainingRequest with profile_id.

    Returns:
        {"ok": True}

    Raises:
        HTTPException(404): No active job found for this profile.
    """
    job = manager.get_job(request.profile_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active training job for profile: {request.profile_id}",
        )

    manager.cancel_job(request.profile_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# GET /api/training/status/{profile_id}
# ---------------------------------------------------------------------------

@router.get("/status/{profile_id}", response_model=StatusResponse)
async def get_status(profile_id: str) -> StatusResponse:
    """Return current training job status for a profile.

    Args:
        profile_id: The profile to query.

    Returns:
        StatusResponse with phase, progress_pct, status, error.

    Raises:
        HTTPException(404): No active job for this profile.
    """
    job = manager.get_job(profile_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active training job for profile: {profile_id}",
        )

    return StatusResponse(
        phase=job.phase,
        progress_pct=job.progress_pct,
        status=job.status,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# WS /ws/training/{profile_id}  (registered on ws_router, no prefix)
# ---------------------------------------------------------------------------

@ws_router.websocket("/ws/training/{profile_id}")
async def training_websocket(websocket: WebSocket, profile_id: str) -> None:
    """Stream training log messages over WebSocket.

    Polls for the job to appear (up to 30s) to handle the race where a
    client connects before start_job has been called. Once the job is found,
    drains the queue forwarding each message as JSON until type=="done" or
    type=="error".

    Args:
        websocket: The WebSocket connection.
        profile_id: The profile whose training job to stream.
    """
    await websocket.accept()

    # Poll up to 30s for the job to appear (300 × 100ms)
    job = None
    for _ in range(300):
        job = manager.get_job(profile_id)
        if job is not None:
            break
        await asyncio.sleep(0.1)

    if job is None:
        await websocket.send_json({
            "type": "error",
            "message": "job not found",
            "phase": "idle",
        })
        await websocket.close()
        return

    try:
        ws_start = time.monotonic()
        while True:
            try:
                msg = await asyncio.wait_for(job.queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                # No message in 10s — send a keepalive ping so the browser
                # knows training is still running, then keep waiting.
                # Only stop if the job has already finished/failed AND the
                # queue is empty (i.e. the pipeline task has exited).
                if job.status in ("trained", "failed") and job.queue.empty():
                    # Job is done; send appropriate terminal message and stop
                    if job.status == "failed":
                        await websocket.send_json({
                            "type": "error",
                            "message": job.error or "Training failed",
                            "phase": job.phase,
                        })
                    else:
                        await websocket.send_json({"type": "done", "phase": job.phase})
                    break
                # Still running — send a silent keepalive so the browser WS
                # doesn't time out. No message: frontend skips it.
                await websocket.send_json({
                    "type": "keepalive",
                    "phase": job.phase,
                    "elapsed_s": int(time.monotonic() - ws_start),
                })
                continue

            await websocket.send_json(msg)

            if msg.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        logger.info(
            json.dumps({
                "event": "ws_disconnect",
                "profile_id": profile_id,
            })
        )

