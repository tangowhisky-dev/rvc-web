"""Training REST + WebSocket router.

Endpoints:
  GET    /api/training/hardware            — hardware info + batch size sweet spot
  POST   /api/training/start               — start a training job
  POST   /api/training/cancel              — cancel a running training job
  GET    /api/training/status/{profile_id} — poll job status
  GET    /api/training/losses/{profile_id} — historical epoch losses for chart
  WS     /ws/training/{profile_id}         — live log stream
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


class EpochLossPoint(BaseModel):
    epoch: int
    loss_mel: Optional[float] = None
    loss_gen: Optional[float] = None
    loss_disc: Optional[float] = None
    loss_fm: Optional[float] = None
    loss_kl: Optional[float] = None
    trained_at: str


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

    # All batch sizes must be powers of two — RVC's data loader and gradient
    # accumulation behave correctly only on 2^n values.
    POW2 = [1, 2, 4, 8, 16, 32]

    def nearest_pow2(x: float) -> int:
        """Largest power-of-two that does not exceed x, clamped to [1, 32]."""
        best = 1
        for p in POW2:
            if p <= x:
                best = p
        return best

    # Sweet spot: 60% of total RAM, ~1.5 GB per batch (model + grads + cache)
    sweet_spot = nearest_pow2(total_gb * 0.60 / 1.5)

    # Max safe: 80% of *currently available* RAM — must be ≥ sweet_spot
    max_safe = max(sweet_spot, nearest_pow2(avail_gb * 0.80 / 1.5))

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
        HTTPException(400): Profile has no sample_path or PROJECT_ROOT is not set.
        HTTPException(409): A training job is already running for this profile.
    """
    # Look up profile
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, name, status, sample_path, batch_size, profile_dir, total_epochs_trained, needs_retraining FROM profiles WHERE id = ?",
            (request.profile_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {request.profile_id}")

    project_root = os.environ.get("PROJECT_ROOT", "")
    if not project_root:
        raise HTTPException(status_code=400, detail="PROJECT_ROOT not set")
    project_root = os.path.abspath(project_root)

    # Resolve profile_dir — new profiles have it stored; legacy ones derive it
    stored_profile_dir = row["profile_dir"] if "profile_dir" in row.keys() else None
    if stored_profile_dir:
        profile_dir = os.path.abspath(stored_profile_dir)
    else:
        # Legacy: profiles created before profile_dir was stored — derive from PROJECT_ROOT
        profile_dir = os.path.join(project_root, "data", "profiles", request.profile_id)

    # sample_dir = profile_dir/audio/ so all audio files are picked up by
    # preprocess.py (it does os.listdir on this directory).
    # Fall back to legacy sample_path dirname for profiles without profile_dir.
    # Use abspath — preprocess.py runs with cwd=backend/rvc so relative paths
    # would resolve against the wrong directory.
    audio_dir = os.path.abspath(os.path.join(profile_dir, "audio"))
    if os.path.isdir(audio_dir):
        sample_dir = audio_dir
    elif row["sample_path"]:
        sample_dir = os.path.dirname(os.path.abspath(row["sample_path"]))
    else:
        raise HTTPException(status_code=400, detail="Profile has no audio files")

    # Verify at least one WAV exists in sample_dir
    try:
        wav_files = [f for f in os.listdir(sample_dir) if f.endswith(".wav")]
    except OSError:
        wav_files = []
    if not wav_files:
        raise HTTPException(status_code=400, detail="No audio files found in profile — add audio before training")

    # Request batch_size always wins over the profile's stored default.
    # Persist it back so the DB stays in sync with what was actually used.
    batch_size = request.batch_size
    async with get_db() as db:
        await db.execute(
            "UPDATE profiles SET batch_size = ? WHERE id = ?",
            (batch_size, request.profile_id),
        )
        await db.commit()

    try:
        job = manager.start_job(
            request.profile_id,
            sample_dir,
            project_root,
            total_epoch=request.epochs,
            save_every=request.save_every,
            batch_size=batch_size,
            profile_dir=profile_dir,
            prior_epochs=int(row["total_epochs_trained"] or 0),
            files_changed=bool(row["needs_retraining"]),
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
# ---------------------------------------------------------------------------
# GET /api/training/losses/{profile_id}
# ---------------------------------------------------------------------------

@router.get("/losses/{profile_id}", response_model=list[EpochLossPoint])
async def get_losses(profile_id: str) -> list[EpochLossPoint]:
    """Return all persisted epoch losses for a profile, ordered by epoch.

    Returns an empty list if no loss data exists yet (new profile, or
    training has never completed an epoch).
    """
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT epoch, loss_mel, loss_gen, loss_disc, loss_fm, loss_kl, trained_at
               FROM epoch_losses
               WHERE profile_id = ?
               ORDER BY epoch ASC""",
            (profile_id,),
        )
        rows = await cursor.fetchall()
    return [
        EpochLossPoint(
            epoch=r["epoch"],
            loss_mel=r["loss_mel"],
            loss_gen=r["loss_gen"],
            loss_disc=r["loss_disc"],
            loss_fm=r["loss_fm"],
            loss_kl=r["loss_kl"],
            trained_at=r["trained_at"],
        )
        for r in rows
    ]


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

