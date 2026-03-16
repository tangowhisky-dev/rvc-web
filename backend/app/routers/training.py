"""Training REST + WebSocket router.

Endpoints:
  POST   /api/training/start              — start a training job
  POST   /api/training/cancel             — cancel a running training job
  GET    /api/training/status/{profile_id} — poll job status
  WS     /ws/training/{profile_id}        — live log stream
"""

import asyncio
import json
import logging
import os
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


class CancelTrainingRequest(BaseModel):
    profile_id: str


class StartTrainingResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    phase: str
    progress_pct: int
    status: str
    error: Optional[str] = None


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
            "SELECT id, name, status, sample_path FROM profiles WHERE id = ?",
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
    rvc_root = os.path.abspath(rvc_root)  # ensure absolute so subprocess paths resolve correctly

    # Derive absolute sample directory from sample_path (which is the file path)
    sample_dir = os.path.dirname(os.path.abspath(row["sample_path"]))

    try:
        job = manager.start_job(
            request.profile_id,
            sample_dir,
            rvc_root,
            total_epoch=request.epochs,
            save_every=request.save_every,
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
        while True:
            try:
                msg = await asyncio.wait_for(job.queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                # Send a keepalive-style error and stop
                await websocket.send_json({
                    "type": "error",
                    "message": "queue timeout — no messages for 60s",
                    "phase": job.phase,
                })
                break

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

