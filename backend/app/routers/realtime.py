"""Realtime voice conversion REST + WebSocket router.

Endpoints:
  POST   /api/realtime/start              — start a realtime voice session
  POST   /api/realtime/stop               — stop a running session
  POST   /api/realtime/params             — hot-update inference params
  GET    /api/realtime/status             — poll active session state
  WS     /ws/realtime/{session_id}        — stream waveform frames (~20 fps)

Two-router pattern (D023):
  ``router``    — prefixed ``/api/realtime`` for REST endpoints
  ``ws_router`` — no prefix; WebSocket at ``/ws/realtime/{session_id}``
"""

import asyncio
import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.app.db import get_db
from backend.app.realtime import manager

logger = logging.getLogger("rvc_web.realtime")

router = APIRouter(prefix="/api/realtime", tags=["realtime"])

# Separate router for WebSocket — no prefix (path is /ws/realtime/{session_id})
ws_router = APIRouter(tags=["realtime"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    profile_id: str
    input_device_id: int
    output_device_id: int
    pitch: float = 0.0
    index_rate: float = 0.50
    protect: float = 0.33
    silence_threshold_db: float = -55.0
    output_gain: float = 1.0
    noise_reduction: bool = True
    sola_crossfade_ms: int = 20
    save_path: Optional[str] = None
    use_best: bool = False  # if True, use model_best.pth instead of model_infer.pth


class StopSessionRequest(BaseModel):
    session_id: str


class UpdateParamsRequest(BaseModel):
    session_id: str
    pitch: Optional[float] = None
    index_rate: Optional[float] = None
    protect: Optional[float] = None
    silence_threshold_db: Optional[float] = None
    output_gain: Optional[float] = None
    noise_reduction: Optional[bool] = None
    sola_crossfade_ms: Optional[int] = None


class StartSessionResponse(BaseModel):
    session_id: str


class StatusResponse(BaseModel):
    active: bool
    session_id: Optional[str] = None
    profile_id: Optional[str] = None
    input_device: Optional[int] = None
    output_device: Optional[int] = None


# ---------------------------------------------------------------------------
# POST /api/realtime/start
# ---------------------------------------------------------------------------

@router.post("/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest) -> StartSessionResponse:
    """Start a realtime voice conversion session.

    Args:
        request: StartSessionRequest with device IDs and inference params.

    Returns:
        StartSessionResponse with the new session_id.

    Raises:
        HTTPException(404): Profile not found in DB.
        HTTPException(400): Profile not trained, no index file, or PROJECT_ROOT not set.
        HTTPException(409): A session is already active.
    """
    # Check for an already-active session first (fast 409 path; no DB needed)
    if manager.active_session() is not None:
        raise HTTPException(status_code=409, detail="session already active")

    # Look up profile
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, name, status FROM profiles WHERE id = ?",
            (request.profile_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Profile not found: {request.profile_id}",
        )

    if row["status"] != "trained":
        raise HTTPException(
            status_code=400,
            detail=f"Profile is not trained (status: {row['status']})",
        )

    # Pass project_root to manager; if it's empty the manager will raise ValueError
    project_root = os.environ.get("PROJECT_ROOT") or None
    if project_root:
        project_root = os.path.abspath(project_root)  # ensure absolute so inference paths resolve correctly

    try:
        result = manager.start_session(
            profile_id=request.profile_id,
            input_device_id=request.input_device_id,
            output_device_id=request.output_device_id,
            pitch=request.pitch,
            index_rate=request.index_rate,
            protect=request.protect,
            silence_threshold_db=request.silence_threshold_db,
            output_gain=request.output_gain,
            noise_reduction=request.noise_reduction,
            sola_crossfade_ms=request.sola_crossfade_ms,
            rvc_root=project_root,
            save_path=request.save_path,
            use_best=request.use_best,
        )
        # start_session is async on the real manager; MagicMock returns sync in tests
        session = await result if asyncio.iscoroutine(result) else result
    except HTTPException:
        # Re-raise HTTPExceptions from manager (e.g. 400 no index file found)
        raise
    except ValueError as exc:
        msg = str(exc).lower()
        if "already active" in msg:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Worker startup failure — log full message (includes traceback from worker)
        logger.error("worker startup error:\n%s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        json.dumps({
            "event": "realtime_start",
            "session_id": session.session_id,
            "profile_id": request.profile_id,
        })
    )

    return StartSessionResponse(session_id=session.session_id)


# ---------------------------------------------------------------------------
# POST /api/realtime/stop
# ---------------------------------------------------------------------------

@router.post("/stop")
async def stop_session(request: StopSessionRequest) -> dict:
    """Stop a running realtime voice conversion session.

    Args:
        request: StopSessionRequest with session_id (or "all" to stop all).

    Returns:
        {"ok": True}

    Raises:
        HTTPException(404): Session not found (only if specific session_id requested).
    """
    if request.session_id == "all":
        # Stop all active sessions
        for session_id in list(manager._sessions.keys()):
            try:
                manager.stop_session(session_id)
            except Exception:
                pass  # Ignore errors for already-stopping sessions
        return {"ok": True}

    session = manager.get_session(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}",
        )

    # stop_session is synchronous — sets stop_event and joins audio thread
    manager.stop_session(request.session_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# POST /api/realtime/params
# ---------------------------------------------------------------------------

@router.post("/params")
async def update_params(request: UpdateParamsRequest) -> dict:
    """Hot-update inference parameters for a running session.

    Args:
        request: UpdateParamsRequest with session_id and optional param overrides.

    Returns:
        {"ok": True}

    Raises:
        HTTPException(404): Session not found.
    """
    session = manager.get_session(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}",
        )

    manager.update_params(
        request.session_id,
        pitch=request.pitch,
        index_rate=request.index_rate,
        protect=request.protect,
        silence_threshold_db=request.silence_threshold_db,
        output_gain=request.output_gain,
        noise_reduction=request.noise_reduction,
        sola_crossfade_ms=request.sola_crossfade_ms,
    )
    return {"ok": True}


# ---------------------------------------------------------------------------
# GET /api/realtime/status
# ---------------------------------------------------------------------------

@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Return current active session state.

    Returns:
        StatusResponse with active=False and null fields if no active session.
        StatusResponse with active=True and session details if a session is running.
    """
    session = manager.active_session()
    if session is None:
        return StatusResponse(
            active=False,
            session_id=None,
            profile_id=None,
            input_device=None,
            output_device=None,
        )
    return StatusResponse(
        active=True,
        session_id=session.session_id,
        profile_id=session.profile_id,
        input_device=session.input_device,
        output_device=session.output_device,
    )


# ---------------------------------------------------------------------------
# WS /ws/realtime/{session_id}  (registered on ws_router, no prefix)
# ---------------------------------------------------------------------------

@ws_router.websocket("/ws/realtime/{session_id}")
async def realtime_websocket(websocket: WebSocket, session_id: str) -> None:
    """Stream waveform frames over WebSocket at ~20 fps.

    Polls up to 30s for the session to become active (handles the race where
    a client connects immediately after POST /start returns). Once found,
    drains ``session._waveform_deque`` with 50ms sleeps (~20 fps). Sends
    ``{type: "done"}`` and closes when the sentinel ``None`` is dequeued or
    when the stop_event is set with an empty deque.

    Note: ``_waveform_deque`` is a ``collections.deque`` — single-producer
    (audio thread) / single-consumer (this coroutine). ``popleft()`` is
    thread-safe for this access pattern.

    Args:
        websocket: The WebSocket connection.
        session_id: The session whose waveform stream to relay.
    """
    await websocket.accept()

    # Poll up to 30s for the session to appear (300 × 100ms)
    session = None
    for _ in range(300):
        session = manager.get_session(session_id)
        if session is not None:
            break
        await asyncio.sleep(0.1)

    if session is None:
        await websocket.send_json({"type": "error", "message": "session not found"})
        await websocket.close()
        return

    try:
        while True:
            # Drain all available frames from the deque
            drained_any = False
            while session._waveform_deque:
                try:
                    msg = session._waveform_deque.popleft()
                except IndexError:
                    break  # deque became empty between check and pop — harmless

                if msg is None:
                    # Sentinel: audio loop is done
                    await websocket.send_json({"type": "done"})
                    return

                await websocket.send_json(msg)
                drained_any = True

            # Check stop_event after draining (deque may have been cleared already)
            if session._stop_event.is_set() and not session._waveform_deque:
                await websocket.send_json({"type": "done"})
                return

            # ~20 fps poll rate
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        logger.info(
            json.dumps({
                "event": "ws_realtime_disconnect",
                "session_id": session_id,
            })
        )


# ---------------------------------------------------------------------------
# GET /api/realtime/default-save-dir
# ---------------------------------------------------------------------------

@router.get("/default-save-dir")
async def default_save_dir() -> dict:
    """Return the user's Downloads directory as the default save location."""
    import pathlib  # noqa: PLC0415
    downloads = pathlib.Path.home() / "Downloads"
    return {"path": str(downloads)}
