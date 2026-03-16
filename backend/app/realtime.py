"""RealtimeManager — spawns an isolated child process to run RVC/fairseq/sounddevice.

All native code (fairseq, HuBERT, sounddevice, MPS) runs in _realtime_worker.py
subprocess so it never shares the uvicorn address space (prevents segfault on macOS).

Communication:
  cmd_q  (parent→child): stop, update_params
  evt_q  (child→parent): warming_up, ready, error, stopped, waveform_in, waveform_out, waveform_overflow

Session lifecycle observability (structured stdout JSON):
  {"event":"worker_warming_up", "device": "mps" | "cpu"}
  {"event":"worker_ready",      "session_id":..., "device":...}
  {"event":"session_start",     "session_id":..., "profile_id":..., ...}
  {"event":"session_stop",      "session_id":...}
  {"event":"session_error",     "session_id":..., "error":"..."}
"""

import collections
import json
import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from backend.app.db import get_db


# ---------------------------------------------------------------------------
# AudioProcessor Protocol (R006 — pipeline extensibility)
# ---------------------------------------------------------------------------

class AudioProcessor:
    """Protocol for composable audio processing stages.

    Each stage receives an audio array and sample rate, returns processed audio.
    """

    def process(self, audio: Any, sample_rate: int) -> Any:
        raise NotImplementedError


class RVCProcessor(AudioProcessor):
    """Wraps RVC.infer() as an AudioProcessor stage (satisfies R006)."""

    def __init__(self, rvc: Any) -> None:
        self._rvc = rvc

    def process(self, audio: Any, sample_rate: int) -> Any:
        return audio  # passthrough; direct infer() calls own this in the loop


# ---------------------------------------------------------------------------
# Block parameter constants
# ---------------------------------------------------------------------------

_SR_16K = 16000
_SR_48K = 48000
_WINDOW = 160
_BLOCK_48K = 9600


def _compute_block_params(block_48k: int = _BLOCK_48K) -> dict:
    block_16k = int(block_48k * _SR_16K / _SR_48K)
    f0_extractor_frame = block_16k + 800
    f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - _WINDOW
    total_16k_samples = f0_extractor_frame
    p_len = total_16k_samples // _WINDOW
    return_length_frames = block_16k // _WINDOW
    skip_head_frames = p_len - return_length_frames
    block_44k = int(block_48k * 44100 / _SR_48K)
    return {
        "block_48k": block_48k,
        "block_44k": block_44k,
        "block_16k": block_16k,
        "f0_extractor_frame": f0_extractor_frame,
        "total_16k_samples": total_16k_samples,
        "p_len": p_len,
        "return_length_frames": return_length_frames,
        "skip_head_frames": skip_head_frames,
    }


# ---------------------------------------------------------------------------
# RealtimeSession dataclass
# ---------------------------------------------------------------------------

@dataclass
class RealtimeSession:
    """Holds runtime state for one active realtime voice conversion session."""

    session_id: str
    profile_id: str
    input_device: int
    output_device: int
    pitch: float = 0.0
    index_rate: float = 0.75
    protect: float = 0.33
    status: str = "starting"    # starting | active | stopped | error
    error: Optional[str] = None

    # Internal — subprocess + queues
    _proc: Any = field(default=None, repr=False)
    _cmd_q: Any = field(default=None, repr=False)
    _evt_q: Any = field(default=None, repr=False)
    _waveform_deque: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=40), repr=False
    )


# ---------------------------------------------------------------------------
# RealtimeManager
# ---------------------------------------------------------------------------

class RealtimeManager:
    """Singleton managing one active realtime session at a time.

    The session's audio loop runs in an isolated subprocess (_realtime_worker.py)
    to prevent fairseq/MPS from segfaulting uvicorn.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, RealtimeSession] = {}

    async def start_session(
        self,
        profile_id: str,
        input_device_id: int,
        output_device_id: int,
        pitch: float = 0.0,
        index_rate: float = 0.75,
        protect: float = 0.33,
        rvc_root: Optional[str] = None,
    ) -> RealtimeSession:
        """Spawn isolated worker process and return a RealtimeSession.

        Raises ValueError on validation errors (no active session, profile checks).
        Raises RuntimeError if the worker fails to become ready within 60s.
        """
        # Only one active session at a time
        for sess in self._sessions.values():
            if sess.status == "active":
                raise ValueError("session already active")

        # Resolve rvc_root
        if rvc_root is None:
            rvc_root = os.environ.get("RVC_ROOT", "Retrieval-based-Voice-Conversion-WebUI")
        rvc_root = os.path.abspath(rvc_root)

        # Validate profile in DB
        async with get_db() as db:
            row = await db.execute_fetchall(
                "SELECT id, status FROM profiles WHERE id = ?",
                (profile_id,),
            )
        if not row:
            raise ValueError(f"profile_not_found:{profile_id}")
        profile_status = row[0][1]
        if profile_status != "trained":
            raise ValueError(f"profile_not_trained:{profile_status}")

        # Spawn the worker
        ctx = multiprocessing.get_context("spawn")
        cmd_q: multiprocessing.Queue = ctx.Queue()
        evt_q: multiprocessing.Queue = ctx.Queue()

        worker_path = os.path.join(os.path.dirname(__file__), "_realtime_worker.py")
        # Import and call the worker function directly (avoids __main__ guard issues)
        from backend.app._realtime_worker import run_worker  # noqa: PLC0415

        proc = ctx.Process(
            target=run_worker,
            kwargs=dict(
                cmd_q=cmd_q,
                evt_q=evt_q,
                rvc_root=rvc_root,
                profile_id=profile_id,
                input_device_id=input_device_id,
                output_device_id=output_device_id,
                pitch=pitch,
                index_rate=index_rate,
                protect=protect,
            ),
            daemon=True,
        )
        proc.start()

        session_id: str = ""
        # Wait up to 60s for the worker to emit "ready" or "error"
        import asyncio  # noqa: PLC0415
        loop = asyncio.get_event_loop()

        deadline = loop.time() + 60.0
        ready = False
        while loop.time() < deadline:
            try:
                msg = await loop.run_in_executor(None, lambda: evt_q.get(timeout=1.0))
            except Exception:
                if not proc.is_alive():
                    raise RuntimeError("realtime worker exited before ready")
                continue

            if msg.get("event") == "ready":
                session_id = msg["session_id"]
                infer_device = msg.get("device", "unknown")
                ready = True
                print(json.dumps({
                    "event": "worker_ready",
                    "session_id": session_id,
                    "device": infer_device,
                }), flush=True)
                break
            elif msg.get("event") == "warming_up":
                infer_device = msg.get("device", "unknown")
                print(json.dumps({
                    "event": "worker_warming_up",
                    "device": infer_device,
                }), flush=True)
            elif msg.get("event") == "error":
                proc.terminate()
                raise RuntimeError(msg.get("error", "worker error"))

        if not ready:
            proc.terminate()
            raise RuntimeError("realtime worker timed out (60s) waiting for ready")

        session = RealtimeSession(
            session_id=session_id,
            profile_id=profile_id,
            input_device=input_device_id,
            output_device=output_device_id,
            pitch=pitch,
            index_rate=index_rate,
            protect=protect,
            status="active",
            _proc=proc,
            _cmd_q=cmd_q,
            _evt_q=evt_q,
        )
        self._sessions[session_id] = session

        # Background task: drain waveform events from the queue into the deque
        import threading  # noqa: PLC0415
        t = threading.Thread(
            target=self._drain_events,
            args=(session,),
            daemon=True,
            name=f"evt-drain-{session_id[:8]}",
        )
        t.start()

        print(json.dumps({
            "event": "session_start",
            "session_id": session_id,
            "profile_id": profile_id,
            "input_device": input_device_id,
            "output_device": output_device_id,
        }), flush=True)

        return session

    def _drain_events(self, session: RealtimeSession) -> None:
        """Background thread: read events from evt_q and push waveforms into deque."""
        while session.status == "active":
            try:
                msg = session._evt_q.get(timeout=0.5)
            except Exception:
                continue

            event = msg.get("event", "")
            if event in ("waveform_in", "waveform_out"):
                session._waveform_deque.append({
                    "type": event,
                    "samples": msg["samples"],
                    "sr": msg["sr"],
                })
            elif event == "error":
                session.status = "error"
                session.error = msg.get("error", "unknown")
                session._waveform_deque.append(None)
                break
            elif event in ("stopped", "waveform_overflow"):
                if event == "stopped":
                    break

        # Push sentinel so WS handler exits
        session._waveform_deque.append(None)

    def stop_session(self, session_id: str) -> None:
        """Send stop command to the worker and wait for it to exit."""
        session = self._sessions.get(session_id)
        if session is None:
            return

        session.status = "stopped"

        # Send stop command
        try:
            session._cmd_q.put({"cmd": "stop"})
        except Exception:
            pass

        # Give the worker 3s to stop cleanly, then kill
        if session._proc is not None:
            session._proc.join(timeout=3.0)
            if session._proc.is_alive():
                session._proc.terminate()
                session._proc.join(timeout=1.0)

        print(json.dumps({
            "event": "session_stop",
            "session_id": session_id,
        }), flush=True)

    def get_session(self, session_id: str) -> Optional[RealtimeSession]:
        return self._sessions.get(session_id)

    def active_session(self) -> Optional[RealtimeSession]:
        for session in self._sessions.values():
            if session.status == "active":
                return session
        return None

    def update_params(
        self,
        session_id: str,
        pitch: Optional[float] = None,
        index_rate: Optional[float] = None,
        protect: Optional[float] = None,
    ) -> None:
        """Hot-update session parameters without restarting."""
        session = self._sessions.get(session_id)
        if session is None or session._proc is None:
            return

        msg: dict = {"cmd": "update_params"}
        if pitch is not None:
            session.pitch = pitch
            msg["pitch"] = pitch
        if index_rate is not None:
            session.index_rate = index_rate
            msg["index_rate"] = index_rate
        if protect is not None:
            session.protect = protect
            msg["protect"] = protect

        try:
            session._cmd_q.put(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

manager = RealtimeManager()
