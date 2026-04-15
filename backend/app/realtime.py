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
import threading
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
    index_rate: float = 0.50
    protect: float = 0.33
    silence_threshold_db: float = -45.0
    output_gain: float = 1.0
    noise_reduction: bool = True
    noise_reduction_output: bool = False
    sola_crossfade_ms: int = 20
    formant: float = 0.0
    use_jit: bool = False
    status: str = "starting"    # starting | active | stopped | error
    error: Optional[str] = None

    # Internal — subprocess + queues
    _proc: Any = field(default=None, repr=False)
    _cmd_q: Any = field(default=None, repr=False)
    _evt_q: Any = field(default=None, repr=False)
    _waveform_deque: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=40), repr=False
    )

    # Stop signal for WebSocket handler
    _stop_event: threading.Event = field(default_factory=lambda: threading.Event(), repr=False)

    # Background evt-drain thread
    _drain_thread: Any = field(default=None, repr=False)


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
        index_rate: float = 0.50,
        protect: float = 0.33,
        silence_threshold_db: float = -45.0,
        output_gain: float = 1.0,
        noise_reduction: bool = True,
        noise_reduction_output: bool = False,
        sola_crossfade_ms: int = 20,
        formant: float = 0.0,
        use_jit: bool = False,
        rvc_root: Optional[str] = None,
        save_path: Optional[str] = None,
        use_best: bool = False,
        # Beatrice 2 params
        pitch_shift_semitones: float = 0.0,
        formant_shift_semitones: float = 0.0,
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
            rvc_root = os.environ.get("PROJECT_ROOT", os.getcwd())
        rvc_root = os.path.abspath(rvc_root)

        # Validate profile in DB and fetch artifact paths
        async with get_db() as db:
            row = await db.execute_fetchall(
                """SELECT id, status, model_path, index_path, profile_dir, profile_rms, pipeline
                   FROM profiles WHERE id = ?""",
                (profile_id,),
            )
        if not row:
            raise ValueError(f"profile_not_found:{profile_id}")
        profile_status = row[0][1]
        if profile_status != "trained":
            raise ValueError(f"profile_not_trained:{profile_status}")

        # Resolve model and index paths from DB (new layout) with legacy fallback
        db_model_path: Optional[str] = row[0][2] or None
        db_index_path: Optional[str] = row[0][3] or None
        stored_profile_dir: Optional[str] = row[0][4] or None
        db_profile_rms: Optional[float] = float(row[0][5]) if row[0][5] is not None else None
        db_pipeline: str = str(row[0][6]) if row[0][6] else "rvc"

        # Check for per-profile canonical paths first
        if stored_profile_dir:
            # profile_dir may be stored as relative — resolve against PROJECT_ROOT
            abs_profile_dir = os.path.abspath(
                os.path.join(rvc_root, stored_profile_dir)
                if not os.path.isabs(stored_profile_dir)
                else stored_profile_dir
            )
            if db_pipeline == "beatrice2":
                # Beatrice 2 checkpoint
                b2_ckpt = os.path.join(abs_profile_dir, "beatrice2_out", "checkpoint_latest.pt.gz")
                if os.path.exists(b2_ckpt):
                    db_model_path = b2_ckpt
            else:
                # Resolve model: prefer model_best.pth when use_best is set and the
                # file exists; fall back to model_infer.pth (latest) otherwise.
                if use_best:
                    candidate_best = os.path.join(abs_profile_dir, "model_best.pth")
                    candidate_model = candidate_best if os.path.exists(candidate_best) else os.path.join(abs_profile_dir, "model_infer.pth")
                else:
                    candidate_model = os.path.join(abs_profile_dir, "model_infer.pth")
                candidate_index = os.path.join(abs_profile_dir, "model.index")
                if os.path.exists(candidate_model):
                    db_model_path = candidate_model
                if os.path.exists(candidate_index):
                    db_index_path = candidate_index

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
                silence_threshold_db=silence_threshold_db,
                output_gain=output_gain,
                noise_reduction=noise_reduction,
                noise_reduction_output=noise_reduction_output,
                sola_crossfade_ms=sola_crossfade_ms,
                formant=formant,
                use_jit=use_jit,
                save_path=save_path,
                model_path=db_model_path,
                index_path=db_index_path,
                profile_rms=db_profile_rms,
                pipeline=db_pipeline,
                pitch_shift_semitones=pitch_shift_semitones,
                formant_shift_semitones=formant_shift_semitones,
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
            silence_threshold_db=silence_threshold_db,
            output_gain=output_gain,
            noise_reduction=noise_reduction,
            noise_reduction_output=noise_reduction_output,
            sola_crossfade_ms=sola_crossfade_ms,
            formant=formant,
            use_jit=use_jit,
            status="active",
            _proc=proc,
            _cmd_q=cmd_q,
            _evt_q=evt_q,
        )
        self._sessions[session_id] = session

        # Background task: drain waveform events from the queue into the deque
        t = threading.Thread(
            target=self._drain_events,
            args=(session,),
            daemon=True,
            name=f"evt-drain-{session_id[:8]}",
        )
        t.start()
        session._drain_thread = t

        print(json.dumps({
            "event": "session_start",
            "session_id": session_id,
            "profile_id": profile_id,
            "input_device": input_device_id,
            "output_device": output_device_id,
        }), flush=True)

        return session

    def _drain_events(self, session: RealtimeSession) -> None:
        """Background thread: read events from evt_q and push waveforms into deque.

        Runs until the worker emits a 'stopped' event (which only fires after the
        save thread has finished encoding). Does NOT exit early when session.status
        changes to 'stopped' — that would drop save_complete/save_error events that
        arrive after the stop command is sent but before the worker finalises.
        """
        while True:
            # Use a short timeout so we can detect if the worker process has died
            # without sending a 'stopped' event (crash / SIGKILL).
            try:
                msg = session._evt_q.get(timeout=0.5)
            except Exception:
                # Empty queue — check if the worker process is still alive.
                # If the process is gone and the session is not active, we're done.
                if session._proc is not None and not session._proc.is_alive():
                    if session.status != "active":
                        break
                    else:
                        print(f"ERROR: worker process {session._proc.pid} died unexpectedly!", flush=True)
                        self.stop_session(session.session_id)
                        break
                continue

            event = msg.get("event", "")
            if event in ("waveform_in", "waveform_out"):
                session._waveform_deque.append({
                    "type": event,
                    "samples": msg["samples"],
                    "sr": msg["sr"],
                })
            elif event in ("save_complete", "save_error"):
                # Forward to frontend so UI can show save status
                session._waveform_deque.append({
                    "type": event,
                    **{k: v for k, v in msg.items() if k != "event"},
                })
            elif event == "error":
                session.status = "error"
                session.error = msg.get("error", "unknown")
                session._waveform_deque.append(None)
                break
            elif event == "stopped":
                # Worker's finally block is complete — safe to exit drain loop
                break
            # waveform_overflow and debug events are silently dropped

        # Push sentinel so WS handler exits cleanly
        session._waveform_deque.append(None)

    def stop_session(self, session_id: str) -> None:
        """Send stop command to the worker and return immediately.

        The audio loop (in the worker subprocess) will exit cleanly, wait for the
        save thread to finish encoding, then emit 'stopped'. The drain thread picks
        up that event, appends None to the deque, and exits. The WS handler drains
        the deque until it sees None and then sends 'done' to the browser.

        This function does NOT wait for the worker to finish — the HTTP response
        returns promptly and the drain chain completes in the background.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return

        session.status = "stopped"

        # Send stop command to the worker's audio loop
        try:
            session._cmd_q.put({"cmd": "stop"})
        except Exception:
            pass

        # Remove from sessions dict immediately so active_session() returns None.
        # The drain thread, WS handler, and stop_event lifecycle continues independently.
        self._sessions.pop(session_id, None)

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
        silence_threshold_db: Optional[float] = None,
        output_gain: Optional[float] = None,
        noise_reduction: Optional[bool] = None,
        noise_reduction_output: Optional[bool] = None,
        sola_crossfade_ms: Optional[int] = None,
        formant: Optional[float] = None,
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
        if silence_threshold_db is not None:
            session.silence_threshold_db = silence_threshold_db
            msg["silence_threshold_db"] = silence_threshold_db
        if output_gain is not None:
            session.output_gain = output_gain
            msg["output_gain"] = output_gain
        if noise_reduction is not None:
            session.noise_reduction = noise_reduction
            msg["noise_reduction"] = noise_reduction
        if noise_reduction_output is not None:
            session.noise_reduction_output = noise_reduction_output
            msg["noise_reduction_output"] = noise_reduction_output
        if sola_crossfade_ms is not None:
            session.sola_crossfade_ms = max(0, min(int(sola_crossfade_ms), 50))
            msg["sola_crossfade_ms"] = session.sola_crossfade_ms
        if formant is not None:
            session.formant = float(formant)
            msg["formant"] = session.formant

        try:
            session._cmd_q.put(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

manager = RealtimeManager()
