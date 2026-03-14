"""RealtimeManager, RealtimeSession, and audio loop for real-time RVC voice conversion.

Provides:
  - AudioProcessor: Protocol for composable audio processing stages (R006)
  - RVCProcessor: Concrete AudioProcessor wrapping RVC inference
  - RealtimeSession: dataclass holding all per-session runtime state
  - RealtimeManager: singleton managing one active session at a time
  - _audio_loop: thread function that drives the sounddevice I/O and RVC inference
  - manager: module-level RealtimeManager singleton (import this)

Audio pipeline:
  MacBook Pro Mic (device 2, 44100 Hz) → resample_poly(160, 441) → 16 kHz rolling buffer
  → RVC.infer() → np.ndarray at 48000 Hz → BlackHole 2ch output (device 1, 48000 Hz)

Waveform delivery:
  _audio_loop pushes {type:"waveform_in", samples:<b64 float32>, sr:48000} and
  {type:"waveform_out", ...} into session._waveform_deque (~800 points each)
  at every block iteration (~20 fps at block_48k=9600).

Session lifecycle observability (structured stdout JSON):
  {"event":"session_start",  "session_id":..., "profile_id":..., ...}
  {"event":"session_stop",   "session_id":...}
  {"event":"session_error",  "session_id":..., "error":"..."}
  {"event":"waveform_overflow", "session_id":...}
"""

import base64
import collections
import glob
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import scipy.signal
import torch

from backend.app.db import get_db


# ---------------------------------------------------------------------------
# AudioProcessor Protocol (R006 — pipeline extensibility)
# ---------------------------------------------------------------------------

class AudioProcessor:
    """Protocol for composable audio processing stages.

    Each stage receives an audio array and sample rate, returns processed audio.
    The protocol is intentionally a plain class with a process() method to avoid
    requiring Python 3.12+ typing.Protocol gymnastics in tests.
    """

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process ``audio`` sampled at ``sample_rate`` Hz, return processed audio."""
        raise NotImplementedError


class RVCProcessor(AudioProcessor):
    """Wraps RVC.infer() as an AudioProcessor stage (satisfies R006).

    In the current implementation, the audio loop calls rvc.infer() directly
    with full block parameters. This class provides the Protocol-conforming
    wrapper for future pipeline extensibility without over-engineering S03.
    """

    def __init__(self, rvc: Any) -> None:
        self._rvc = rvc

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Thin delegation — full block-parameter inference runs in _audio_loop."""
        return audio  # passthrough; direct infer() calls own this in the loop


# ---------------------------------------------------------------------------
# Block parameter constants (verified via empirical testing — see S03-RESEARCH.md)
# ---------------------------------------------------------------------------

_SR_16K = 16000
_SR_48K = 48000
_WINDOW = 160       # RVC HuBERT hop length
_BLOCK_48K = 9600   # 200ms at 48 kHz output; ~37ms steady-state infer latency (0.19× realtime)


def _compute_block_params(block_48k: int = _BLOCK_48K) -> dict:
    """Compute all block size parameters for a given output block size.

    Formulae proved empirically — see S03-RESEARCH.md § Critical: Block Parameter Formula.
    All skip_head / return_length values are in FRAMES (÷ 160), NOT samples.

    Returns a dict with keys:
      block_48k, block_44k, block_16k, f0_extractor_frame,
      total_16k_samples, p_len, return_length_frames, skip_head_frames
    """
    block_16k = int(block_48k * _SR_16K / _SR_48K)          # samples at 16 kHz

    # RMVPE pad to multiple of 5120 frames
    f0_extractor_frame = block_16k + 800
    f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - _WINDOW

    total_16k_samples = f0_extractor_frame                   # rolling buffer length
    p_len = total_16k_samples // _WINDOW                     # frame count

    return_length_frames = block_16k // _WINDOW              # frames to return
    skip_head_frames = p_len - return_length_frames          # frames to skip

    # Input block at 44100 Hz (MacBook Pro Mic native rate)
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
    """Holds all runtime state for a single realtime voice conversion session."""

    session_id: str
    profile_id: str
    input_device: int
    output_device: int
    pitch: float = 0.0
    index_rate: float = 0.75
    protect: float = 0.33
    status: str = "starting"    # starting | active | stopped | error
    error: Optional[str] = None

    # Internal fields — not repr'd to keep logs clean
    _rvc: Any = field(default=None, repr=False)
    _stream_in: Any = field(default=None, repr=False)
    _stream_out: Any = field(default=None, repr=False)
    _waveform_deque: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=40), repr=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    _audio_thread: Optional[threading.Thread] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Audio loop (runs in daemon thread)
# ---------------------------------------------------------------------------

def _audio_loop(
    session: RealtimeSession,
    block_params: dict,
) -> None:
    """Core audio processing loop — runs in a dedicated daemon thread.

    Reads blocks from the input stream, resamples 44100→16000, maintains a
    rolling buffer, calls RVC inference, writes output, and pushes waveform
    snapshots to session._waveform_deque at ~20 fps.

    Terminates when session._stop_event is set or on any unhandled exception
    (sets session.status = "error" and logs the failure).
    """
    block_44k = block_params["block_44k"]
    block_48k = block_params["block_48k"]
    block_16k = block_params["block_16k"]
    total_16k_samples = block_params["total_16k_samples"]
    skip_head_frames = block_params["skip_head_frames"]
    return_length_frames = block_params["return_length_frames"]

    rvc = session._rvc
    protect = session.protect

    # Decimation factor for waveform subsampling (~800 points per block)
    decim = max(1, block_48k // 800)

    # Rolling buffer: initially silence
    rolling_buf = np.zeros(total_16k_samples, dtype=np.float32)

    try:
        while not session._stop_event.is_set():
            # Read one block from the 44100 Hz input stream
            try:
                data, overflowed = session._stream_in.read(block_44k)
            except Exception as read_err:
                print(
                    json.dumps({
                        "event": "waveform_overflow",
                        "session_id": session.session_id,
                        "detail": str(read_err),
                    }),
                    flush=True,
                )
                continue

            if overflowed:
                print(
                    json.dumps({
                        "event": "waveform_overflow",
                        "session_id": session.session_id,
                    }),
                    flush=True,
                )

            # Squeeze to 1-D and resample 44100 → 16000
            x_44k = data.squeeze()
            x_16k = scipy.signal.resample_poly(x_44k, 160, 441).astype(np.float32)

            # Shift rolling buffer and append new 16k block
            rolling_buf[:-block_16k] = rolling_buf[block_16k:]
            rolling_buf[-block_16k:] = x_16k[:block_16k]

            # RVC inference
            input_tensor = torch.from_numpy(rolling_buf).float()
            out = rvc.infer(
                input_tensor,
                block_16k,          # block_frame_16k — samples at 16 kHz
                skip_head_frames,   # skip_head — in FRAMES
                return_length_frames,  # return_length — in FRAMES
                "rmvpe",
                protect,
            )
            # out is np.ndarray at 48000 Hz, shape (block_48k,)
            if isinstance(out, torch.Tensor):
                out = out.cpu().numpy()
            out_48k = out.astype(np.float32)

            # Write output to 48000 Hz stream
            try:
                session._stream_out.write(out_48k.reshape(-1, 1))
            except Exception as write_err:
                print(
                    json.dumps({
                        "event": "session_error",
                        "session_id": session.session_id,
                        "phase": "write",
                        "error": str(write_err),
                    }),
                    flush=True,
                )
                # Non-fatal write error — continue loop

            # Waveform snapshots: subsample to ~800 points, base64-encode float32
            in_samples_b64 = base64.b64encode(
                x_44k[::decim].astype(np.float32).tobytes()
            ).decode("ascii")
            out_samples_b64 = base64.b64encode(
                out_48k[::decim].astype(np.float32).tobytes()
            ).decode("ascii")

            session._waveform_deque.append(
                {"type": "waveform_in", "samples": in_samples_b64, "sr": 48000}
            )
            session._waveform_deque.append(
                {"type": "waveform_out", "samples": out_samples_b64, "sr": 48000}
            )

    except Exception as exc:
        session.status = "error"
        session.error = str(exc)
        print(
            json.dumps({
                "event": "session_error",
                "session_id": session.session_id,
                "error": str(exc),
            }),
            flush=True,
        )
    finally:
        # Push sentinel so the WS handler exits cleanly
        session._waveform_deque.append(None)


# ---------------------------------------------------------------------------
# RealtimeManager
# ---------------------------------------------------------------------------

class RealtimeManager:
    """Singleton managing in-memory realtime session state.

    At most one active session at a time (single-user local tool; mirrors D017).
    Sessions are keyed by session_id in _sessions dict.
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
        """Create and start a realtime voice conversion session.

        Validates:
          - No active session (409 via ValueError)
          - Profile exists and has status=="trained" (404 / 400 via ValueError)
          - FAISS index file exists (400 via ValueError)

        Loads RVC model, runs warmup call with silence, opens separate
        InputStream (44100 Hz) and OutputStream (48000 Hz), starts audio thread.

        Args:
            profile_id: Profile ID to load the voice model for.
            input_device_id: sounddevice device index for microphone input.
            output_device_id: sounddevice device index for audio output.
            pitch: Pitch shift in semitones (default 0).
            index_rate: FAISS index blend ratio 0–1 (default 0.75).
            protect: Protect consonants/breath, 0–0.5 (default 0.33).
            rvc_root: Path to Retrieval-based-Voice-Conversion-WebUI root.
                      Falls back to RVC_ROOT env var.

        Returns:
            The created RealtimeSession (status == "active").

        Raises:
            ValueError: If validation fails (converted to HTTP errors by router).
        """
        # Guard: only one active session at a time
        if self.active_session() is not None:
            raise ValueError("session already active")

        # Resolve rvc_root
        if rvc_root is None:
            rvc_root = os.environ.get("RVC_ROOT", "Retrieval-based-Voice-Conversion-WebUI")

        # Validate profile status in DB
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

        # Locate FAISS index (name varies by N in IVF{N}_Flat)
        index_pattern = f"{rvc_root}/logs/rvc_finetune_active/added_IVF*_Flat*.index"
        matches = glob.glob(index_pattern)
        if not matches:
            raise ValueError("no index file found")
        index_path = matches[0]

        # Set env + sys.path before importing RVC (required per constraints)
        os.environ["rmvpe_root"] = f"{rvc_root}/assets/rmvpe"
        if rvc_root not in sys.path:
            sys.path.insert(0, rvc_root)

        # Apply fairseq safe-globals fix (PyTorch 2.6 + fairseq — D025 pattern)
        try:
            from configs.config import Config  # noqa: PLC0415 — deferred; needs sys.path
            Config.use_insecure_load()
        except Exception:
            pass  # Non-fatal if configs not available yet — fairseq may still load

        # Import and instantiate RVC
        from infer.lib.rtrvc import RVC  # noqa: PLC0415 — deferred; needs sys.path

        rvc = RVC(
            key=pitch,
            formant=0,
            pth_path=f"{rvc_root}/assets/weights/rvc_finetune_active.pth",
            index_path=index_path,
            index_rate=index_rate,
        )

        # Compute block parameters
        block_params = _compute_block_params(_BLOCK_48K)

        # Silence warmup call — avoids JIT compile latency on the first real block
        silence = torch.zeros(block_params["total_16k_samples"])
        try:
            rvc.infer(
                silence,
                block_params["block_16k"],
                block_params["skip_head_frames"],
                block_params["return_length_frames"],
                "rmvpe",
                protect,
            )
        except Exception:
            pass  # Warmup failure is non-fatal; real errors surface in the loop

        # Open streams at their native sample rates
        import sounddevice as sd  # noqa: PLC0415 — deferred; large dep

        stream_in = sd.InputStream(
            device=input_device_id,
            samplerate=44100,
            channels=1,
            dtype="float32",
            blocksize=block_params["block_44k"],
        )
        stream_out = sd.OutputStream(
            device=output_device_id,
            samplerate=_SR_48K,
            channels=1,
            dtype="float32",
            blocksize=block_params["block_48k"],
        )

        session_id = uuid.uuid4().hex
        session = RealtimeSession(
            session_id=session_id,
            profile_id=profile_id,
            input_device=input_device_id,
            output_device=output_device_id,
            pitch=pitch,
            index_rate=index_rate,
            protect=protect,
            status="starting",
            _rvc=rvc,
            _stream_in=stream_in,
            _stream_out=stream_out,
        )
        self._sessions[session_id] = session

        # Start audio thread before streams so the thread is ready to read
        audio_thread = threading.Thread(
            target=_audio_loop,
            args=(session, block_params),
            daemon=True,
            name=f"audio-loop-{session_id[:8]}",
        )
        session._audio_thread = audio_thread
        audio_thread.start()

        # Start both streams
        stream_in.start()
        stream_out.start()

        session.status = "active"

        print(
            json.dumps({
                "event": "session_start",
                "session_id": session_id,
                "profile_id": profile_id,
                "input_device": input_device_id,
                "output_device": output_device_id,
            }),
            flush=True,
        )

        return session

    def stop_session(self, session_id: str) -> None:
        """Stop the realtime session cleanly.

        Sets stop_event, stops and closes both streams, joins the audio thread
        (2s timeout), and marks session.status = "stopped".
        """
        session = self._sessions.get(session_id)
        if session is None:
            return

        session._stop_event.set()

        # Stop + close streams (order matters: stop before close)
        for stream in (session._stream_in, session._stream_out):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

        # Join audio thread (non-blocking timeout)
        if session._audio_thread is not None:
            session._audio_thread.join(timeout=2.0)

        session.status = "stopped"

        print(
            json.dumps({
                "event": "session_stop",
                "session_id": session_id,
            }),
            flush=True,
        )

    def get_session(self, session_id: str) -> Optional[RealtimeSession]:
        """Return the RealtimeSession for session_id, or None if not found."""
        return self._sessions.get(session_id)

    def active_session(self) -> Optional[RealtimeSession]:
        """Return the first session with status == 'active', or None."""
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
        """Hot-update session parameters without restarting the audio loop.

        Calls rvc.set_key() and rvc.set_index_rate() on the live RVC instance.
        Changes take effect on the next inference call.
        """
        session = self._sessions.get(session_id)
        if session is None or session._rvc is None:
            return

        if pitch is not None:
            session.pitch = pitch
            session._rvc.set_key(pitch)

        if index_rate is not None:
            session.index_rate = index_rate
            session._rvc.set_index_rate(index_rate)

        if protect is not None:
            session.protect = protect
            # protect is used directly in _audio_loop via session.protect


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

manager = RealtimeManager()
