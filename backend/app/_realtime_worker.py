"""Isolated realtime worker process.

Spawned by RealtimeManager as a separate process so that fairseq/RVC/sounddevice
native code never shares the uvicorn address space (prevents segfault on macOS/MPS).

Protocol (multiprocessing.Queue messages):

  Parent → child (cmd_q):
    {"cmd": "stop"}
    {"cmd": "update_params", "pitch": float, "index_rate": float, "protect": float}

  Child → parent (evt_q):
    {"event": "warming_up",      "device": "mps" | "cpu"}
    {"event": "ready",           "session_id": str, "device": str}
    {"event": "error",           "error": str}
    {"event": "stopped"}
    {"event": "waveform_in",     "samples": <b64 float32>, "sr": 48000}
    {"event": "waveform_out",    "samples": <b64 float32>, "sr": 48000}
    {"event": "waveform_overflow", "detail": str}
    {"event": "save_complete",   "path": str, "duration_s": float}
    {"event": "save_error",      "error": str}

Device selection:
  - Prefers MPS (Metal Performance Shaders) on Apple Silicon for inference
  - Falls back to CPU if MPS not available
  - Warmup call ensures all MPS kernel compilation happens before "ready" signal
  - Typical warmup: ~2s on MPS (first call), ~50ms on subsequent calls

Usage:
    python _realtime_worker.py  (not meant to be invoked directly)
"""

import base64
import json
import logging
import multiprocessing
import os
import queue
import sys
import threading
import uuid

# Silence numba DEBUG flood before any library imports it.
# Numba JIT compiles on first call; the compilation trace goes to the 'numba'
# logger. Without this the worker log is dominated by SSA/byteflow debug noise.
logging.basicConfig(level=logging.WARNING)
for _noisy in ("numba", "numba.core", "numba.core.ssa", "numba.core.byteflow",
               "numba.core.interpreter", "numba.typed"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import numpy as np
import scipy.signal
import torch


# ---------------------------------------------------------------------------
# Block parameter constants (must match realtime.py)
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
# MP3 save thread — drains audio_q and writes to mp3 on stop
# ---------------------------------------------------------------------------

_SAVE_SENTINEL = None  # signals the save thread to flush and exit


def _save_thread(audio_q: queue.Queue, save_path: str, sr: int, evt_q: multiprocessing.Queue) -> None:
    """Background thread: accumulate PCM blocks from audio_q, encode to MP3 on sentinel."""
    chunks = []
    total_samples = 0

    # Expand ~ in save_path (may happen if the browser fallback sends ~/... paths)
    save_path = os.path.expanduser(save_path)

    try:
        while True:
            block = audio_q.get()
            if block is _SAVE_SENTINEL:
                break
            chunks.append(block)
            total_samples += len(block)

        if not chunks:
            evt_q.put({"event": "save_error", "error": "no audio captured"})
            return

        # Concatenate, normalise to int16, encode via pydub
        pcm = np.concatenate(chunks).astype(np.float32)
        # Normalise to [-1, 1] if clipping
        peak = np.abs(pcm).max()
        if peak > 1.0:
            pcm /= peak

        from pydub import AudioSegment  # noqa: PLC0415
        seg = AudioSegment(
            pcm.tobytes(),
            frame_rate=sr,
            sample_width=4,   # float32 = 4 bytes
            channels=1,
        )
        # Ensure parent directory exists (os.path.dirname returns '' for bare filenames,
        # so guard against that before calling makedirs).
        parent_dir = os.path.dirname(os.path.abspath(save_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        seg.export(save_path, format="mp3", bitrate="128k")
        duration_s = total_samples / sr
        evt_q.put({"event": "save_complete", "path": save_path, "duration_s": round(duration_s, 2)})

    except Exception as exc:
        evt_q.put({"event": "save_error", "error": str(exc)})


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

def run_worker(
    cmd_q: multiprocessing.Queue,
    evt_q: multiprocessing.Queue,
    rvc_root: str,
    profile_id: str,
    input_device_id: int,
    output_device_id: int,
    pitch: float,
    index_rate: float,
    protect: float,
    save_path: str | None = None,
) -> None:
    """Run in isolated child process. Owns all native code (fairseq, sounddevice, MPS)."""

    import glob as _glob

    # Must be set before any native library is loaded — prevents OpenMP conflict
    # between fairseq and faiss which each link a different libomp on macOS.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["rmvpe_root"] = f"{rvc_root}/assets/rmvpe"
    os.environ["hubert_path"] = f"{rvc_root}/assets/hubert/hubert_base.pt"

    if rvc_root not in sys.path:
        sys.path.insert(0, rvc_root)

    session_id = uuid.uuid4().hex

    # If save_path requested, start the save thread immediately so it's ready
    # before the audio loop. Uses an in-process queue (thread-safe, not mp.Queue).
    audio_save_q: queue.Queue | None = None
    save_t: threading.Thread | None = None
    if save_path:
        audio_save_q = queue.Queue(maxsize=0)  # unbounded — save thread drains async
        save_t = threading.Thread(
            target=_save_thread,
            args=(audio_save_q, save_path, _SR_48K, evt_q),
            daemon=True,
            name="mp3-save",
        )
        save_t.start()

    try:
        # Import torch first, then faiss with nthreads=1, before fairseq.
        # This order prevents the faiss+fairseq OpenMP collision on macOS.
        import torch as _torch  # noqa: PLC0415
        import faiss as _faiss  # noqa: PLC0415
        _faiss.omp_set_num_threads(1)

        # Resolve inference device — prefer MPS (Apple Silicon GPU), fall back to CPU
        if _torch.backends.mps.is_available():
            infer_device = "mps"
        else:
            infer_device = "cpu"

        # fairseq safe-globals fix (PyTorch 2.6 — D025)
        try:
            from configs.config import Config  # noqa: PLC0415
            Config.use_insecure_load()
        except Exception:
            pass

        from infer.lib.rtrvc import RVC  # noqa: PLC0415

        index_pattern = f"{rvc_root}/logs/rvc_finetune_active/added_IVF*_Flat*.index"
        matches = _glob.glob(index_pattern)
        if not matches:
            raise RuntimeError("no index file found — run training first")
        index_path = matches[0]

        # is_half=False: MPS doesn't support float16 in all ops reliably
        rvc = RVC(
            key=pitch,
            formant=0,
            pth_path=f"{rvc_root}/assets/weights/rvc_finetune_active.pth",
            index_path=index_path,
            index_rate=index_rate,
            device=infer_device,
            is_half=False,
        )

        block_params = _compute_block_params(_BLOCK_48K)

        # Warmup: run one full infer pass so MPS compiles its kernels.
        # This ensures the "ready" signal only fires after real-time latency
        # is stable (first call is ~2s on MPS, subsequent calls ~50ms).
        evt_q.put({"event": "warming_up", "device": infer_device})
        silence = _torch.zeros(block_params["total_16k_samples"])
        rvc.infer(
            silence,
            block_params["block_16k"],
            block_params["skip_head_frames"],
            block_params["return_length_frames"],
            "rmvpe",
            protect,
        )

        import sounddevice as sd  # noqa: PLC0415

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

        stream_in.start()
        stream_out.start()

        evt_q.put({"event": "ready", "session_id": session_id, "device": infer_device})

        # Audio loop
        block_44k = block_params["block_44k"]
        block_48k = block_params["block_48k"]
        block_16k = block_params["block_16k"]
        total_16k_samples = block_params["total_16k_samples"]
        skip_head_frames = block_params["skip_head_frames"]
        return_length_frames = block_params["return_length_frames"]
        decim = max(1, block_48k // 800)

        rolling_buf = np.zeros(total_16k_samples, dtype=np.float32)

        while True:
            # Check for stop command (non-blocking)
            try:
                msg = cmd_q.get_nowait()
                if isinstance(msg, dict) and msg.get("cmd") == "stop":
                    break
                # Handle hot-update params
                if isinstance(msg, dict) and msg.get("cmd") == "update_params":
                    if "pitch" in msg:
                        rvc.set_key(msg["pitch"])
                    if "index_rate" in msg:
                        rvc.set_index_rate(msg["index_rate"])
                    if "protect" in msg:
                        protect = msg["protect"]
            except Exception:
                pass  # Empty queue — continue

            try:
                data, overflowed = stream_in.read(block_44k)
            except Exception as e:
                evt_q.put({"event": "waveform_overflow", "detail": str(e)})
                continue

            if overflowed:
                evt_q.put({"event": "waveform_overflow", "detail": "overflow"})

            x_44k = data.squeeze()
            x_16k = scipy.signal.resample_poly(x_44k, 160, 441).astype(np.float32)

            rolling_buf[:-block_16k] = rolling_buf[block_16k:]
            rolling_buf[-block_16k:] = x_16k[:block_16k]

            input_tensor = _torch.from_numpy(rolling_buf).float()
            out = rvc.infer(
                input_tensor,
                block_16k,
                skip_head_frames,
                return_length_frames,
                "rmvpe",
                protect,
            )
            if isinstance(out, torch.Tensor):
                out = out.cpu().numpy()
            out_48k = out.astype(np.float32)

            try:
                stream_out.write(out_48k.reshape(-1, 1))
            except Exception:
                pass

            # Enqueue block for MP3 save (non-blocking — save thread drains independently)
            if audio_save_q is not None:
                try:
                    audio_save_q.put_nowait(out_48k.copy())
                except queue.Full:
                    pass  # drop block if save queue is somehow full

            # Waveform snapshots (~800 pts)
            evt_q.put({
                "event": "waveform_in",
                "samples": base64.b64encode(x_44k[::decim].astype(np.float32).tobytes()).decode(),
                "sr": 48000,
            })
            evt_q.put({
                "event": "waveform_out",
                "samples": base64.b64encode(out_48k[::decim].astype(np.float32).tobytes()).decode(),
                "sr": 48000,
            })

    except Exception as exc:
        evt_q.put({"event": "error", "error": str(exc)})

    finally:
        # Signal save thread to flush and encode — happens after audio loop exits
        if audio_save_q is not None:
            evt_q.put({"event": "debug", "message": f"Sending sentinel to save queue, items waiting: {audio_save_q.qsize()}"})
            audio_save_q.put(_SAVE_SENTINEL)
            if save_t is not None:
                evt_q.put({"event": "debug", "message": f"Waiting for save thread to finish..."})
                save_t.join(timeout=30)  # up to 30s for pydub/ffmpeg to encode
                evt_q.put({"event": "debug", "message": f"Save thread finished"})

        # Clean up streams — guard against NameError if streams were never assigned
        # (exception during model loading, before sd.InputStream was created).
        try:
            stream_in.stop(); stream_in.close()  # noqa: F821
        except Exception:
            pass
        try:
            stream_out.stop(); stream_out.close()  # noqa: F821
        except Exception:
            pass

    evt_q.put({"event": "stopped"})

