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
import scipy.signal  # kept as CPU fallback; primary resample is torchaudio on-device
import torch


# ---------------------------------------------------------------------------
# Block parameter constants (must match realtime.py)
# ---------------------------------------------------------------------------

_SR_16K = 16000
_SR_48K = 48000
_WINDOW = 160
_BLOCK_48K = 9600          # 200ms output block at 48kHz
_EXTRA_TIME = 2.0          # seconds of lookahead context (matches Vonovox/RVC-WebUI-MacOS)
_ZC = _SR_48K // 100       # zero-crossing unit = 10ms at 48kHz = 480 samples
_SOLA_BUF_48K = 4 * _ZC   # 40ms SOLA crossfade buffer (RVC-WebUI-MacOS default)
_SOLA_SEARCH_48K = _ZC     # 10ms search window for best alignment offset


def _compute_block_params(block_48k: int = _BLOCK_48K, extra_time: float = _EXTRA_TIME) -> dict:
    """Compute all buffer sizes needed by the audio loop.

    The RVC model (HuBERT + RMVPE + net_g) needs substantial historical context
    to produce good pitch and feature estimates. In practice 2s of lookahead
    produces clearly intelligible output (Vonovox: "recommended 2.0 for best
    quality/latency ratio"; RVC-WebUI-MacOS default extra_time=2.5s).

    We satisfy the rtrvc.py constraint (total_16k must equal f0_extractor_frame)
    by making total_16k = extra_16k + f0_extractor_frame and deriving skip_head
    so the model discards the extra context internally — only the last
    return_length_frames are used as output.
    """
    block_16k = int(block_48k * _SR_16K / _SR_48K)

    # Minimum valid buffer for rtrvc.py RMVPE frame constraint
    f0_min = block_16k + 800
    f0_extractor_frame = 5120 * ((f0_min - 1) // 5120 + 1) - _WINDOW

    # Extra context rounded to WINDOW boundary so p_len stays integer
    extra_16k = (int(extra_time * _SR_16K) // _WINDOW) * _WINDOW

    total_16k_samples = extra_16k + f0_extractor_frame
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
        "extra_16k": extra_16k,
        "p_len": p_len,
        "return_length_frames": return_length_frames,
        "skip_head_frames": skip_head_frames,
        "sola_buf_48k": _SOLA_BUF_48K,
        "sola_search_48k": _SOLA_SEARCH_48K,
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
        # pydub expects integer PCM; convert float32 [-1, 1] → int16
        pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
        seg = AudioSegment(
            pcm_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,   # int16 = 2 bytes
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
    silence_threshold_db: float = -45.0,
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
    audio_in_save_q: queue.Queue | None = None
    save_in_t: threading.Thread | None = None
    if save_path:
        audio_save_q = queue.Queue(maxsize=0)  # unbounded — save thread drains async
        save_t = threading.Thread(
            target=_save_thread,
            args=(audio_save_q, save_path, _SR_48K, evt_q),
            daemon=True,
            name="mp3-save",
        )
        save_t.start()

        # Input save: same dir as output, named rvc_input.mp3
        _save_dir = os.path.dirname(os.path.abspath(os.path.expanduser(save_path)))
        input_save_path = os.path.join(_save_dir, "rvc_input.mp3")
        audio_in_save_q = queue.Queue(maxsize=0)
        save_in_t = threading.Thread(
            target=_save_thread,
            args=(audio_in_save_q, input_save_path, _SR_48K, evt_q),
            daemon=True,
            name="mp3-save-input",
        )
        save_in_t.start()

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

        # ---------------------------------------------------------------
        # Patch rvc.infer to fix the pitchf/p_len mismatch that occurs when
        # total_16k_samples > f0_extractor_frame (our 2s-context design).
        #
        # rtrvc.py computes pitch via _get_f0(input_wav[-f0_extractor_frame:])
        # which returns pitch.shape[0] = f0_extractor_frame // window = 31 frames.
        # But p_len = total_16k_samples // window = 231 frames.
        # When protect < 0.5 the code does:
        #   pitchff = pitchf.clone()          # [1, 31]
        #   pitchff = pitchff.unsqueeze(-1)   # [1, 31, 1]
        #   feats = feats * pitchff + ...     # [1, 231, 256] * [1, 31, 1] → CRASH
        #
        # Fix: wrap infer to pad pitch/pitchf at the head with zeros (unvoiced)
        # so they match p_len. The head frames are discarded by skip_head anyway.
        # ---------------------------------------------------------------
        _orig_get_f0 = rvc._get_f0
        _p_len_expected = block_params["p_len"]  # 231

        def _patched_get_f0(x, f0_up_key, filter_radius=None, method="fcpe"):
            c, f = _orig_get_f0(x, f0_up_key, filter_radius=filter_radius, method=method)
            # c, f shape: [frames] where frames = f0_extractor_frame // window (31)
            # Pad head to p_len so pitchff broadcast with feats [1, p_len, H] succeeds
            if c.shape[-1] < _p_len_expected:
                pad = _p_len_expected - c.shape[-1]
                c = _torch.cat([_torch.zeros(pad, dtype=c.dtype, device=c.device), c])
                f = _torch.cat([_torch.zeros(pad, dtype=f.dtype, device=f.device), f])
            return c, f

        rvc._get_f0 = _patched_get_f0

        # Warmup: run one full infer pass so MPS compiles its kernels.
        # This ensures the "ready" signal only fires after real-time latency
        # is stable (first call is ~2s on MPS, subsequent calls ~50ms).
        evt_q.put({"event": "warming_up", "device": infer_device})
        silence = _torch.zeros(block_params["total_16k_samples"], device=infer_device)
        rvc.infer(
            silence,
            block_params["block_16k"],
            block_params["skip_head_frames"],
            block_params["return_length_frames"],
            "rmvpe",
            protect,
        )

        import sounddevice as sd  # noqa: PLC0415
        from torchaudio.transforms import Resample as _Resample  # noqa: PLC0415

        # Build on-device resampler: 44100 → 16000 using torchaudio (MPS/CPU).
        # Much faster than scipy.signal.resample_poly for small blocks since
        # it avoids the CPU↔device round-trip.
        _resample_44_16 = _Resample(orig_freq=44100, new_freq=16000,
                                    dtype=torch.float32).to(infer_device)
        _resample_44_48 = _Resample(orig_freq=44100, new_freq=_SR_48K,
                                    dtype=torch.float32).to(infer_device)

        stream_in = sd.InputStream(
            device=input_device_id,
            samplerate=44100,
            channels=1,
            dtype="float32",
            blocksize=block_params["block_44k"],
            latency="low",
        )
        stream_out = sd.OutputStream(
            device=output_device_id,
            samplerate=_SR_48K,
            channels=1,
            dtype="float32",
            blocksize=block_params["block_48k"],
            latency="low",   # was 'high' (sounddevice global default) — key stutter fix
        )

        stream_in.start()
        stream_out.start()

        evt_q.put({"event": "ready", "session_id": session_id, "device": infer_device})

        # Audio loop locals
        block_44k = block_params["block_44k"]
        block_48k = block_params["block_48k"]
        block_16k = block_params["block_16k"]
        total_16k_samples = block_params["total_16k_samples"]
        skip_head_frames = block_params["skip_head_frames"]
        return_length_frames = block_params["return_length_frames"]
        sola_buf_48k = block_params["sola_buf_48k"]
        sola_search_48k = block_params["sola_search_48k"]
        decim = max(1, block_48k // 800)

        # Silence gate
        _silence_rms = 10 ** (silence_threshold_db / 20.0)

        # Hold/release gate (Vonovox 400ms release window concept).
        # At 200ms blocks: HOLD=3→600ms, RELEASE=2→400ms fade.
        _HOLD_BLOCKS = 3
        _RELEASE_BLOCKS = 2
        _hold_count = 0
        _release_count = 0
        _gate_gain = 0.0

        # Rolling 16k context buffer — holds extra_16k + f0_extractor_frame samples.
        # This is the key fix: 2s of lookahead gives HuBERT and RMVPE enough
        # historical context to produce intelligible output (was only 110ms before).
        rolling_buf = _torch.zeros(total_16k_samples, dtype=_torch.float32,
                                   device=infer_device)

        # SOLA crossfade state (RVC-WebUI-MacOS algorithm).
        # Eliminates clicks at block boundaries by finding the best alignment
        # offset between the previous block's tail and the new block's head.
        sola_buf = _torch.zeros(sola_buf_48k, dtype=_torch.float32, device=infer_device)
        fade_in  = _torch.sin(0.5 * np.pi * _torch.linspace(0., 1., sola_buf_48k,
                                                              device=infer_device)) ** 2
        fade_out = 1.0 - fade_in
        import torch.nn.functional as _F  # noqa: PLC0415

        while True:
            # Non-blocking command check
            try:
                msg = cmd_q.get_nowait()
                if isinstance(msg, dict) and msg.get("cmd") == "stop":
                    break
                if isinstance(msg, dict) and msg.get("cmd") == "update_params":
                    if "pitch" in msg:
                        rvc.set_key(msg["pitch"])
                    if "index_rate" in msg:
                        rvc.set_index_rate(msg["index_rate"])
                    if "protect" in msg:
                        protect = msg["protect"]
                    if "silence_threshold_db" in msg:
                        _silence_rms = 10 ** (msg["silence_threshold_db"] / 20.0)
            except Exception:
                pass

            try:
                data, overflowed = stream_in.read(block_44k)
            except Exception as e:
                evt_q.put({"event": "waveform_overflow", "detail": str(e)})
                continue
            if overflowed:
                evt_q.put({"event": "waveform_overflow", "detail": "overflow"})

            x_44k = data.squeeze()
            x_44k_t = _torch.from_numpy(x_44k).to(infer_device)
            x_16k_t = _resample_44_16(x_44k_t.unsqueeze(0)).squeeze(0)

            # Slide rolling buffer: shift left, append new block at end
            rolling_buf = _torch.roll(rolling_buf, -block_16k)
            rolling_buf[-block_16k:] = x_16k_t[:block_16k]

            # --- Hold/release gate ---
            input_rms = float(_torch.sqrt(_torch.mean(x_16k_t ** 2)).item())
            if input_rms >= _silence_rms:
                _gate_gain = 1.0
                _hold_count = _HOLD_BLOCKS
                _release_count = _RELEASE_BLOCKS
            elif _hold_count > 0:
                _hold_count -= 1
                _gate_gain = 1.0
            elif _release_count > 0:
                _release_count -= 1
                _gate_gain = _release_count / _RELEASE_BLOCKS
            else:
                _gate_gain = 0.0

            # --- RVC inference (always runs to keep MPS kernels warm) ---
            out = rvc.infer(
                rolling_buf,
                block_16k,
                skip_head_frames,
                return_length_frames,
                "rmvpe",
                protect,
            )
            infer_wav = out if isinstance(out, _torch.Tensor) else _torch.from_numpy(out)
            infer_wav = infer_wav.to(infer_device)

            if _gate_gain == 0.0:
                # Silent: output zeros, reset SOLA buffer to avoid crossfade bleed
                out_48k = np.zeros(block_48k, dtype=np.float32)
                sola_buf.zero_()
            else:
                # --- RMS volume envelope ---
                # Scale output RMS to match input RMS (capped at 1.0x gain).
                # Prevents the model from outputting at full volume during quiet speech.
                out_rms = float(_torch.sqrt(_torch.mean(infer_wav ** 2)).item()) + 1e-8
                rms_scale = min(input_rms / out_rms, 1.0)
                infer_wav = infer_wav * (rms_scale * _gate_gain)

                # --- SOLA crossfade (RVC-WebUI-MacOS algorithm) ---
                # Find offset in infer_wav[0:sola_buf+sola_search] that best
                # correlates with sola_buf (previous block's tail). This alignment
                # removes the block-boundary click caused by async inference drift.
                search_len = sola_buf_48k + sola_search_48k
                if infer_wav.shape[0] >= search_len:
                    conv_in = infer_wav[None, None, :search_len]
                    cor_nom = _F.conv1d(conv_in, sola_buf[None, None, :])
                    cor_den = _torch.sqrt(
                        _F.conv1d(conv_in ** 2,
                                  _torch.ones(1, 1, sola_buf_48k, device=infer_device))
                        + 1e-8
                    )
                    sola_offset = int(_torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item())
                else:
                    sola_offset = 0

                infer_wav = infer_wav[sola_offset:]

                # Crossfade: blend previous tail (fade_out) with new head (fade_in)
                if infer_wav.shape[0] >= sola_buf_48k:
                    infer_wav[:sola_buf_48k] = (infer_wav[:sola_buf_48k] * fade_in
                                                + sola_buf * fade_out)
                    # Store new tail for next block's crossfade
                    end = sola_buf_48k + block_48k
                    if infer_wav.shape[0] >= end:
                        sola_buf[:] = infer_wav[block_48k:end]

                out_block = infer_wav[:block_48k]
                out_48k = out_block.cpu().numpy().astype(np.float32)

            try:
                stream_out.write(out_48k.reshape(-1, 1))
            except Exception:
                pass

            if audio_save_q is not None:
                try:
                    audio_save_q.put_nowait(out_48k.copy())
                except queue.Full:
                    pass

            if audio_in_save_q is not None:
                try:
                    x_48k = _resample_44_48(x_44k_t.unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)
                    audio_in_save_q.put_nowait(x_48k.copy())
                except queue.Full:
                    pass

            evt_q.put({
                "event": "waveform_in",
                "samples": base64.b64encode(x_44k[::decim].tobytes()).decode(),
                "sr": 48000,
            })
            evt_q.put({
                "event": "waveform_out",
                "samples": base64.b64encode(out_48k[::decim].tobytes()).decode(),
                "sr": 48000,
            })

    except Exception as exc:
        import traceback as _tb
        evt_q.put({"event": "error", "error": str(exc) + "\n" + _tb.format_exc()})

    finally:
        # Signal save thread to flush and encode — happens after audio loop exits
        if audio_save_q is not None:
            evt_q.put({"event": "debug", "message": f"Sending sentinel to save queue, items waiting: {audio_save_q.qsize()}"})
            audio_save_q.put(_SAVE_SENTINEL)
            if save_t is not None:
                evt_q.put({"event": "debug", "message": f"Waiting for save thread to finish..."})
                save_t.join(timeout=30)  # up to 30s for pydub/ffmpeg to encode
                evt_q.put({"event": "debug", "message": f"Save thread finished"})

        if audio_in_save_q is not None:
            audio_in_save_q.put(_SAVE_SENTINEL)
            if save_in_t is not None:
                save_in_t.join(timeout=30)

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

