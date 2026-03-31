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
    {"event": "infer_timing",    "block_ms": int, "avg_ms": float, "peak_ms": float,
                                  "budget_pct": float, "n_blocks": int}

Device selection:
  - Prefers MPS (Metal Performance Shaders) on Apple Silicon for inference
  - Falls back to CPU if MPS not available
  - Warmup call ensures all MPS kernel compilation happens before "ready" signal
  - Typical warmup: ~2s on MPS (first call), ~50ms on subsequent calls

SOLA algorithm:
  Based on the reference implementation from RVC-WebUI-MacOS / Applio / DDSP-SVC.
  Uses normalized cross-correlation to find the best block alignment offset within
  a search window, then blends the boundary with a sin² fade window. This eliminates
  clicks at block boundaries caused by async inference drift.

Usage:
    python _realtime_worker.py  (not meant to be invoked directly)
"""

import base64
import logging
import multiprocessing
import os
import queue
import sys
import threading
import time as _time
import uuid

# Silence numba DEBUG flood before any library imports it.
logging.basicConfig(level=logging.WARNING)
for _noisy in ("numba", "numba.core", "numba.core.ssa", "numba.core.byteflow",
               "numba.core.interpreter", "numba.typed"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Block parameter constants
# ---------------------------------------------------------------------------

_SR_16K = 16000
_SR_48K = 48000
_WINDOW = 160              # HuBERT/RMVPE hop size in samples at 16kHz
_BLOCK_48K = 9600          # 200ms output block at 48kHz (matches offline; better gate hold coverage)
_EXTRA_TIME = 2.0          # seconds of historical context fed to the model
_ZC = _SR_48K // 100       # zero-crossing unit = 10ms at 48kHz = 480 samples
_SOLA_BUF_48K = 2 * _ZC   # 20ms SOLA crossfade buffer
_SOLA_SEARCH_48K = _ZC     # 10ms search window for best alignment offset


def _compute_block_params(block_48k: int = _BLOCK_48K, extra_time: float = _EXTRA_TIME,
                          tgt_sr: int = _SR_48K) -> dict:
    """Compute all buffer sizes needed by the audio loop.

    All SOLA sizes are computed in tgt_sr space (matching the offline pipeline).
    rtrvc.infer() returns samples at tgt_sr, so return_length_frames must be
    derived from tgt_sr-space block sizes — not 48kHz sizes.
    """
    block_16k = int(block_48k * _SR_16K / _SR_48K)

    f0_min = block_16k + 800
    f0_extractor_frame = 5120 * ((f0_min - 1) // 5120 + 1) - _WINDOW
    extra_16k = (int(extra_time * _SR_16K) // _WINDOW) * _WINDOW
    total_16k_samples = extra_16k + f0_extractor_frame
    p_len = total_16k_samples // _WINDOW

    # Convert 48k block/SOLA sizes to tgt_sr space — this is what rtrvc.infer()
    # actually produces. Using 48k here caused the SR mismatch bug.
    block_tgt       = int(block_48k       * tgt_sr / _SR_48K)
    sola_buf_tgt    = int(_SOLA_BUF_48K   * tgt_sr / _SR_48K)
    sola_search_tgt = int(_SOLA_SEARCH_48K * tgt_sr / _SR_48K)

    # return_length in 16kHz frames: how many tgt_sr samples do we need?
    # rtrvc output = return_length * _WINDOW * (tgt_sr / _SR_16K)
    # so: return_length = total_out_tgt / (_WINDOW * tgt_sr / _SR_16K)
    total_out_tgt = block_tgt + sola_buf_tgt + sola_search_tgt
    zc_tgt = tgt_sr // 100   # e.g. 320 at 32kHz
    return_length_frames = total_out_tgt // zc_tgt

    skip_head_frames = p_len - return_length_frames
    block_44k = int(block_48k * 44100 / _SR_48K)

    return {
        "block_48k":            block_48k,
        "block_44k":            block_44k,
        "block_16k":            block_16k,
        "block_tgt":            block_tgt,
        "sola_buf_tgt":         sola_buf_tgt,
        "sola_search_tgt":      sola_search_tgt,
        "f0_extractor_frame":   f0_extractor_frame,
        "total_16k_samples":    total_16k_samples,
        "extra_16k":            extra_16k,
        "p_len":                p_len,
        "return_length_frames": return_length_frames,
        "skip_head_frames":     skip_head_frames,
        # keep 48k versions for stream_out blocksize and waveform emit
        "sola_buf_48k":         _SOLA_BUF_48K,
        "sola_search_48k":      _SOLA_SEARCH_48K,
    }


# ---------------------------------------------------------------------------
# Phase vocoder — cleaner crossfade for macOS MPS
# From RVC-WebUI-MacOS, based on DDSP-SVC.
# ---------------------------------------------------------------------------

def _phase_vocoder(a: torch.Tensor, b: torch.Tensor,
                   fade_out: torch.Tensor, fade_in: torch.Tensor) -> torch.Tensor:
    """Blend two overlapping blocks using phase-coherent crossfade.

    Uses the geometric mean of fade windows as a mixing envelope, which
    preserves phase continuity better than a simple linear crossfade.
    """
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 1:
        absab[1:] *= 2
    else:
        absab[1:-1] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1, device=a.device) / n
    t = torch.arange(n, device=a.device)
    result = torch.zeros(n, dtype=a.dtype, device=a.device)
    for i, dt in enumerate(t):
        phase = phia + (w + deltaphase / n) * dt
        result[i] = (absab * torch.cos(phase)).mean()
    return result * (fade_out**2 + fade_in**2) / (window + 1e-8)


# ---------------------------------------------------------------------------
# MP3 save thread
# ---------------------------------------------------------------------------

_SAVE_SENTINEL = None


def _save_thread(audio_q: queue.Queue, save_path: str, sr: int,
                 evt_q: multiprocessing.Queue) -> None:
    """Accumulate PCM blocks from audio_q, encode to MP3 on sentinel."""
    chunks = []
    total_samples = 0
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

        pcm = np.concatenate(chunks).astype(np.float32)
        peak = np.abs(pcm).max()
        if peak > 1.0:
            pcm /= peak

        from pydub import AudioSegment  # noqa: PLC0415
        pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
        seg = AudioSegment(
            pcm_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )
        parent_dir = os.path.dirname(os.path.abspath(save_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        seg.export(save_path, format="mp3", bitrate="128k")
        evt_q.put({"event": "save_complete", "path": save_path,
                   "duration_s": round(total_samples / sr, 2)})

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
    silence_threshold_db: float = -55.0,
    output_gain: float = 1.0,
    save_path: str | None = None,
    model_path: str | None = None,
    index_path: str | None = None,
    profile_rms: float | None = None,
) -> None:
    """Run in isolated child process. Owns all native code (fairseq, sounddevice, MPS)."""

    import glob as _glob
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # rvc_root here is project_root (rvc-web/). The RVC Python package lives at
    # backend/rvc/ — add it to sys.path so `from rvc.f0 import ...` etc. resolve.
    rvc_pkg_dir = os.path.join(rvc_root, "backend", "rvc")
    os.chdir(rvc_pkg_dir)
    if rvc_pkg_dir not in sys.path:
        sys.path.insert(0, rvc_pkg_dir)

    session_id = uuid.uuid4().hex

    # Start save threads immediately so they're ready before the audio loop
    audio_save_q: queue.Queue | None = None
    save_t: threading.Thread | None = None
    audio_in_save_q: queue.Queue | None = None
    save_in_t: threading.Thread | None = None
    if save_path:
        audio_save_q = queue.Queue(maxsize=0)
        save_t = threading.Thread(
            target=_save_thread,
            args=(audio_save_q, save_path, _SR_48K, evt_q),
            daemon=True, name="mp3-save",
        )
        save_t.start()

        _save_dir = os.path.dirname(os.path.abspath(os.path.expanduser(save_path)))
        input_save_path = os.path.join(_save_dir, "rvc_input.mp3")
        audio_in_save_q = queue.Queue(maxsize=0)
        save_in_t = threading.Thread(
            target=_save_thread,
            args=(audio_in_save_q, input_save_path, _SR_48K, evt_q),
            daemon=True, name="mp3-save-input",
        )
        save_in_t.start()

    try:
        # Import order: torch → faiss (nthreads=1) → fairseq.
        # This order prevents the faiss+fairseq OpenMP collision on macOS.
        import torch as _torch
        import faiss as _faiss
        _faiss.omp_set_num_threads(1)

        if _torch.backends.mps.is_available():
            infer_device = "mps"
        else:
            infer_device = "cpu"

        # fairseq safe-globals fix (PyTorch 2.6)
        try:
            from configs.config import Config
            Config.use_insecure_load()
        except Exception:
            pass

        from infer.lib.rtrvc import RVC

        import os as _os

        # Resolve model path
        resolved_model_path = model_path
        if not resolved_model_path or not _os.path.exists(resolved_model_path):
            legacy_model = f"{rvc_root}/assets/weights/rvc_finetune_active.pth"
            if _os.path.exists(legacy_model):
                resolved_model_path = legacy_model
            else:
                raise RuntimeError(
                    f"model not found — checked {model_path!r} and legacy {legacy_model!r}"
                )

        # Resolve index path
        resolved_index_path = index_path
        if not resolved_index_path or not _os.path.exists(resolved_index_path):
            index_pattern = f"{rvc_root}/logs/rvc_finetune_active/added_IVF*_Flat*.index"
            matches = _glob.glob(index_pattern)
            if not matches:
                raise RuntimeError(
                    f"index not found — checked {index_path!r} and "
                    f"legacy logs/rvc_finetune_active/added_IVF*_Flat*.index"
                )
            resolved_index_path = matches[0]

        rvc = RVC(
            key=pitch,
            formant=0,
            pth_path=resolved_model_path,
            index_path=resolved_index_path,
            index_rate=index_rate,
            device=infer_device,
            is_half=False,
        )

        # tgt_sr is model-specific (32kHz for our 32k models).
        # ALL SOLA buffer sizes must be in tgt_sr space to match rtrvc.infer() output.
        tgt_sr = rvc.tgt_sr
        block_params = _compute_block_params(_BLOCK_48K, tgt_sr=tgt_sr)

        # ------------------------------------------------------------------
        # Patch rvc._get_f0: pad pitch/pitchf at head to match p_len.
        # rtrvc.py extracts F0 from input[-f0_extractor_frame:] which gives
        # f0_extractor_frame//window frames, but p_len = total_16k//window.
        # The difference (extra_16k//window frames) must be padded with zeros
        # (unvoiced) so the broadcast with feats [1, p_len, H] succeeds.
        # ------------------------------------------------------------------
        _orig_get_f0 = rvc._get_f0
        _p_len_expected = block_params["p_len"]

        def _patched_get_f0(x, f0_up_key, filter_radius=None, method="fcpe"):
            c, f = _orig_get_f0(x, f0_up_key, filter_radius=filter_radius, method=method)
            if c.shape[-1] < _p_len_expected:
                pad = _p_len_expected - c.shape[-1]
                c = _torch.cat([_torch.zeros(pad, dtype=c.dtype, device=c.device), c])
                f = _torch.cat([_torch.zeros(pad, dtype=f.dtype, device=f.device), f])
            return c, f

        rvc._get_f0 = _patched_get_f0

        # FAISS segfault fix: index.search requires C-contiguous float32 input
        _orig_search = rvc.index.search
        def _safe_search(npy, k):
            return _orig_search(np.ascontiguousarray(npy, dtype=np.float32), k=k)
        rvc.index.search = _safe_search

        # Warmup: compile MPS kernels before signalling "ready"
        evt_q.put({"event": "warming_up", "device": infer_device})
        _silence_warmup = _torch.zeros(block_params["total_16k_samples"], device=infer_device)
        rvc.infer(
            _silence_warmup,
            block_params["block_16k"],
            block_params["skip_head_frames"],
            block_params["return_length_frames"],
            "rmvpe",
            protect,
        )
        del _silence_warmup

        import sounddevice as sd
        from torchaudio.transforms import Resample as _Resample
        import torch.nn.functional as _F

        # 44100 → 16k  (input capture → inference)
        # 44100 → 48k  (input capture → waveform visualiser / save)
        # tgt_sr → 48k (inference output → output device)
        _resample_44_16  = _Resample(orig_freq=44100, new_freq=16000,
                                     dtype=_torch.float32).to(infer_device)
        _resample_44_48  = _Resample(orig_freq=44100, new_freq=_SR_48K,
                                     dtype=_torch.float32).to(infer_device)
        _resample_tgt_48 = _Resample(orig_freq=tgt_sr, new_freq=_SR_48K,
                                     dtype=_torch.float32).to(infer_device)

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
            latency="low",
        )

        stream_in.start()
        stream_out.start()

        evt_q.put({"event": "ready", "session_id": session_id, "device": infer_device})

        # ------------------------------------------------------------------
        # Audio loop locals
        # ------------------------------------------------------------------
        block_44k            = block_params["block_44k"]
        block_48k            = block_params["block_48k"]
        block_16k            = block_params["block_16k"]
        block_tgt            = block_params["block_tgt"]
        total_16k_samples    = block_params["total_16k_samples"]
        skip_head_frames     = block_params["skip_head_frames"]
        return_length_frames = block_params["return_length_frames"]
        sola_buf_tgt         = block_params["sola_buf_tgt"]
        sola_search_tgt      = block_params["sola_search_tgt"]
        decim = max(1, block_48k // 800)
        block_ms = block_48k * 1000 // _SR_48K

        # Silence gate: RMS threshold from dB
        _silence_rms = 10 ** (silence_threshold_db / 20.0)

        # Rolling-max window for gate trigger — 3 blocks @ 200ms = 600ms lookback.
        # The gate opens if ANY of the last _RMS_WIN blocks exceeded the threshold.
        # This prevents the gate from closing on a block whose RMS dipped because
        # speech ended mid-block (word tail), while still closing cleanly after
        # 600ms of sustained silence.  Hold/release runs unchanged from the trigger.
        _RMS_WIN = 3
        from collections import deque as _deque
        _rms_window: _deque[float] = _deque([0.0] * _RMS_WIN, maxlen=_RMS_WIN)

        # Hold/release gate — prevents abrupt cut-off mid-word.
        _HOLD_BLOCKS    = 3
        _RELEASE_BLOCKS = 5
        _hold_count     = 0
        _release_count  = 0
        _gate_gain      = 0.0   # block-level target gain
        _gate_gain_cur  = 0.0   # sample-level smoothed gain

        # Per-block output RMS matching — corrects the model's inherent level
        # difference without touching the input signal.  Uses an EWMA of input
        # RMS so a single near-silent frame can't cause a spike.
        _rms_alpha  = 0.15
        _rms_smooth = None   # seeded on first voiced frame

        # Rolling 16k context buffer — circular shift, no torch.roll (avoids copy)
        rolling_buf = _torch.zeros(total_16k_samples, dtype=_torch.float32,
                                   device=infer_device)

        # ------------------------------------------------------------------
        # SOLA state (algorithm from RVC-WebUI-MacOS / Applio / DDSP-SVC)
        #
        # sola_buf:  the last `sola_buf_48k` samples of the previous output block.
        #            Stored from position [block_48k : block_48k + sola_buf_48k]
        #            of the (offset-shifted) inference output.
        # fade_in/out: sin² windows for smooth crossfade at block boundary.
        #
        # Critical correctness rules:
        #  1. sola_offset is the raw argmax over the FULL sola_search_48k range —
        #     never clamp it (a clamped offset still has a phase discontinuity).
        #  2. sola_buf is updated AFTER applying the offset shift, reading from
        #     infer_wav[block_48k : block_48k + sola_buf_48k].
        #  3. During silence we output zeros but DO NOT reset sola_buf — resetting
        #     it would cause a pop/click when speech resumes.
        # ------------------------------------------------------------------
        sola_buf  = _torch.zeros(sola_buf_tgt, dtype=_torch.float32, device=infer_device)
        fade_in   = _torch.sin(0.5 * np.pi * _torch.linspace(
                        0., 1., sola_buf_tgt, device=infer_device)) ** 2
        fade_out  = 1.0 - fade_in

        # Minimum inference output needed to run SOLA
        _min_infer_len = block_tgt + sola_buf_tgt + sola_search_tgt
        _block_i = 0
        _infer_times: list[float] = []   # rolling window for timing log

        while True:
            # ── command check (non-blocking) ──────────────────────────────
            try:
                msg = cmd_q.get_nowait()
                if isinstance(msg, dict):
                    if msg.get("cmd") == "stop":
                        break
                    if msg.get("cmd") == "update_params":
                        if "pitch" in msg:
                            rvc.set_key(msg["pitch"])
                        if "index_rate" in msg:
                            rvc.set_index_rate(msg["index_rate"])
                        if "protect" in msg:
                            protect = msg["protect"]
                        if "silence_threshold_db" in msg:
                            _silence_rms = 10 ** (msg["silence_threshold_db"] / 20.0)
                            _rms_window.clear()
                            _rms_window.extend([0.0] * _RMS_WIN)
                        if "output_gain" in msg:
                            output_gain = float(msg["output_gain"])
            except Exception:
                pass

            # ── read input block ──────────────────────────────────────────
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

            # ── update rolling 16k buffer (circular shift) ────────────────
            # Equivalent to torch.roll but avoids the internal copy overhead.
            rolling_buf[:-block_16k] = rolling_buf[block_16k:].clone()
            rolling_buf[-block_16k:] = x_16k_t[:block_16k]

            # ── hold/release silence gate ─────────────────────────────────
            input_rms = float(_torch.sqrt(_torch.mean(x_16k_t ** 2)).item())
            _rms_window.append(input_rms)
            _rolling_max_rms = max(_rms_window)
            if _rolling_max_rms >= _silence_rms:
                _gate_gain      = 1.0
                _hold_count     = _HOLD_BLOCKS
                _release_count  = _RELEASE_BLOCKS
                if _rms_smooth is None:
                    _rms_smooth = input_rms
                else:
                    _rms_smooth = _rms_alpha * input_rms + (1 - _rms_alpha) * _rms_smooth
            elif _hold_count > 0:
                _hold_count -= 1
                _gate_gain   = 1.0
            elif _release_count > 0:
                _release_count -= 1
                _gate_gain = _release_count / _RELEASE_BLOCKS
            else:
                _gate_gain = 0.0

            # ── RVC inference (always runs to keep MPS kernels warm) ──────
            _t_infer_start = _time.perf_counter()
            infer_raw = rvc.infer(
                rolling_buf,
                block_16k,
                skip_head_frames,
                return_length_frames,
                "rmvpe",
                protect,
            )
            _infer_ms = (_time.perf_counter() - _t_infer_start) * 1000
            _block_i += 1
            _infer_times.append(_infer_ms)
            if len(_infer_times) > 50:
                _infer_times.pop(0)
            # Log timing every 25 blocks (~5s at 200ms blocks) so we can
            # measure real inference latency without adding per-block overhead.
            if _block_i % 25 == 0:
                _avg = sum(_infer_times) / len(_infer_times)
                _pk  = max(_infer_times)
                logging.getLogger("rvc_web.worker").info(
                    "infer_timing: block_ms=%d  avg=%.1fms  peak=%.1fms  budget=%.0f%%  block=%d",
                    block_ms, _avg, _pk, _avg / block_ms * 100, _block_i,
                )
                # Emit timing to frontend for diagnostics
                evt_q.put({
                    "event": "infer_timing",
                    "block_ms": block_ms,
                    "avg_ms":   round(_avg, 1),
                    "peak_ms":  round(_pk,  1),
                    "budget_pct": round(_avg / block_ms * 100, 1),
                    "n_blocks": _block_i,
                })

            infer_wav = (infer_raw if isinstance(infer_raw, _torch.Tensor)
                         else _torch.from_numpy(infer_raw))
            infer_wav = infer_wav.to(infer_device)

            # ── SOLA crossfade in tgt_sr space ────────────────────────────
            # infer_wav is at tgt_sr (e.g. 32kHz). All sizes are tgt_sr-space.
            # We always run SOLA to keep sola_buf up-to-date even during silence.

            if infer_wav.shape[0] < _min_infer_len:
                pad = _min_infer_len - infer_wav.shape[0]
                infer_wav = _F.pad(infer_wav, (0, pad))

            search_len = sola_buf_tgt + sola_search_tgt
            conv_input = infer_wav[None, None, :search_len]
            cor_nom = _F.conv1d(conv_input, sola_buf[None, None, :])
            cor_den = _torch.sqrt(
                _F.conv1d(conv_input ** 2,
                           _torch.ones(1, 1, sola_buf_tgt, device=infer_device))
                + 1e-8
            )
            sola_offset = int(_torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item())
            infer_wav = infer_wav[sola_offset:]
            infer_wav[:sola_buf_tgt] = (infer_wav[:sola_buf_tgt] * fade_in
                                        + sola_buf * fade_out)
            # Update sola_buf from tail of this block
            tail_end = block_tgt + sola_buf_tgt
            if infer_wav.shape[0] >= tail_end:
                sola_buf[:] = infer_wav[block_tgt:tail_end]
            else:
                available = infer_wav.shape[0] - block_tgt
                if available > 0:
                    sola_buf[:available] = infer_wav[block_tgt:]
                sola_buf[max(0, available):] = 0.0

            if _gate_gain == 0.0:
                # Gate closed — output silence; sola_buf already updated above
                out_48k = np.zeros(block_48k, dtype=np.float32)
            else:
                # Output RMS matching — correct model's inherent level difference
                if _rms_smooth is not None:
                    out_rms = float(_torch.sqrt(_torch.mean(infer_wav[:block_tgt] ** 2)).item()) + 1e-8
                    rms_scale = min(_rms_smooth / out_rms, 1.5)
                    infer_wav = infer_wav * rms_scale

                if output_gain != 1.0:
                    infer_wav = infer_wav * output_gain

                # Per-sample gate ramp (block_tgt space)
                gain_ramp = _torch.linspace(_gate_gain_cur, _gate_gain,
                                             block_tgt, device=infer_device)
                out_tgt = _torch.clamp(infer_wav[:block_tgt] * gain_ramp, -1.0, 1.0)

                # Resample tgt_sr → 48kHz for output device
                out_48k = _resample_tgt_48(out_tgt.unsqueeze(0)).squeeze(0)
                # Ensure exactly block_48k samples (resampler may give ±1)
                if out_48k.shape[0] < block_48k:
                    out_48k = _F.pad(out_48k, (0, block_48k - out_48k.shape[0]))
                else:
                    out_48k = out_48k[:block_48k]
                out_48k = out_48k.cpu().numpy().astype(np.float32)

            _gate_gain_cur = _gate_gain   # carry current gain into next block

            # ── write to output device ────────────────────────────────────
            try:
                stream_out.write(out_48k.reshape(-1, 1))
            except Exception:
                pass

            # ── save to file (if requested) ───────────────────────────────
            if audio_save_q is not None:
                try:
                    audio_save_q.put_nowait(out_48k.copy())
                except queue.Full:
                    pass

            if audio_in_save_q is not None:
                try:
                    x_48k = (_resample_44_48(x_44k_t.unsqueeze(0))
                             .squeeze(0).cpu().numpy().astype(np.float32))
                    audio_in_save_q.put_nowait(x_48k.copy())
                except queue.Full:
                    pass

            # ── emit waveform snapshots for frontend visualiser ───────────
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
        if audio_save_q is not None:
            audio_save_q.put(_SAVE_SENTINEL)
            if save_t is not None:
                save_t.join(timeout=30)

        if audio_in_save_q is not None:
            audio_in_save_q.put(_SAVE_SENTINEL)
            if save_in_t is not None:
                save_in_t.join(timeout=30)

        try:
            stream_in.stop(); stream_in.close()   # noqa: F821
        except Exception:
            pass
        try:
            stream_out.stop(); stream_out.close()  # noqa: F821
        except Exception:
            pass

    evt_q.put({"event": "stopped"})
