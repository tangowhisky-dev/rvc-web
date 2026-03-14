---
estimated_steps: 8
estimated_files: 1
---

# T02: Implement RealtimeSession, RealtimeManager, and audio loop

**Slice:** S03 — Realtime Voice Conversion & Waveform Visualization
**Milestone:** M001

## Description

Build `backend/app/realtime.py` — the complete audio engine. This module owns every aspect of the realtime audio pipeline: loading the RVC model, opening sounddevice streams, driving the rolling buffer, calling RVC inference in a dedicated thread, and publishing waveform snapshots to a deque for WebSocket consumption.

No HTTP is wired here. The router (T03) imports `manager` from this module. All the hard work — fairseq fix, block parameter math, 44100→16000 resampling, warmup call — lives here.

## Steps

1. **Define `AudioProcessor` Protocol and `RVCProcessor`** (satisfies R006):
   ```python
   class AudioProcessor(Protocol):
       def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...

   class RVCProcessor:
       def __init__(self, rvc): self._rvc = rvc
       def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
           # Delegates to RealtimeSession._run_inference — not called directly in the loop,
           # but satisfies the Protocol contract for future pipeline extensibility
           ...
   ```

2. **Define `RealtimeSession` dataclass**:
   ```python
   @dataclass
   class RealtimeSession:
       session_id: str
       profile_id: str
       input_device: int
       output_device: int
       pitch: float = 0.0
       index_rate: float = 0.75
       protect: float = 0.33
       status: str = "starting"   # starting | active | stopped | error
       error: Optional[str] = None
       _rvc: Any = field(default=None, repr=False)
       _stream_in: Any = field(default=None, repr=False)
       _stream_out: Any = field(default=None, repr=False)
       _waveform_deque: collections.deque = field(
           default_factory=lambda: collections.deque(maxlen=40), repr=False
       )
       _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
       _audio_thread: Optional[threading.Thread] = field(default=None, repr=False)
   ```

3. **Define `RealtimeManager` with `start_session`, `stop_session`, `get_session`, `active_session`**:
   - `_sessions: Dict[str, RealtimeSession]` — in-memory dict (mirrors `TrainingManager._jobs`)
   - `start_session(profile_id, input_device_id, output_device_id, pitch, index_rate, protect, rvc_root)`:
     - Validate: check no active session (`active_session() is not None` → raise ValueError for 409)
     - Validate profile status == "trained" via `async with get_db() as db` (caller must be async — this method is an `async def`)
     - Glob for FAISS index: `glob.glob(f"{rvc_root}/logs/rvc_finetune_active/added_IVF*_Flat*.index")`; raise ValueError("no index file found") if empty
     - Set env and sys.path: `os.environ["rmvpe_root"] = f"{rvc_root}/assets/rmvpe"` + `sys.path.insert(0, rvc_root)`
     - Apply fairseq fix: `from configs.config import Config; Config.use_insecure_load()`
     - Import and instantiate RVC: `from infer.lib.rtrvc import RVC; rvc = RVC(pitch, f"{rvc_root}/assets/weights/rvc_finetune_active.pth", index_path, index_rate, "rmvpe", protect, rvc_root)`
     - Compute block params (exact formula from research): `block_48k=9600`, derive `block_16k`, `f0_extractor_frame`, `total_16k_samples`, `p_len`, `return_length_frames`, `skip_head_frames`
     - Run warmup: call `rvc.infer(torch.zeros(total_16k_samples), block_16k, skip_head_frames, return_length_frames, "rmvpe", protect)` once with silence
     - Open streams separately: `sd.InputStream(device=input_device_id, samplerate=44100, channels=1, dtype='float32', blocksize=block_44k)` and `sd.OutputStream(device=output_device_id, samplerate=48000, channels=1, dtype='float32', blocksize=block_48k)` where `block_44k = int(block_48k * 44100 / 48000)`
     - Create `RealtimeSession` and store in `_sessions`
     - Start `_audio_thread = threading.Thread(target=_audio_loop, args=(session, rvc, block params...), daemon=True)`
     - Start both streams; set `session.status = "active"`
     - Return session
   - `stop_session(session_id)`: set `_stop_event`, stop+close both streams, join thread (timeout=2s), set `session.status = "stopped"`
   - `get_session(session_id)`: return `_sessions.get(session_id)`
   - `active_session()`: return first session where `status == "active"`, else None

4. **Implement `_audio_loop(session, rvc, block_params)` — runs in audio thread**:
   - Loop until `session._stop_event.is_set()`
   - `data, overflowed = session._stream_in.read(block_44k)` → shape `(block_44k, 1)`, squeeze to 1D
   - If `overflowed`: emit structured log `{"event":"waveform_overflow","session_id":...}`
   - Resample input 44100→16000: `x_16k = scipy.signal.resample_poly(data.squeeze(), 160, 441)` (ratio = 16000/44100 = 160/441)
   - Maintain rolling buffer `ndarray` of length `total_16k_samples` (initially zeros); shift and append `x_16k`
   - Call `rvc.infer(torch.from_numpy(rolling_buf).float(), block_16k, skip_head_frames, return_length_frames, "rmvpe", protect)` → `out_48k` ndarray
   - Write output: `session._stream_out.write(out_48k.reshape(-1, 1).astype('float32'))`
   - Subsample for waveform: `decim = block_48k // 800` (≈12), take every `decim`-th sample
   - Encode waveforms: `base64.b64encode(arr[::decim].astype('float32').tobytes()).decode('ascii')`
   - Push both messages to `session._waveform_deque`: `{type:"waveform_in", samples:<b64>, sr:48000}` and `{type:"waveform_out", samples:<b64>, sr:48000}`
   - Catch exceptions: set `session.status = "error"`, `session.error = str(e)`, emit structured log, break

5. **Export `manager = RealtimeManager()` at module level.**

6. **Emit structured stdout JSON** on session start, stop, and error:
   ```python
   print(json.dumps({"event": "session_start", "session_id": ..., "profile_id": ..., "input_device": ..., "output_device": ...}), flush=True)
   ```

7. **Add `update_params(session_id, pitch, index_rate, protect)` method to `RealtimeManager`** — calls `rvc.set_key(pitch)` and `rvc.set_index_rate(index_rate)` if provided (hot-update without restarting session).

8. Verify imports cleanly and module-level `manager` is accessible.

## Must-Haves

- [ ] `AudioProcessor` Protocol and `RVCProcessor` class defined and importable
- [ ] `RealtimeSession` dataclass with all required fields
- [ ] `RealtimeManager.start_session` is `async def` (calls `get_db()`)
- [ ] Fairseq fix (`Config.use_insecure_load()`) called before `RVC.__init__`
- [ ] `rmvpe_root` env var set before `from infer.lib.rtrvc import RVC`
- [ ] Block parameter formula matches research exactly (skip_head/return_length in FRAMES not samples)
- [ ] Silence warmup call before `stream.start()`
- [ ] Separate `sd.InputStream` and `sd.OutputStream` (not `sd.Stream` — different native rates)
- [ ] FAISS index path via glob; ValueError raised if empty
- [ ] Profile status gate: ValueError if status != "trained"
- [ ] `_audio_loop` runs in `daemon=True` thread; catches exceptions and sets `session.error`
- [ ] Waveform subsampled to ~800 points before base64 encoding
- [ ] `manager` singleton exported at module level
- [ ] Structured JSON stdout events on session_start, session_stop, session_error

## Verification

```bash
# Module imports without error (no RVC_ROOT needed at import time)
cd /Users/tango16/code/rvc-web
conda run -n rvc python -c "
from backend.app.realtime import RealtimeManager, RealtimeSession, AudioProcessor, RVCProcessor, manager
print('RealtimeManager:', RealtimeManager)
print('manager:', manager)
print('imports ok')
"

# Confirm manager is a singleton instance
conda run -n rvc python -c "
from backend.app.realtime import manager
print(type(manager).__name__)  # → RealtimeManager
print(manager.active_session())  # → None
"
```

## Observability Impact

- Signals added/changed:
  - `{"event":"session_start", "session_id":..., "profile_id":..., "input_device":..., "output_device":...}` — printed at session start
  - `{"event":"session_stop", "session_id":...}` — printed at clean stop
  - `{"event":"session_error", "session_id":..., "error":"..."}` — printed on audio thread crash
  - `{"event":"waveform_overflow", "session_id":...}` — printed when sounddevice reports overflow
- How a future agent inspects this: `GET /api/realtime/status` (wired in T03); grep stdout JSON for `session_start|session_error|waveform_overflow`
- Failure state exposed: `session.status == "error"` + `session.error` string persists in `_sessions` dict; `GET /api/realtime/status` will expose `{active:false, error:"..."}` (T03 wiring)

## Inputs

- `backend/app/training.py` — `TrainingJob`/`TrainingManager` dataclass + dict pattern to mirror exactly (D017 pattern)
- `backend/app/db.py` — `get_db()` async context manager for profile status validation
- `S03-RESEARCH.md` → Critical: Block Parameter Formula — exact values and formulas to implement
- `S03-RESEARCH.md` → Common Pitfalls — wrong units trap; must use FRAMES for skip_head/return_length
- `S03-RESEARCH.md` → Constraints — rmvpe_root, sys.path, Config.use_insecure_load() ordering
- `Retrieval-based-Voice-Conversion-WebUI/configs/config.py` — `Config.use_insecure_load()` method
- `Retrieval-based-Voice-Conversion-WebUI/infer/lib/rtrvc.py` — `RVC` class signature

## Expected Output

- `backend/app/realtime.py` — complete audio engine module (~250 lines): `AudioProcessor` Protocol, `RVCProcessor`, `RealtimeSession` dataclass, `RealtimeManager` singleton with start/stop/get/active/update_params, `_audio_loop` thread function, `manager` singleton
