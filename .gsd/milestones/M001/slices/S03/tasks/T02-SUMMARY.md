---
id: T02
parent: S03
milestone: M001
provides:
  - backend/app/realtime.py — complete audio engine with RealtimeManager singleton, RealtimeSession dataclass, AudioProcessor Protocol, RVCProcessor, _audio_loop thread function, and manager singleton
key_files:
  - backend/app/realtime.py
key_decisions:
  - D027 (in-memory session dict, single active session at a time — mirrors D017 TrainingManager)
  - D028 (separate sd.InputStream + sd.OutputStream at native rates 44100 Hz in / 48000 Hz out)
  - D029 (daemon threading.Thread for audio loop; deque for waveform handoff)
  - D030 (asyncio polls _waveform_deque every 50ms for WebSocket delivery)
  - D031 (block_48k=9600, 200ms blocks, ~37ms steady-state latency on MPS)
  - D032 (block_48k // 800 = 12× decimation to ~800 waveform points per message)
patterns_established:
  - AudioProcessor Protocol as plain class with process() method (no typing.Protocol — easier testing)
  - RVCProcessor wraps rvc.infer() as AudioProcessor stage for future pipeline extensibility
  - _compute_block_params() helper centralises block size formula (skip_head/return_length in FRAMES)
  - Silence warmup call before stream.start() — avoids JIT compile spike on first real block
  - None sentinel pushed to _waveform_deque on loop exit — tells WS handler to send {type:"done"}
  - update_params() hot-updates pitch/index_rate/protect on live session without restart
observability_surfaces:
  - stdout JSON {"event":"session_start","session_id":...,"profile_id":...,"input_device":...,"output_device":...}
  - stdout JSON {"event":"session_stop","session_id":...}
  - stdout JSON {"event":"session_error","session_id":...,"error":"..."}
  - stdout JSON {"event":"waveform_overflow","session_id":...}
  - session.status and session.error fields persist on RealtimeSession after failure
  - GET /api/realtime/status (wired in T03) will expose {active, session_id, error}
duration: ~45min
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T02: Implement RealtimeSession, RealtimeManager, and audio loop

**Built `backend/app/realtime.py` — the complete audio engine: AudioProcessor Protocol, RVCProcessor, RealtimeSession dataclass, RealtimeManager singleton with start/stop/get/active/update_params, _audio_loop thread function, and module-level manager singleton.**

## What Happened

Implemented the full audio engine in `backend/app/realtime.py` (~300 lines), mirroring the `TrainingManager`/`TrainingJob` pattern from `backend/app/training.py` (D017).

Key implementation choices:

**Block parameter formula** (`_compute_block_params`): Centralized the RMVPE-aligned block size math from S03-RESEARCH.md. All skip_head/return_length values are in FRAMES (÷ 160), not samples. Validated: `skip_head_frames=11 + return_length_frames=20 == p_len=31` for block_48k=9600.

**RVC constructor signature**: The actual `rtrvc.py` RVC class takes `(key, formant, pth_path, index_path, index_rate)` — `formant` is required (not in the task plan's example). Set `formant=0` for no formant shift. The `f0method` and `protect` are per-infer-call args, not constructor args.

**Separate streams**: `sd.InputStream(samplerate=44100)` + `sd.OutputStream(samplerate=48000)` — MacBook Pro Mic is natively 44100 Hz; `sd.Stream` with mixed rates would be more complex.

**Fairseq fix ordering**: `os.environ["rmvpe_root"]` set, `sys.path.insert(0, rvc_root)` done, `Config.use_insecure_load()` called, then `from infer.lib.rtrvc import RVC`. All imports are deferred until `start_session()` so the module imports cleanly without RVC_ROOT at import time.

**None sentinel**: `_audio_loop` appends `None` to `_waveform_deque` in its `finally` block — signals the T03 WS handler to send `{type:"done"}` and close. This matches the contract test's `_make_fake_session()` helper which pre-loads the deque with 2 messages + None.

**update_params**: Uses `session._rvc.set_key()` and `session._rvc.set_index_rate()` for hot param changes. `protect` is stored on the session and read on each loop iteration.

## Verification

```
# Module imports cleanly (no RVC_ROOT needed at import time)
conda run -n rvc python -c "
from backend.app.realtime import RealtimeManager, RealtimeSession, AudioProcessor, RVCProcessor, manager
print('imports ok')
"
# → imports ok

# manager singleton confirmed
conda run -n rvc python -c "
from backend.app.realtime import manager
print(type(manager).__name__)   # → RealtimeManager
print(manager.active_session()) # → None
"
# → RealtimeManager
# → None

# Block params math validated
_compute_block_params(9600) = {
  block_48k: 9600, block_44k: 8820, block_16k: 3200,
  f0_extractor_frame: 4960, total_16k_samples: 4960, p_len: 31,
  return_length_frames: 20, skip_head_frames: 11
}
# skip_head + return_length == p_len: 11 + 20 == 31 ✓
# total_16k_samples == f0_extractor_frame: 4960 == 4960 ✓

# No regressions in existing tests
conda run -n rvc pytest backend/tests/ --ignore=backend/tests/test_realtime.py
# → 25 passed, 1 skipped

# Contract tests: all 10 collect and fail with correct error (router not yet wired)
conda run -n rvc pytest backend/tests/test_realtime.py -v
# → 10 errors: AttributeError: module 'backend.app.routers' has no attribute 'realtime'
# (Expected — router is T03's job)
```

## Diagnostics

- Import the module cleanly: `python -c "from backend.app.realtime import manager; print(manager.active_session())"`
- grep structured events: `grep -E 'session_start|session_stop|session_error|waveform_overflow'` in server stdout
- Post-session failure state: `manager.get_session(session_id).status` and `.error` persist in `_sessions` dict
- GET /api/realtime/status (wired in T03) will surface `{active, session_id, error}`

## Deviations

**RVC constructor signature**: The task plan example showed `RVC(pitch, pth_path, index_path, index_rate, "rmvpe", protect, rvc_root)`. The actual `rtrvc.py` signature is `RVC(key, formant, pth_path, index_path, index_rate, n_cpu, device, ...)`. Added `formant=0` (no formant shift). `f0method` and `protect` are per-call args to `rvc.infer()`, not constructor args. This is correct per the actual rtrvc.py source.

**AudioProcessor as plain class**: Implemented `AudioProcessor` as a plain class with a `process()` stub raising `NotImplementedError` rather than `typing.Protocol`. This avoids Protocol covariance complexity in tests while satisfying R006's extensibility contract. `RVCProcessor` subclasses it directly.

## Known Issues

- The T03 router's `test_params_200_ok` test accesses `fake_session.rvc` (no underscore), but `RealtimeSession` uses `_rvc`. T03's router will need to call `manager.update_params(session_id, ...)` rather than accessing `session._rvc` directly, OR the router tests already accommodate this via the mock (the mock's `get_session` returns a MagicMock that has both `.rvc` and `._rvc` stubs). This is a T03 concern — T03 should use `manager.update_params()` for hot param updates.

## Files Created/Modified

- `backend/app/realtime.py` — new: complete audio engine module (~300 lines)
