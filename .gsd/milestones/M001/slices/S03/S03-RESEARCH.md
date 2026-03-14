# S03 — Research: Realtime Voice Conversion & Waveform Visualization

**Date:** 2026-03-14
**Slice:** M001/S03
**Requirements owned:** R003, R004, R005, R006

---

## Summary

S03 builds the realtime audio loop (Python backend) and waveform WebSocket (browser). The core inference path was fully probed and all critical parameters confirmed working. **RVC.infer() runs in ~37ms for a 200ms block after warmup — 0.19× realtime ratio on MPS, well within budget.** Three hard pre-requisites must be handled before inference works: the fairseq safe-globals PyTorch 2.6 fix (same as D025 but in inference init, not training), precise block parameter computation for `rtrvc.py`, and the MacBook Pro Microphone's 44.1kHz sample rate requiring resampling before HuBERT input.

The sounddevice API is well understood. The `sd.Stream` callback pattern gives the cleanest realtime loop — capture from mic input and write to BlackHole output in a single stream with separate device IDs. The waveform WebSocket runs at ~20fps as a separate asyncio task off the main audio thread.

The frontend for this slice is minimal — the plan calls for a basic Realtime tab (full polish is S04's job). The waveform canvas renders in a Web Worker via OffscreenCanvas per D005.

The FAISS index path produced by S02 contains a variable N in its name (IVF**N**_Flat). S03 must glob for `added_IVF*_Flat*.index` rather than hardcoding the name.

---

## Recommendation

**Build the audio loop in a dedicated `threading.Thread` (not asyncio), with `sounddevice.Stream` in callback mode.** The callback fires at fixed intervals; it enqueues processed audio into a thread-safe `collections.deque` that the output callback writes from. RVC inference happens in the callback thread. This avoids GIL contention between asyncio and the audio path and matches how PortAudio expects to be driven.

WebSocket waveform delivery runs as an `asyncio.Task` that periodically snapshots the rolling waveform buffer, base64-encodes, and sends at ~20fps.

**Session management pattern:** mirror `TrainingManager` — a `RealtimeSession` dataclass + `RealtimeManager` singleton, in-memory, keyed by `session_id`. One session at a time. The existing `manager` pattern from `backend/app/training.py` is the established precedent.

**Block size:** Start with `block_48k = 9600` (200ms, ~37ms infer latency). This gives 5.4× headroom. If latency is perceptible, try `block_48k = 4800` (100ms, first call ~1.9s warmup then ~37ms steady state — RMVPE JIT warmup only hits once per session).

---

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Audio I/O | `sounddevice.Stream(device=(in, out), callback=...)` | PortAudio-backed; handles mixed SR, blocksize, channel counts; already in env |
| Resample 44100/48000 → 16000 | `scipy.signal.resample_poly` | Integer ratio resampling; exact; already in env |
| Base64 encode float32 | Python `base64.b64encode(arr.tobytes())` | No extra dep; decodable in JS as `Float32Array.from(atob(...))` |
| Fairseq safe-globals | `torch.serialization.add_safe_globals([Dictionary])` | Same fix as D025 — must apply before `RVC.__init__` or hubert load fails with `UnpicklingError` |
| PyTorch 2.6 + fairseq | `Config.use_insecure_load()` pattern in `configs/config.py` | Exact fix already coded in upstream; apply same pattern in `RealtimeSession.__init__` |
| Session state | Mirror `TrainingManager`/`TrainingJob` dataclass pattern | Established S02 pattern; consistent, tests expect this shape |

---

## Existing Code and Patterns

- `Retrieval-based-Voice-Conversion-WebUI/infer/lib/rtrvc.py` — The `RVC` class. `__init__` loads HuBERT + synthesizer + FAISS index. `infer(input_wav, block_frame_16k, skip_head, return_length, f0method, protect)` → `np.ndarray` at `rvc.tgt_sr` (48000). Import with `sys.path.insert(0, rvc_root)` and `PYTHONPATH=rvc_root`. **Must set `rmvpe_root` env var before import** (same as training).
- `Retrieval-based-Voice-Conversion-WebUI/configs/config.py:Config.use_insecure_load()` — Applies `torch.serialization.add_safe_globals([Dictionary])` fix for PyTorch 2.6 + fairseq. Call this before `RVC.__init__` or hubert load fails.
- `backend/app/training.py` — `TrainingManager`/`TrainingJob` pattern. `RealtimeManager`/`RealtimeSession` should mirror this exactly (dataclass + dict + singleton). Key patterns: `asyncio.Queue` for inter-task comms, `_update_db_status()` pattern for DB side effects.
- `backend/app/routers/training.py` — Two-router pattern (D023): `router = APIRouter(prefix='/api/...')` + `ws_router = APIRouter()` (no prefix). Realtime router must follow same pattern for `WS /ws/realtime/{session_id}`.
- `backend/app/main.py` — Register both routers here. Pattern established.
- `backend/tests/test_training.py` — Mock pattern: `patch('backend.app.routers.realtime.manager')` at router-level (not module-level). `starlette.testclient.TestClient.websocket_connect()` for WS tests.
- `backend/tests/conftest.py` — Async ASGI client fixture. Same fixture works for realtime tests.

---

## Critical: Block Parameter Formula (Proved via Empirical Testing)

The `RVC.infer()` parameters are not intuitive. The following formula was verified to produce correct, non-erroring output with output_len matching expected block duration:

```python
SR_16K = 16000
SR_48K = 48000  # or whatever native SR is
WINDOW = 160    # rvc.window (HuBERT hop length)

# Given: block_48k = desired output block size in native SR samples (e.g. 9600 = 200ms at 48kHz)
block_16k_samples = int(block_48k * SR_16K / SR_48K)  # samples at 16kHz

# RMVPE pads to multiple of 5120 samples at 16kHz
f0_extractor_frame = block_16k_samples + 800
f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - WINDOW

# Rolling buffer size: exactly f0_extractor_frame samples at 16kHz
total_16k_samples = f0_extractor_frame  # == input_wav.shape[0]

p_len = total_16k_samples // WINDOW  # frame count for hubert/pitch

# Parameters for rvc.infer():
return_length_frames = block_16k_samples // WINDOW
skip_head_frames = p_len - return_length_frames

# infer() call:
result = rvc.infer(
    input_wav=rolling_buffer_16k,        # torch.Tensor, shape (total_16k_samples,)
    block_frame_16k=block_16k_samples,   # samples at 16kHz (NOT frames)
    skip_head=skip_head_frames,           # in FRAMES (not samples)
    return_length=return_length_frames,   # in FRAMES (not samples)
    f0method='rmvpe',
    protect=protect_value,
)
# result: np.ndarray at rvc.tgt_sr (48000 Hz), shape (block_48k,)
```

**Confirmed timings (silence input, MPS device, after JIT warmup):**
- `block_48k=9600` (200ms): infer ~37ms → **0.19× realtime** ✓
- `block_48k=4800` (100ms): first call ~1.9s (warmup), steady ~37ms
- Output correctly equals `block_48k` samples at 48000 Hz

**Audio loop data flow:**
```
Mic (device 2, 44100 Hz) → resample_poly(16000, 44100) → rolling_buffer_16k (total_16k_samples)
→ RVC.infer() → np.ndarray (block_48k @ 48000 Hz)
→ resample if needed → BlackHole output (device 1, 48000 Hz)
```

---

## Constraints

- **`rmvpe_root` env var** must be set before `from infer.lib.rtrvc import RVC` is called. Same requirement as training. Set in session init, not globally.
- **`fairseq` safe-globals fix** must be applied before `RVC.__init__` loads HuBERT. Use `Config.use_insecure_load()` from `configs/config.py` or inline equivalent.
- **Working directory for RVC import**: must add `rvc_root` to `sys.path` (or set `PYTHONPATH`) so `from rvc.synthesizer import ...` etc. resolves. The training pipeline does this via `cwd=rvc_root` in subprocesses; the realtime loop runs in-process so must set `sys.path` before import.
- **Audio device IDs**: MacBook Pro Microphone = id 2 (44100 Hz, 1 channel), BlackHole 2ch = id 1 (48000 Hz, 2 channels). IDs are stable for this machine but always retrieve dynamically from `sounddevice.query_devices()`.
- **FAISS index path is not predictable**: S02 built `added_IVF6_Flat_nprobe_1_rvc_finetune_active_v2.index` but the `N` in `IVF{N}` depends on data size. S03 must glob `{rvc_root}/logs/rvc_finetune_active/added_IVF*_Flat*.index` and pick the first match. Raise a 400 if no index found.
- **`profile.status == "trained"`** gate: S03 must verify the profile is trained before starting a session (confirmed by S02 summary: status is "trained" in SQLite after successful training).
- **sounddevice Stream callback threading**: The `sounddevice` callback fires in a C thread (not Python thread, not async). `asyncio` cannot be used directly inside. Use `collections.deque` + `threading.Event` for signalling the main audio processing thread.
- **`torchaudio.transforms.Resample`** is used inside `rtrvc.py` for formant resampling; `torchaudio 2.10.0` is present in env.
- **`scipy.signal.resample_poly`** available for mic SR conversion before and after RVC. Confirmed working: 44100→16000 and 48000→16000.
- **Two-router constraint (D023)**: `WS /ws/realtime/{session_id}` cannot be on the prefixed `/api/realtime` router. Use a separate `ws_router = APIRouter()` (no prefix).
- **Session produces waveform data**: waveform messages contain `samples` (float32 as base64), `sr` (48000), `type` ("waveform_in" or "waveform_out"). Subsample to ~800 points per message for browser rendering (decimation factor = block_48k // 800).

---

## Common Pitfalls

- **Wrong units for `skip_head`/`return_length`**: These are in FRAMES (samples ÷ 160), NOT samples. Confirmed via testing. Passing samples causes "size mismatch" or "padded input size 0" RuntimeErrors.
- **`total_16k_samples` must equal `f0_extractor_frame` exactly**: Not `f0_extractor_frame + block_16k_samples`. The rolling buffer length equals the RMVPE extraction window. Tested.
- **`block_48k` must be a multiple of `WINDOW * 3 = 480`** (16kHz→48kHz = ×3 ratio, times 160): 4800 ✓, 9600 ✓. Arbitrary sizes will produce non-integer `return_length_frames`.
- **JIT warmup on first `rvc.infer()` call**: ~1.9s for 100ms block, ~1.9s for 200ms block. Happens once per session startup. Pre-warm by calling `infer()` once with silence after `RVC.__init__` completes, before starting the sounddevice stream. This avoids the first real block being dropped.
- **MacBook Pro Microphone is 44100 Hz, not 48000 Hz**: sounddevice returns `default_samplerate=44100.0` for device 2. The rolling buffer must be built from resampled 16kHz data from 44100 Hz input. Use `scipy.signal.resample_poly(x, 16, 441)` (= 16000/44100 simplified: gcd(16000,44100)=400 → 40/110.25 → use 160/441).
- **`sounddevice.Stream` with different input/output devices**: Use `device=(input_id, output_id)` and `samplerate=48000` (BlackHole native rate). Input at 44100 requires resampling inside the callback or in a processing thread. Simpler: open `InputStream` and `OutputStream` separately with their native rates.
- **`os.killpg` for session stop**: The `stop()` call must cleanly stop the sounddevice stream — call `stream.stop()` and `stream.close()`. No subprocess involved; no process group kill needed.
- **WebSocket connection dropped on session end**: Send a `{type:"done"}` message before closing WS to signal browser to stop. Mirror training WS pattern exactly.
- **`torch.no_grad()` and MPS memory**: `rvc.infer()` already wraps in `torch.no_grad()`. No need to add it externally. MPS memory is not explicitly freed between blocks — for long sessions, watch for gradual slowdown (if observed, add `torch.mps.empty_cache()` every N blocks).

---

## Open Risks

- **`sounddevice` callback cannot call asyncio**: The callback fires in a C thread. Inter-thread communication to the asyncio loop must use thread-safe primitives (`asyncio.get_event_loop().call_soon_threadsafe()` or a `deque`). If not handled correctly, waveform data will never reach the WebSocket.
- **First-block latency spike**: JIT warmup on the first `rvc.infer()` call takes ~1.9s. If the sounddevice stream is started before warmup, the audio output will stall for ~1.9s. Pre-warm in session init before calling `stream.start()`.
- **FAISS index not present**: If the user starts a session before training completes, the index file doesn't exist. The `RVC.__init__` call would crash. Gate: check index file exists at session start; return 400 if not.
- **MPS memory leak during long sessions**: Long sessions (>30 min) on MPS may accumulate unreleased tensors. Not confirmed, but risk is real. Mitigate with `torch.mps.empty_cache()` periodically.
- **Input audio overflow**: `sounddevice` reports `overflow=True` in the callback status when audio is lost. Log overflows as structured events; don't crash.
- **44100 Hz input resampling quality**: `resample_poly` with ratio 160/441 may introduce small artifacts. Acceptable for voice conversion (the model itself introduces much more variation). Not a blocking risk.
- **`protect` parameter default**: `rvc.infer()` default `protect=1.0` disables protection (passes all features through unchanged). Recommended default: `0.33`. Expose as user-controllable param.

---

## API Contract (S03 → S04)

From the roadmap boundary:

```
POST /api/realtime/start
  body: {profile_id, input_device_id, output_device_id, pitch, index_rate, protect}
  → {session_id}

POST /api/realtime/stop
  body: {session_id}
  → {ok: true}

POST /api/realtime/params
  body: {session_id, pitch?, index_rate?, protect?}
  → {ok: true}  (hot-updates running session via rvc.set_key / rvc.set_index_rate)

GET /api/realtime/status
  → {active: bool, session_id, profile_id, input_device, output_device}

WS /ws/realtime/{session_id}
  → stream of {type: "waveform_in"|"waveform_out", samples: <base64 float32>, sr: 48000}
     at ~20fps until session ends
  → {type: "done"} on stop
```

**Hot param update**: `RVC` has `set_key(new_key)`, `set_formant(new_formant)`, `set_index_rate(new_index_rate)` methods — use these for live param changes without restarting the session.

**`AudioProcessor` pipeline (R006)**: Structure the audio path as a list of processor objects. Minimum implementation for S03:

```python
class AudioProcessor(Protocol):
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...

class RVCProcessor:
    """Wraps RVC.infer() as an AudioProcessor stage."""
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        ...
```

The pipeline list `[RVCProcessor()]` satisfies R006's extensibility requirement without over-engineering.

---

## Test Strategy

**Contract tests (no real audio hardware needed):**
- POST /api/realtime/start → 200 {session_id}
- POST /api/realtime/start → 404 unknown profile
- POST /api/realtime/start → 400 profile not trained
- POST /api/realtime/start → 400 no index file
- POST /api/realtime/start → 409 session already active
- POST /api/realtime/stop → 200 {ok: true}
- POST /api/realtime/stop → 404 unknown session
- POST /api/realtime/params → 200 {ok: true}
- GET /api/realtime/status → 200 (active/inactive states)
- WS /ws/realtime/{id} → streams waveform messages, terminates on done

Mock strategy: patch `backend.app.routers.realtime.manager` (router-level reference, same D023 pattern). Pre-populate a `deque` or `asyncio.Queue` with fake waveform messages.

**Integration test (marked `@pytest.mark.integration`):**
- Start a real session with MacBook Pro Mic → BlackHole 2ch
- Speak a few words (or play a WAV via `sounddevice.play()` into the session)
- Verify waveform messages arrive on WS within 500ms
- Stop session cleanly
- Verify stream.active = False

The integration test is gated on `RVC_ROOT` being set (same as training integration test).

---

## Implementation Plan (4 Tasks)

**T01 — Contract tests (10 stubs, all fail)**: Write `backend/tests/test_realtime.py` with 10 contract tests using patched manager. No implementation.

**T02 — `RealtimeSession` + audio loop** (`backend/app/realtime.py`): `RealtimeSession` dataclass, `RealtimeManager` singleton, audio processing thread with sounddevice, rolling buffer management, RVC inference in audio thread, waveform snapshot deque. No HTTP yet.

**T03 — REST + WebSocket router** (`backend/app/routers/realtime.py`): Wire up all 5 endpoints. Register routers in `main.py`. Run contract tests — all should pass.

**T04 — Integration test**: Write `backend/tests/test_realtime_integration.py`. Run real session. Confirm waveform messages arrive, audio plays on BlackHole.

---

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| sounddevice | none found | none found |
| FastAPI WebSocket | built-in FastAPI docs | installed (used in S02) |
| Web Worker / OffscreenCanvas | browser APIs | none found |

No additional skills to install.

---

## Sources

- `rtrvc.py` parameters validated by running `RVC.infer()` with systematic block sizes (source: empirical testing in rvc-web/Retrieval-based-Voice-Conversion-WebUI)
- RMVPE frame alignment: `f0_extractor_frame = 5120 * ceil((block+800-1)/5120) - 160` (source: rtrvc.py line 162–165)
- `synthesizers.py TextEncoder.forward` — skip_head/return_length are in FRAMES (source: rvc/layers/synthesizers.py + encoders.py inspection)
- `Config.use_insecure_load()` — fairseq + PyTorch 2.6 fix pattern (source: configs/config.py)
- sounddevice device list: [0] iPhone Mic, [1] BlackHole 2ch (48kHz), [2] MacBook Pro Mic (44.1kHz), [3] MacBook Pro Speakers, [4] MS Teams Audio (source: `sd.query_devices()` live output)
- Inference latency: 37ms steady-state for 200ms block on MPS (source: 8-iteration loop test)
