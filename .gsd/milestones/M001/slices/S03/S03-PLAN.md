# S03: Realtime Voice Conversion & Waveform Visualization

**Goal:** A running FastAPI backend endpoint that starts a realtime RVC audio loop reading from the MacBook Pro Microphone and writing converted audio to BlackHole 2ch, while streaming input/output waveform data over WebSocket at ~20fps; all five API endpoints verified by passing contract tests; a minimal Realtime tab in the Next.js frontend renders live waveform canvases via a Web Worker.

**Demo:** User opens the browser Realtime tab, selects MacBook Pro Microphone and BlackHole 2ch, clicks Start — their voice is converted through the trained RVC model and plays on BlackHole output; two waveform canvases animate live in the browser. Clicking Stop halts the session cleanly.

## Must-Haves

- `backend/app/realtime.py` — `RealtimeSession` dataclass + `RealtimeManager` singleton with audio loop thread, rolling buffer, RVC inference, and waveform snapshot deque
- `backend/app/routers/realtime.py` — five endpoints: POST /api/realtime/start, POST /api/realtime/stop, POST /api/realtime/params, GET /api/realtime/status, WS /ws/realtime/{session_id}
- `AudioProcessor` protocol + `RVCProcessor` implementation (satisfies R006)
- Fairseq safe-globals fix (`Config.use_insecure_load()`) applied before `RVC.__init__`
- `rmvpe_root` env var set in session init before rtrvc.py import
- Block parameter formula from research applied exactly (skip_head/return_length in FRAMES)
- 44100 Hz mic input resampled via `scipy.signal.resample_poly` before HuBERT
- JIT warmup call (silence block) before `stream.start()` to avoid first-block latency spike
- FAISS index path discovered by glob `added_IVF*_Flat*.index`; 400 if not found
- Profile `status == "trained"` gate on session start
- `WS /ws/realtime/{session_id}` sends `{type:"waveform_in"|"waveform_out", samples:<base64 float32>, sr:48000}` at ~20fps; sends `{type:"done"}` on stop
- `backend/tests/test_realtime.py` — 10 contract tests, all passing
- `frontend/app/realtime/page.tsx` — minimal Realtime tab with device selector, Start/Stop button, two waveform canvases
- `frontend/public/waveform-worker.js` — Web Worker rendering waveforms via OffscreenCanvas
- Both routers registered in `backend/app/main.py`

## Proof Level

- This slice proves: contract (10 pytest tests) + integration (real audio session on hardware)
- Real runtime required: yes (integration test requires MacBook Pro Mic + BlackHole 2ch)
- Human/UAT required: yes (must audibly confirm converted voice on BlackHole output)

## Verification

```bash
# Contract tests — all 10 green
conda run -n rvc pytest backend/tests/test_realtime.py -v
# Expected: 10 passed

# Full suite — no regressions
conda run -n rvc pytest backend/tests/ -v
# Expected: 35 passed, 1 skipped

# Integration smoke test (requires RVC_ROOT + audio hardware)
RVC_ROOT=Retrieval-based-Voice-Conversion-WebUI conda run -n rvc \
  pytest backend/tests/test_realtime_integration.py -v -s -m integration --timeout=60
# Expected: 1 passed — waveform messages received, stream stops cleanly

# Status endpoint (running server)
curl -s http://localhost:8000/api/realtime/status
# → {"active":false,"session_id":null,"profile_id":null,"input_device":null,"output_device":null}

# Frontend waveform tab
# Navigate to http://localhost:3000/realtime — two canvas elements visible
```

## Observability / Diagnostics

- Runtime signals: structured JSON stdout events (`session_start`, `session_stop`, `session_error`, `waveform_overflow`); `stream.active` bool polled by status endpoint
- Inspection surfaces:
  - `GET /api/realtime/status` → `{active, session_id, profile_id, input_device, output_device}` — live session state at any time
  - `sqlite3 backend/data/rvc.db "SELECT id,name,status FROM profiles WHERE status='trained';"` — confirms trained model available
  - `ls Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/added_IVF*_Flat*.index` — confirms FAISS index present
- Failure visibility: session error stored on `RealtimeSession.error`; `GET /api/realtime/status` exposes `{active:false, error:"..."}` after a failed session; structured JSON log includes exception message and phase
- Redaction constraints: none (no secrets or PII in audio metadata)

## Integration Closure

- Upstream surfaces consumed:
  - `backend/app/db.py` — `get_db()`, profile row with `status` and `id` fields (from S01)
  - `backend/app/routers/devices.py` — device IDs used by start request (from S01)
  - `Retrieval-based-Voice-Conversion-WebUI/infer/lib/rtrvc.py` — `RVC` class (RVC_ROOT dep, from S02)
  - `Retrieval-based-Voice-Conversion-WebUI/configs/config.py` — `Config.use_insecure_load()` (fairseq fix pattern from S02/D025)
  - `assets/weights/rvc_finetune_active.pth` and `logs/rvc_finetune_active/added_IVF*_Flat*.index` (produced by S02)
  - `backend/app/main.py` — router registration (pattern from S01/S02)
- New wiring introduced in this slice:
  - `RealtimeManager` singleton imported in `routers/realtime.py`, registered in `main.py`
  - `sounddevice.InputStream` / `OutputStream` opened at session start, closed at stop
  - WebSocket waveform pusher asyncio task dispatching from deque populated by audio thread
  - Next.js `/realtime` page connecting to `WS /ws/realtime/{session_id}`
  - Web Worker receiving waveform messages and drawing to OffscreenCanvas
- What remains before the milestone is truly usable end-to-end:
  - S04 polish: full UI with profile selector, all tabs wired, `start.sh` / `stop.sh` scripts, BlackHole setup guide

## Tasks

- [x] **T01: Write 10 contract tests for realtime API (all initially failing)** `est:45m`
  - Why: Contract-first — defines the exact shape every endpoint must deliver before any implementation exists; creates the objective stopping condition for T03
  - Files: `backend/tests/test_realtime.py`
  - Do: Write `test_realtime.py` with 10 tests covering all five endpoints. Use `sync_client` fixture (starlette TestClient) for WS tests and `client` fixture (httpx AsyncClient) for REST. Mock strategy: `patch('backend.app.routers.realtime.manager')` at router module scope (mirror `mock_manager` pattern from test_training.py). Tests: (1) POST start → 200 {session_id}; (2) POST start → 404 unknown profile; (3) POST start → 400 profile not trained; (4) POST start → 400 no index file; (5) POST start → 409 session already active; (6) POST stop → 200 {ok:true}; (7) POST stop → 404 unknown session; (8) POST params → 200 {ok:true}; (9) GET status → 200 with active/inactive states; (10) WS connect → receives waveform_in/waveform_out messages, terminates on {type:"done"}. All 10 must fail (ImportError or 404) until T02–T03 exist.
  - Verify: `conda run -n rvc pytest backend/tests/test_realtime.py -v` — confirms 10 tests collected, all fail (not error on syntax)
  - Done when: All 10 tests are syntactically valid and collected by pytest; all fail for implementation-missing reasons (not fixture or syntax errors)

- [x] **T02: Implement RealtimeSession, RealtimeManager, and audio loop** `est:2h`
  - Why: Core audio engine — the `RealtimeSession` owns the sounddevice streams, rolling buffer, RVC inference thread, and waveform deque that all other tasks depend on
  - Files: `backend/app/realtime.py`
  - Do: Create `backend/app/realtime.py`. Define `AudioProcessor` Protocol and `RVCProcessor` class (satisfies R006). Define `RealtimeSession` dataclass mirroring `TrainingJob`: fields `session_id`, `profile_id`, `input_device`, `output_device`, `status` ("starting"|"active"|"stopped"|"error"), `error`, `_stream_in` (sd.InputStream), `_stream_out` (sd.OutputStream), `_waveform_deque` (collections.deque, maxlen=40), `_stop_event` (threading.Event), `_audio_thread` (threading.Thread). Implement `RealtimeManager` singleton with `start_session(profile_id, input_device_id, output_device_id, pitch, index_rate, protect, rvc_root) → RealtimeSession` and `stop_session(session_id)`, `get_session(session_id)`, `active_session() → Optional[RealtimeSession]`. In `start_session`: (a) validate profile status==trained via `get_db()`; (b) glob for FAISS index; (c) set `rmvpe_root` env var and `sys.path.insert(0, rvc_root)`; (d) call `Config.use_insecure_load()`; (e) instantiate `RVC` (this triggers HuBERT + synthesizer load); (f) compute block params per research formula (block_48k=9600); (g) run one silence warmup call; (h) open separate `sd.InputStream(device=input_id, samplerate=44100, channels=1, dtype='float32')` and `sd.OutputStream(device=output_id, samplerate=48000, channels=1, dtype='float32')`; (i) start `_audio_thread` running `_audio_loop(session)`; (j) start the streams. `_audio_loop` reads from InputStream with `stream_in.read(block_44k)`, resamples 44100→16000 with `resample_poly`, appends to rolling buffer, calls `rvc.infer(...)` with correct FRAME params, appends result to `_waveform_deque` as `{type, samples (base64 subsampled), sr}` tuples for both input and output, writes output to OutputStream. Log `waveform_overflow` if sd overflow flag is True. `stop_session`: sets `_stop_event`, calls `stream_in.stop()/close()`, `stream_out.stop()/close()`, joins thread.
  - Verify: `python -c "from backend.app.realtime import RealtimeManager, AudioProcessor, RVCProcessor; print('imports ok')"` (without RVC_ROOT set — should import cleanly since RVC import is lazy inside start_session). No pytest tests pass yet (router not wired).
  - Done when: Module imports without error; `RealtimeManager`, `RealtimeSession`, `AudioProcessor`, `RVCProcessor` all defined; `manager` singleton exported at module level

- [x] **T03: Implement REST + WebSocket router and wire into main.py** `est:45m`
  - Why: Exposes the audio engine over HTTP and WebSocket; registers routes in `main.py`; makes all 10 contract tests pass
  - Files: `backend/app/routers/realtime.py`, `backend/app/main.py`
  - Do: Create `backend/app/routers/realtime.py`. Follow D023 two-router pattern: `router = APIRouter(prefix='/api/realtime')` for REST, `ws_router = APIRouter()` for WebSocket. Implement: POST `/start` → validate inputs, call `manager.start_session(...)`, return `{session_id}`; return 404/400/409 as specified in tests. POST `/stop` → find session, call `manager.stop_session(...)`, return `{ok:true}`; 404 if not found. POST `/params` → find session, call `rvc.set_key()` / `rvc.set_index_rate()` with provided values, return `{ok:true}`. GET `/status` → return `{active, session_id, profile_id, input_device, output_device}`. `WS /ws/realtime/{session_id}` on `ws_router`: accept connection, poll for session (up to 30s), then loop consuming `session._waveform_deque` (drain in asyncio task using `asyncio.get_event_loop().run_in_executor` bridge or `asyncio.sleep(0.05)` poll) sending each `{type, samples, sr}` message as JSON; when `_stop_event` is set, send `{type:"done"}` and close. Register both routers in `backend/app/main.py`: `app.include_router(realtime_router)` + `app.include_router(realtime_ws_router)`.
  - Verify: `conda run -n rvc pytest backend/tests/test_realtime.py -v` → 10 passed; `conda run -n rvc pytest backend/tests/ -v` → 35 passed, 1 skipped
  - Done when: All 10 contract tests green; full suite 35 passed

- [x] **T04: Write and run integration test; build minimal Realtime frontend tab** `est:1.5h`
  - Why: Proves the real audio loop works on hardware (R003, R005); provides the browser UI surface (R004, R005); retires rtrvc.py block size and fairseq import environment risks
  - Files: `backend/tests/test_realtime_integration.py`, `frontend/app/realtime/page.tsx`, `frontend/public/waveform-worker.js`
  - Do: **Integration test** — write `backend/tests/test_realtime_integration.py` with one `@pytest.mark.integration` test: (a) skip if `RVC_ROOT` not set or profile not trained (check SQLite); (b) call `POST /api/realtime/start` with MacBook Pro Mic (id=2) → BlackHole 2ch (id=1) and trained profile_id; (c) connect to `WS /ws/realtime/{session_id}`; (d) wait up to 10s to receive at least 5 waveform messages; (e) call `POST /api/realtime/stop`; (f) verify next WS message is `{type:"done"}`; (g) assert `GET /api/realtime/status` returns `{active:false}`. Mark test `@pytest.mark.integration` and add skip guard. **Frontend tab** — create `frontend/app/realtime/page.tsx`: React component with `useEffect` to fetch `/api/devices`, two `<select>` dropdowns (input/output devices), pitch/index_rate/protect sliders, Start/Stop button, two `<canvas>` elements (input waveform, output waveform). On Start: POST `/api/realtime/start`, then open `new WebSocket('ws://localhost:8000/ws/realtime/{session_id}')`, transfer canvases to `waveform-worker.js` via `transferControlToOffscreen`. On Stop: POST `/api/realtime/stop`. **Web Worker** — create `frontend/public/waveform-worker.js`: receives `{type, samples, sr}` messages; decodes base64 float32 array; draws waveform to the appropriate OffscreenCanvas; clears and redraws on each message.
  - Verify: Integration: `RVC_ROOT=Retrieval-based-Voice-Conversion-WebUI conda run -n rvc pytest backend/tests/test_realtime_integration.py -v -s -m integration --timeout=60` → 1 passed. Frontend: start server + frontend, navigate to `http://localhost:3000/realtime`, confirm two canvases visible, Start/Stop button present, device dropdowns populated.
  - Done when: Integration test passes (waveform messages received from real hardware session); frontend page renders at `/realtime` with canvases and controls; Web Worker file exists and is loaded without console errors

## Files Likely Touched

- `backend/app/realtime.py` — NEW: RealtimeSession, RealtimeManager, AudioProcessor, RVCProcessor
- `backend/app/routers/realtime.py` — NEW: REST + WebSocket realtime router
- `backend/app/main.py` — updated: register realtime_router + realtime_ws_router
- `backend/tests/test_realtime.py` — NEW: 10 contract tests
- `backend/tests/test_realtime_integration.py` — NEW: real hardware integration test
- `frontend/app/realtime/page.tsx` — NEW: Realtime tab page component
- `frontend/public/waveform-worker.js` — NEW: Web Worker for OffscreenCanvas waveform rendering
