---
estimated_steps: 7
estimated_files: 4
---

# T04: Write and run integration test; build minimal Realtime frontend tab

**Slice:** S03 — Realtime Voice Conversion & Waveform Visualization
**Milestone:** M001

## Description

Two parallel deliverables in one task:

1. **Integration test** — proves the real audio pipeline works on hardware (MacBook Pro Mic → RVC inference → BlackHole 2ch output); retires the rtrvc.py block size risk and fairseq import risk from the milestone.

2. **Minimal Realtime frontend tab** — creates `frontend/app/realtime/page.tsx` with device dropdowns, Start/Stop controls, and two waveform canvases; creates `frontend/public/waveform-worker.js` as the OffscreenCanvas Web Worker. This is the browser-visible proof of R004 (live waveform visualization) and R005 (device selection).

The integration test must pass. The frontend must render at `/realtime` with canvases that animate when a session is active.

## Steps

1. **Write `backend/tests/test_realtime_integration.py`**:
   - Module-level skip guard: `pytestmark = pytest.mark.integration`
   - `@pytest.fixture(autouse=True)` that calls `pytest.skip("RVC_ROOT not set")` if `os.environ.get("RVC_ROOT", "") == ""`
   - Additional skip guard: query SQLite for a profile with `status='trained'`; skip if none found
   - One test `test_realtime_session_produces_waveforms`:
     - Use `httpx.AsyncClient` (async test, decorated `@pytest.mark.anyio`)
     - Look up trained profile_id from DB
     - Query `GET /api/devices` to confirm BlackHole (id=1) and MacBook Pro Mic (id=2) present
     - POST `/api/realtime/start` with `{profile_id, input_device_id=2, output_device_id=1, pitch=0, index_rate=0.75, protect=0.33}` → assert 200, capture `session_id`
     - Connect to `WS /ws/realtime/{session_id}` using `starlette.testclient.TestClient`
     - Collect messages for up to 10 seconds; assert at least 5 waveform messages received (type in `waveform_in`, `waveform_out`)
     - POST `/api/realtime/stop` with `{session_id}` → assert 200 `{ok: true}`
     - Assert next WS message has `type == "done"` (or WS closes)
     - GET `/api/realtime/status` → assert `{active: false}`
   - Add to `pytest.ini` under `markers` if not already present (mirrors D026): `integration: marks tests as requiring real hardware`

2. **Run integration test** (requires audio hardware):
   ```bash
   RVC_ROOT=Retrieval-based-Voice-Conversion-WebUI conda run -n rvc \
     pytest backend/tests/test_realtime_integration.py -v -s -m integration --timeout=60
   ```
   Iterate on any failures (block param errors, import errors, device errors) until 1 passed.

3. **Check frontend project structure** — confirm `frontend/app/` exists (Next.js App Router); check if `frontend/app/realtime/` directory needs to be created.

4. **Create `frontend/app/realtime/page.tsx`** — React client component:
   ```tsx
   'use client';
   ```
   - `useEffect` to fetch `GET http://localhost:8000/api/devices` on mount; populate `inputDevices` and `outputDevices` state arrays
   - Two `<select>` elements for input/output device selection (default to BlackHole for output if found, MacBook Pro Mic for input)
   - Three `<input type="range">` sliders: pitch (-12 to +12, step 1, default 0), index_rate (0–1, step 0.05, default 0.75), protect (0–1, step 0.05, default 0.33)
   - `<button>` that toggles Start / Stop
   - Two `<canvas ref>` elements (one for input waveform, one for output waveform), each labeled "Input" and "Output"
   - `useRef` for two canvas elements and the `sessionId` state
   - On Start:
     - POST `/api/realtime/start` → capture `session_id`
     - Transfer both canvases to offscreen: `canvas.transferControlToOffscreen()`
     - Create `new Worker('/waveform-worker.js')`; `worker.postMessage({type:'init', canvases:[offscreenIn, offscreenOut]}, [offscreenIn, offscreenOut])`
     - Open `new WebSocket('ws://localhost:8000/ws/realtime/' + session_id)`
     - On each WS message: parse JSON; forward to worker: `worker.postMessage(data)`
     - On `{type:'done'}`: close WS; set session stopped state
   - On Stop:
     - POST `/api/realtime/stop` with `{session_id}`
     - Close WS and worker
   - Params sliders: on change while active, POST `/api/realtime/params` with `{session_id, pitch, index_rate, protect}`

5. **Create `frontend/public/waveform-worker.js`** — plain JS (not TypeScript; must be in `/public/` for browser direct fetch):
   ```js
   let canvasIn = null;
   let canvasOut = null;
   let ctxIn = null;
   let ctxOut = null;

   self.onmessage = function(e) {
     const msg = e.data;
     if (msg.type === 'init') {
       canvasIn = msg.canvases[0];
       canvasOut = msg.canvases[1];
       ctxIn = canvasIn.getContext('2d');
       ctxOut = canvasOut.getContext('2d');
       return;
     }
     if (msg.type === 'waveform_in' || msg.type === 'waveform_out') {
       const ctx = msg.type === 'waveform_in' ? ctxIn : ctxOut;
       const canvas = msg.type === 'waveform_in' ? canvasIn : canvasOut;
       if (!ctx) return;
       // Decode base64 float32
       const binary = atob(msg.samples);
       const bytes = new Uint8Array(binary.length);
       for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
       const samples = new Float32Array(bytes.buffer);
       // Draw waveform
       ctx.clearRect(0, 0, canvas.width, canvas.height);
       ctx.beginPath();
       ctx.strokeStyle = msg.type === 'waveform_in' ? '#3b82f6' : '#10b981';
       ctx.lineWidth = 1.5;
       const step = canvas.width / samples.length;
       const mid = canvas.height / 2;
       for (let i = 0; i < samples.length; i++) {
         const x = i * step;
         const y = mid + samples[i] * mid * 0.9;
         i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
       }
       ctx.stroke();
     }
   };
   ```

6. **Verify frontend builds and renders**:
   - Start frontend: `cd frontend && pnpm dev` (or `npm run dev`)
   - Navigate to `http://localhost:3000/realtime`
   - Confirm: page loads without console errors; device dropdowns populated; two canvas elements visible; Start button clickable

7. **Run full backend suite one final time** to confirm no regressions from integration test file addition.

## Must-Haves

- [ ] Integration test skips cleanly when `RVC_ROOT` not set (must not error)
- [ ] Integration test skips cleanly when no trained profile exists in DB
- [ ] Integration test passes: POST start → WS receives ≥5 waveform messages → POST stop → `{type:"done"}` received → status `{active:false}`
- [ ] `frontend/app/realtime/page.tsx` renders at `/realtime` without console errors
- [ ] `frontend/public/waveform-worker.js` exists and is syntactically valid
- [ ] Two waveform canvases present in the DOM
- [ ] Device dropdowns populated from `/api/devices` (requires backend running)
- [ ] Web Worker receives init message with two OffscreenCanvas objects
- [ ] Full backend suite: 35 passed, 1 skipped (no regressions)

## Verification

```bash
# Integration test
RVC_ROOT=Retrieval-based-Voice-Conversion-WebUI conda run -n rvc \
  pytest backend/tests/test_realtime_integration.py -v -s -m integration --timeout=60
# Expected: 1 passed

# Full backend suite — no regressions
conda run -n rvc pytest backend/tests/ -v
# Expected: 35 passed, 1 skipped

# Frontend
cd /Users/tango16/code/rvc-web/frontend && pnpm dev &
# Then open browser to http://localhost:3000/realtime
# Verify: page loads, canvases visible, device dropdowns present, no JS errors
```

## Observability Impact

- Signals added/changed:
  - Integration test log output shows waveform message count and timing — agent-readable proof of block size adequacy
  - `GET /api/realtime/status → {active:false}` after stop is the machine-verifiable session cleanup signal
  - Browser console errors from waveform-worker.js surface immediately in DevTools (no silent failures)
- How a future agent inspects this:
  - `conda run -n rvc pytest backend/tests/test_realtime_integration.py -v -s -m integration` — re-run proof of real audio session
  - `curl -s http://localhost:8000/api/realtime/status` — confirm no stuck active session
  - Browser `/realtime` page — visual confirmation waveforms are animating
- Failure state exposed:
  - Integration test failure output names the phase (start/waveform/stop) that failed
  - `session.error` string from audio thread crash visible via `GET /api/realtime/status` after failed session (T02+T03 wiring)
  - WS `{type:"error", message:"..."}` sent to browser on audio thread exception

## Inputs

- `backend/app/routers/realtime.py` (T03 output) — all five endpoints working; contract tests green
- `backend/tests/conftest.py` — `client` async fixture; `anyio_backend` session fixture
- `pytest.ini` — `integration` mark already registered (D026); `anyio` backend configured
- `sqlite3 backend/data/rvc.db` — must contain a profile with `status='trained'` (produced by S02 integration run)
- `frontend/` — existing Next.js project structure; confirm `frontend/app/` exists before creating subdirectory
- `S03-RESEARCH.md` → Audio loop data flow diagram — confirms device IDs 1 and 2 for test
- `S03-PLAN.md` → Verification section — exact curl commands for final check

## Expected Output

- `backend/tests/test_realtime_integration.py` — integration test with skip guards; 1 passed on real hardware
- `frontend/app/realtime/page.tsx` — Realtime tab: device selectors, sliders, Start/Stop button, two labeled waveform canvases
- `frontend/public/waveform-worker.js` — Web Worker: base64 decode + OffscreenCanvas waveform draw
- Backend suite result: **35 passed, 1 skipped** (integration auto-skipped without RVC_ROOT)
- Frontend: page renders at `/realtime` with canvases and controls, no console errors
- Risk retirement: rtrvc.py block size risk retired (200ms block confirmed adequate in integration run); fairseq import environment risk retired (rtrvc.py imports and RVC.__init__ completes without error)
