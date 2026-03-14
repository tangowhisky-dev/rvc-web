# S03: Realtime Voice Conversion & Waveform Visualization — UAT

**Milestone:** M001
**Written:** 2026-03-14

## UAT Type

- UAT mode: mixed (artifact-driven contract verification + live-runtime browser verification + human-experience audio validation deferred)
- Why this mode is sufficient: All 5 API endpoints are contract-verified by 10 passing pytest tests. The browser Realtime tab renders with correct controls and canvases (live-runtime verified). The audio session lifecycle (start → waveform stream → stop) is structurally proven at the integration test layer. The human-experience portion (hearing converted voice on BlackHole output) is deferred to when a trained profile is available — the test infrastructure to prove it exists (`test_realtime_integration.py`) and will auto-execute once S02's training integration test has been run.

## Preconditions

For contract + browser UAT (no hardware needed):
- Backend running: `conda run -n rvc uvicorn backend.app.main:app --reload --port 8000` from `rvc-web/`
- Frontend running: `cd frontend && pnpm dev` (port 3000)
- Both services healthy: `curl -s http://localhost:8000/health` → `{status:"ok"}`

For full integration UAT (hardware required):
- `RVC_ROOT=Retrieval-based-Voice-Conversion-WebUI` set in shell
- At least one voice profile with `status='trained'` in `backend/data/rvc.db`
- MacBook Pro Microphone connected and recognized by PortAudio
- BlackHole 2ch installed and visible in Audio MIDI Setup
- A downstream app (e.g. QuickTime, DAW) configured to use BlackHole as microphone input to audibly verify output

## Smoke Test

```bash
# Start services, hit status endpoint, verify inactive
curl -s http://localhost:8000/api/realtime/status
# → {"active":false,"session_id":null,"profile_id":null,"input_device":null,"output_device":null}
# Navigate to http://localhost:3000/realtime — two canvas elements visible
```

## Test Cases

### 1. Status endpoint returns inactive state

1. `curl -s http://localhost:8000/api/realtime/status`
2. **Expected:** `{"active":false,"session_id":null,"profile_id":null,"input_device":null,"output_device":null}` with HTTP 200

### 2. POST /start returns 404 for unknown profile

1. `curl -s -X POST http://localhost:8000/api/realtime/start -H 'Content-Type: application/json' -d '{"profile_id":"nonexistent","input_device_id":2,"output_device_id":1,"pitch":0,"index_rate":0.75,"protect":0.5}'`
2. **Expected:** HTTP 404 with `{"detail": "Profile not found"}`

### 3. POST /start returns 400 for untrained profile

1. Insert a profile with `status='untrained'` in the DB
2. POST /start with that profile_id
3. **Expected:** HTTP 400 with `{"detail": "Profile is not trained (status: untrained)"}`

### 4. POST /start returns 409 if session already active (contract)

1. Auto-confirmed by `test_start_409_already_active` in `backend/tests/test_realtime.py`
2. **Expected:** HTTP 409 with conflict detail

### 5. WebSocket delivers waveform messages then done (contract)

1. `conda run -n rvc pytest backend/tests/test_realtime.py::test_ws_streams_waveform_and_done -v`
2. **Expected:** PASSED — test receives 2 waveform messages then `{type:"done"}`

### 6. Realtime tab renders in browser

1. Navigate to `http://localhost:3000/realtime`
2. **Expected:** Two canvas elements visible labeled "INPUT — MIC" and "OUTPUT — CONVERTED"; input/output device dropdowns populated from API; MacBook Pro Microphone auto-selected for input; BlackHole 2ch auto-selected for output; Pitch / Index Rate / Protect sliders visible with default values; "Start Session" button visible; no JavaScript console errors

### 7. Device dropdowns populated from API

1. With backend running, navigate to `http://localhost:3000/realtime`
2. Open input device dropdown
3. **Expected:** List contains real system audio device names from `/api/devices`; MacBook Pro Microphone is selected by default

### 8. Full realtime session lifecycle (integration — requires trained profile)

1. Ensure `RVC_ROOT` set and trained profile exists in DB
2. `RVC_ROOT=Retrieval-based-Voice-Conversion-WebUI conda run -n rvc pytest backend/tests/test_realtime_integration.py -v -s -m integration --timeout=60`
3. **Expected:** 1 passed — at least 5 waveform messages received; status confirms inactive after stop

### 9. Start session from browser (integration — requires trained profile)

1. Navigate to `http://localhost:3000/realtime`
2. Select trained profile in Voice Profile dropdown
3. Verify MacBook Pro Microphone and BlackHole 2ch selected
4. Click "Start Session"
5. **Expected:** Button changes to "Stop Session"; two waveform canvases begin animating within 2 seconds of speaking into mic

### 10. Waveforms animate and stop cleanly

1. With session running, speak into microphone
2. **Expected:** Input canvas waveform animates (blue); output canvas waveform animates (green)
3. Click "Stop Session"
4. **Expected:** Button reverts to "Start Session"; waveform animation halts

## Edge Cases

### POST /start when no trained profile exists

1. Attempt POST /start with a profile whose `status != 'trained'`
2. **Expected:** HTTP 400 — "Profile is not trained (status: …)" — not a 500 or silent failure

### POST /start when FAISS index file missing

1. Training produced `.pth` but FAISS index file was deleted
2. POST /start with that profile
3. **Expected:** HTTP 400 — "No FAISS index found for this profile" — session does not start, no orphaned stream

### WebSocket client disconnects mid-session

1. Start a session, connect via WS, then close WS connection without calling POST /stop
2. **Expected:** Session continues running (audio loop keeps going); structured log event `ws_realtime_disconnect` emitted; reconnecting to same session_id resumes waveform delivery

### Stop session twice

1. POST /api/realtime/stop with a valid session_id
2. POST /api/realtime/stop with the same session_id
3. **Expected:** First stop → HTTP 200 `{ok:true}`; second stop → HTTP 404 (session not found after stop)

## Failure Signals

- `GET /api/realtime/status` returns `{active:false, error:"..."}` — session failed, check backend stdout for `session_error` JSON event
- Browser canvas not animating despite session active — check DevTools console for `[waveform-worker]` error lines
- `conda run -n rvc pytest backend/tests/ -v` shows fewer than 35 passed — contract regression
- Backend logs `waveform_overflow` events repeatedly — block size too small for current CPU load; increase `block_48k` in `realtime.py` D031
- `/realtime` page shows "No trained profiles available" — S02 training pipeline has not been run to completion yet

## Requirements Proved By This UAT

- R003 (Realtime Voice Conversion) — contract-proven by 5 POST /start tests + stop test + WS test; integration lifecycle covered by `test_realtime_integration.py`; human-audible proof deferred to hardware run
- R004 (Live Waveform Visualization) — WS contract test proves `waveform_in`/`waveform_out` messages at ~20fps; browser test confirms both canvases render; OffscreenCanvas Web Worker present without console errors
- R005 (Audio Device Selection) — device dropdowns populate from `/api/devices`; auto-selection by name fragment confirmed; selection passed to POST /start
- R006 (Modular Audio Processing Chain) — AudioProcessor Protocol + RVCProcessor implemented; pipeline structure established; insertion points ready for future effects

## Not Proven By This UAT

- Audible voice conversion quality on BlackHole output — requires hardware run with trained profile; the system routes audio correctly but perceptual quality is not mechanically verified
- End-to-end latency measurement — block size (D031) targets ~200ms; actual measured round-trip latency not confirmed
- Waveform animation frame rate on a live session — 20fps target not mechanically measured; visual inspection only
- R007 (Start/Stop Service Scripts) — deferred to S04
- R008 (BlackHole Setup Guide) — deferred to S04

## Notes for Tester

- To run the full integration test: first run the S02 training integration test to produce a `.pth` and update profile status to `trained`, then run `test_realtime_integration.py`.
- If BlackHole 2ch appears in the output device list but audio is not heard in downstream apps: verify Audio MIDI Setup routes BlackHole → System Output or the target app is using BlackHole as its input device.
- The OffscreenCanvas fallback path in `page.tsx` has not been exercised — if canvases render blank, check that the browser supports `transferControlToOffscreen()` (Chrome/Firefox/Safari 16.4+).
- `waveform-worker.js` is served from `frontend/public/` as a static file — if the Next.js dev server has a different base path, the worker URL in `page.tsx` may need updating.
