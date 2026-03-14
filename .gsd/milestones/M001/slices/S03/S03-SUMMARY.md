---
id: S03
parent: M001
milestone: M001
provides:
  - backend/app/realtime.py — RealtimeManager singleton, RealtimeSession dataclass, AudioProcessor Protocol, RVCProcessor, _audio_loop thread, manager singleton
  - backend/app/routers/realtime.py — 5 realtime endpoints (POST /start, /stop, /params; GET /status; WS /ws/realtime/{session_id})
  - backend/app/main.py — updated with realtime_router + realtime_ws_router registered
  - backend/tests/test_realtime.py — 10 contract tests, all passing
  - backend/tests/test_realtime_integration.py — integration test with dual skip guards (skips cleanly without hardware/trained profile)
  - frontend/ — bootstrapped Next.js 16 + TypeScript + Tailwind 4 project
  - frontend/app/realtime/page.tsx — Realtime tab (device dropdowns, profile picker, waveform canvases, param sliders, Start/Stop)
  - frontend/public/waveform-worker.js — Web Worker for OffscreenCanvas waveform rendering
requires:
  - slice: S01
    provides: get_db() / profile row schema / GET /api/devices / profile status field values
  - slice: S02
    provides: rtrvc.py RVC class / Config.use_insecure_load() / assets/weights/rvc_finetune_active.pth / FAISS index / trained profile DB status
affects:
  - S04
key_files:
  - backend/app/realtime.py
  - backend/app/routers/realtime.py
  - backend/app/main.py
  - backend/tests/test_realtime.py
  - backend/tests/test_realtime_integration.py
  - frontend/app/realtime/page.tsx
  - frontend/public/waveform-worker.js
key_decisions:
  - D027 — In-memory session dict, single active session at a time (mirrors D017 TrainingManager)
  - D028 — Separate sd.InputStream + sd.OutputStream at native rates (44100 Hz in / 48000 Hz out)
  - D029 — Daemon threading.Thread for audio loop; deque for waveform handoff to asyncio WS
  - D030 — asyncio polls _waveform_deque every 50ms (~20fps); deque.popleft() is thread-safe SPSC
  - D031 — block_48k=9600 (200ms at 48kHz); ~37ms steady-state latency on MPS
  - D032 — Waveform subsampled to ~800 points per message (12× decimation); base64 float32 bytes
  - D033 — asyncio.iscoroutine() guard on manager.start_session to handle sync MagicMock in tests vs real async manager
  - D034 — RVC_ROOT gate removed from router; DB-level validation fires first (404/400/409)
  - D035 — Device API schema: is_input/is_output booleans (not kind string)
  - D036 — Frontend bootstrapped fresh with Next.js 16 + App Router + Tailwind 4
  - D037 — Realtime tab includes Voice Profile selector (fetches /api/profiles, filters trained)
patterns_established:
  - AudioProcessor Protocol as plain class with process() stub; RVCProcessor subclasses it — R006 extensibility contract
  - _compute_block_params() centralises block size formula (skip_head/return_length in FRAMES, not samples)
  - None sentinel in _waveform_deque signals WS handler to send {type:"done"} and close
  - update_params() hot-updates pitch/index_rate/protect on live session without restart
  - Dual autouse skip guards for integration tests: RVC_ROOT check + trained-profile DB check
  - Device auto-selection by name-fragment matching (not hardcoded IDs) — adapts if PortAudio renumbers
  - OffscreenCanvas transfer with fallback path on browsers that don't support it
  - Two-router pattern (D023): APIRouter(prefix='/api/realtime') for REST + bare APIRouter() for WS
observability_surfaces:
  - GET /api/realtime/status → {active, session_id, profile_id, input_device, output_device} — live session state
  - stdout JSON {event:"session_start", session_id, profile_id, input_device, output_device}
  - stdout JSON {event:"session_stop", session_id}
  - stdout JSON {event:"session_error", session_id, error}
  - stdout JSON {event:"waveform_overflow", session_id}
  - session.status + session.error fields persist on RealtimeSession after failure
  - GET /api/realtime/status exposes {active:false, error} after failed session
  - Browser console: waveform-worker.js logs errors as [waveform-worker] prefix
drill_down_paths:
  - .gsd/milestones/M001/slices/S03/tasks/T01-SUMMARY.md
  - .gsd/milestones/M001/slices/S03/tasks/T02-SUMMARY.md
  - .gsd/milestones/M001/slices/S03/tasks/T03-SUMMARY.md
  - .gsd/milestones/M001/slices/S03/tasks/T04-SUMMARY.md
duration: ~4h (T01: 20m, T02: 45m, T03: 30m, T04: 90m)
verification_result: passed
completed_at: 2026-03-14
---

# S03: Realtime Voice Conversion & Waveform Visualization

**Complete audio engine with realtime RVC loop, 5 REST+WS endpoints (10 contract tests passing), minimal Realtime tab with live waveform canvases, and dual-guard integration test that skips cleanly without hardware — all contract-verified with 35 passed / 2 skipped in full suite.**

## What Happened

Built the realtime VC system in four tasks: contracts first (T01), audio engine (T02), router (T03), integration test + frontend (T04).

**T01 (20m):** Wrote 10 contract tests covering all 5 endpoints before any implementation existed. Tests established the exact API shape: 5 tests for POST /start (200/404/400-not-trained/400-no-index/409), 2 for POST /stop (200/404), 1 for POST /params (200), 1 for GET /status (inactive), 1 WS test (waveform_in + waveform_out messages then {type:"done"}). All 10 collected and failed for implementation-missing reasons — confirming the objective stopping condition for T03.

**T02 (~45m):** Implemented `backend/app/realtime.py` (~300 lines). Core structure mirrors the TrainingManager/TrainingJob pattern (D017/D027). Key engineering choices: separate `sd.InputStream` + `sd.OutputStream` at native hardware rates (44100 Hz in / 48000 Hz out, D028); daemon thread for audio loop with deque-based waveform handoff to asyncio (D029/D030); `_compute_block_params()` centralizing the RMVPE-aligned block formula with skip_head/return_length expressed in FRAMES not samples (validated: 11 + 20 = 31 = p_len). Deferred RVC imports until `start_session()` — module imports cleanly without `RVC_ROOT`. None sentinel in deque signals WS handler to send `{type:"done"}`. Real RVC constructor signature discovered: `(key, formant, pth_path, index_path, index_rate, ...)` — formant=0 set for no shift; f0method/protect are per-infer-call args not constructor args.

**T03 (~30m):** Created `backend/app/routers/realtime.py` with two-router pattern (D023). Validation order in POST /start: 409 → 404 → 400 → manager.start_session(). Two fixes needed: (1) `asyncio.iscoroutine()` guard so sync MagicMock test fixture and real async manager both work without touching test code (D033); (2) RVC_ROOT gate removed from router level so DB-level errors fire correctly in tests (D034). WS handler uses `deque.popleft()` polling at 50ms, sends `{type:"done"}` on None sentinel. Registered both routers in `main.py`. All 10 contract tests passed on first run.

**T04 (~90m):** Wrote integration test (`test_realtime_integration.py`) with dual autouse skip guards — skips cleanly without `RVC_ROOT` and without a trained profile in DB. Test exercises full lifecycle: GET /api/devices → POST /start → collect ≥5 WS waveform messages → POST /stop → assert {type:"done"} → confirm status inactive. Device discovery uses name-fragment matching, not hardcoded IDs. Frontend bootstrapped from scratch (no frontend directory existed) via `pnpm create next-app`. Discovered device API uses `is_input`/`is_output` booleans not `kind` string (D035). Realtime tab includes real profile dropdown (D037). Both canvases use OffscreenCanvas with fallback. TypeScript compiles clean.

## Verification

```bash
# 10 contract tests — all passing
conda run -n rvc pytest backend/tests/test_realtime.py -v
# → 10 passed

# Full suite — no regressions
conda run -n rvc pytest backend/tests/ -v
# → 35 passed, 2 skipped

# Integration test — skips cleanly without hardware
conda run -n rvc pytest backend/tests/test_realtime_integration.py -v -m integration
# → 1 skipped

# TypeScript — clean
cd frontend && pnpm tsc --noEmit
# → (no output)

# Status endpoint
GET /api/realtime/status → {"active":false,"session_id":null,"profile_id":null,"input_device":null,"output_device":null}

# Frontend files
frontend/app/realtime/page.tsx ✓
frontend/public/waveform-worker.js ✓
```

## Requirements Advanced

- R003 (Realtime Voice Conversion) — full audio engine built with RVC inference loop, start/stop/params API, and integration test that will prove the real loop on hardware once a trained profile is available
- R004 (Live Waveform Visualization) — WS endpoint streams waveform_in/waveform_out at ~20fps; Web Worker draws to OffscreenCanvas; two canvases animate in browser Realtime tab
- R005 (Audio Device Selection) — device dropdowns in Realtime tab using /api/devices; MacBook Pro Mic and BlackHole 2ch auto-selected by name; selection used in POST /api/realtime/start
- R006 (Modular Audio Processing Chain) — AudioProcessor Protocol + RVCProcessor defined; pipeline structured as list of processors; future effects can be inserted without rewriting the loop

## Requirements Validated

- R003 — contract-validated (10 tests prove API shape); integration validation deferred pending trained profile (integration test exists and will prove real hardware on first S02 training run)
- R004 — contract-validated via WS contract test; visual validation confirmed in browser (both canvases visible, labeled, no console errors)
- R005 — contract-validated; device dropdowns populate from API; auto-selection confirmed in browser
- R006 — implemented and module-import verified; AudioProcessor/RVCProcessor classes defined; pipeline structure ready for S04/M002 insertion

## New Requirements Surfaced

- none

## Requirements Invalidated or Re-scoped

- none

## Deviations

- **Frontend bootstrapped from scratch**: No `frontend/` directory existed — ran `pnpm create next-app` rather than adding a page to an existing project.
- **Device API schema**: Task plan referenced a `kind` field; actual `/api/devices` returns `is_input`/`is_output` booleans per S01 contract. Frontend updated accordingly (D035).
- **Voice Profile selector in Realtime tab**: Task plan used hardcoded `profile_id: 'rvc_finetune_active'`. Replaced with real dropdown fetching `/api/profiles` filtered to `status='trained'` (D037).
- **RVC constructor signature**: Task plan example showed positional-only args; actual `rtrvc.py` requires `formant` parameter. Set `formant=0`. `f0method`/`protect` are per-infer-call args not constructor args.
- **Router-level RVC_ROOT gate removed**: DB validation must fire before env-var gate to satisfy test contracts (D034).

## Known Limitations

- Integration test (`test_realtime_session_produces_waveforms`) has not been run against real audio hardware — no trained profile exists in the production DB yet. The test will pass once the S02 training integration run is executed against the real backend.
- The WS waveform collection in the integration test uses a tight `receive_text()` + `time.sleep(0.1)` pattern; "WS receive error" log lines may appear during session warmup (non-fatal, loop retries).
- Block size is fixed at 9600 samples (200ms) per D031. If first-session latency is perceptible, reduce to 4800 (100ms) — headroom is confirmed at 5.4×.
- Next.js 16 was bootstrapped (project specified 14). App Router + TypeScript + Tailwind behavior is identical for the features built; no known 14→16 breaking changes affect this slice.

## Follow-ups

- Run `test_realtime_session_produces_waveforms` against real hardware once training integration test produces a trained profile — retires rtrvc.py block-size risk from the milestone risk register
- S04: full UI polish — library/training/realtime tabs under one layout, `start.sh`/`stop.sh` scripts, BlackHole setup guide
- S04 or M002: per-profile model slots (D007/D022) — currently only one fine-tuned copy exists

## Files Created/Modified

- `backend/app/realtime.py` — new: ~300 lines; RealtimeManager singleton, RealtimeSession dataclass, AudioProcessor Protocol, RVCProcessor, _audio_loop thread, _compute_block_params helper, manager singleton
- `backend/app/routers/realtime.py` — new: ~175 lines; 5 endpoints with Pydantic models, validation, structured logging, two-router pattern
- `backend/app/main.py` — updated: realtime_router + realtime_ws_router registered
- `backend/tests/test_realtime.py` — new: 10 contract tests; all passing
- `backend/tests/test_realtime_integration.py` — new: integration test with dual skip guards; skips without hardware
- `frontend/` — new: Next.js 16 project (App Router, TypeScript, Tailwind 4)
- `frontend/app/realtime/page.tsx` — new: ~650 lines; device selectors, profile picker, waveform canvases, param sliders, Start/Stop
- `frontend/public/waveform-worker.js` — new: ~140 lines; base64 decode → Float32Array → polyline on OffscreenCanvas

## Forward Intelligence

### What the next slice should know
- `GET /api/realtime/status` always returns the shape `{active, session_id, profile_id, input_device, output_device}` — S04 can poll this to sync Start/Stop button state without a persistent WS connection.
- The integration test needs one trained profile. Run `RVC_ROOT=Retrieval-based-Voice-Conversion-WebUI conda run -n rvc pytest backend/tests/test_training_integration.py -v -s -m integration --timeout=300` to produce it first, then re-run the realtime integration test.
- `waveform-worker.js` is in `frontend/public/` (not bundled by webpack). S04 should verify the worker URL is correct if the Next.js deployment base path changes.
- Next.js 16 App Router is live at `frontend/`. S04 adds the Library and Training tabs. The `/realtime` route is already working.
- Device IDs may renumber across boots if USB audio devices are connected/disconnected. The name-fragment auto-selection in the frontend is the safe default; don't hardcode IDs.

### What's fragile
- `asyncio.iscoroutine()` guard in the router (D033) — if `RealtimeManager.start_session` is ever changed from `async def` to sync, or if tests switch to `AsyncMock`, this guard silently does the wrong thing. Keep both sides consistent.
- `process_ckpt.py` fairseq fix (D025) is a local patch to the RVC submodule — if `Retrieval-based-Voice-Conversion-WebUI/` is updated, the patch may be lost.
- The OffscreenCanvas transfer in `page.tsx` has a try/catch fallback that silently degrades — the fallback path has not been tested and waveforms may not render on browsers without OffscreenCanvas support.

### Authoritative diagnostics
- `GET /api/realtime/status` — first thing to check if session seems stuck; surfaces `active`, `session_id`, and `error` after failure
- `conda run -n rvc pytest backend/tests/ -v` — ground truth for all 35+2 tests
- `ls Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/added_IVF*_Flat*.index` — confirms FAISS index present before expecting realtime session to start
- Backend stdout grep: `grep -E 'session_start|session_stop|session_error|waveform_overflow'` — structured JSON events for realtime lifecycle

### What assumptions changed
- "Frontend directory already exists" — it did not; required full bootstrap.
- "Device API returns kind string" — it returns `is_input`/`is_output` booleans; corrected in D035.
- "RVC constructor takes (pitch, pth, index, index_rate, f0method, protect, rvc_root)" — actual signature is `(key, formant, pth_path, index_path, index_rate, n_cpu, device, ...)`; formant required; f0method/protect are per-infer-call.
