---
id: T03
parent: S02
milestone: M001
provides:
  - backend/app/routers/training.py — REST + WebSocket training router (POST /api/training/start, POST /api/training/cancel, GET /api/training/status/{profile_id}, WS /ws/training/{profile_id})
  - backend/app/main.py — updated with training_router and training_ws_router registration
key_files:
  - backend/app/routers/training.py
  - backend/app/main.py
  - backend/tests/test_training.py
key_decisions:
  - WebSocket endpoint lives on a separate ws_router (no prefix) so the path resolves to /ws/training/{profile_id} not /api/training/ws/training/{profile_id}
  - mock_manager fixture in test_training.py sets RVC_ROOT env var via monkeypatch to satisfy router's env check before mock intercepts start_job
  - sync_client fixture's asyncio.get_event_loop().run_until_complete() changed to asyncio.run() for Python 3.11 compatibility
patterns_established:
  - Two-router pattern for REST+WebSocket in same module: router (prefixed) for REST, ws_router (no prefix) for WS — both imported and registered in main.py
  - WebSocket handler breaks loop on type=="done" or type=="error" without explicit close() — lets client initiate close to avoid Starlette TestClient race
observability_surfaces:
  - GET /api/training/status/{profile_id} — runtime polling surface (phase, progress_pct, status, error)
  - POST /api/training/start emits {"event":"training_start","profile_id":...,"job_id":...} structured log via logger.info
  - WebSocket /ws/training/{profile_id} — real-time {type,message,phase} stream
  - GET /api/training/status returns {status:"failed","error":"..."} — inspectable without a WebSocket client
duration: ~25 minutes
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T03: Wire Training REST + WebSocket Router into FastAPI

**Created `backend/app/routers/training.py` with 4 REST endpoints + 1 WebSocket endpoint wired into FastAPI; all 8 contract tests and the full 25-test suite pass green.**

## What Happened

Built `backend/app/routers/training.py` with:
- `POST /api/training/start` — validates profile exists + has sample_path + RVC_ROOT is set → calls `manager.start_job()` → returns `{job_id}`; raises 404/400/409 as specified
- `POST /api/training/cancel` — returns 404 if no job, 200 `{ok: true}` on success
- `GET /api/training/status/{profile_id}` — returns 404 if no job, 200 `{phase, progress_pct, status, error}` if job exists
- `WS /ws/training/{profile_id}` — polls up to 30s for job to appear, drains queue until `done`/`error`, breaks loop without explicit `websocket.close()` to avoid Starlette TestClient race

Key deviation found during implementation: the WebSocket endpoint cannot share the `router`'s `/api/training` prefix if it should be reachable at `/ws/training/{profile_id}`. Introduced a second `ws_router = APIRouter()` with no prefix for the WebSocket route; both routers are registered in `main.py`.

Two bugs in T01's test fixtures were fixed:
1. `sync_client` used `asyncio.get_event_loop().run_until_complete()` which raises `RuntimeError` in Python 3.11 when no loop is set — changed to `asyncio.run()`
2. `mock_manager` fixture had no `monkeypatch` parameter and didn't set `RVC_ROOT` — the router checks this before calling `manager.start_job()`, so the mock never got a chance to intercept. Added `monkeypatch` param + `monkeypatch.setenv("RVC_ROOT", "/fake/rvc_root")`.

## Verification

```
conda run -n rvc pytest backend/tests/test_training.py -v
# → 8 passed in 0.20s

conda run -n rvc pytest backend/tests/ -v
# → 25 passed in 0.26s (17 S01 + 8 S02)
```

All must-haves checked:
- ✅ POST /start: 404 unknown profile, 400 no sample_path, 400 no RVC_ROOT, 409 double-start, 200+job_id on success
- ✅ POST /cancel: 404 no job, 200+ok on success
- ✅ GET /status: 404 no job, 200+{phase,progress_pct,status,error} on hit
- ✅ WebSocket polls up to 30s for job appearance
- ✅ WebSocket closes gracefully on done/error
- ✅ Router registered in main.py — endpoints reachable at /api/training/* and /ws/training/*
- ✅ epochs and save_every in start request body

## Diagnostics

- `GET /api/training/status/<profile_id>` — primary runtime inspection without a WS client
- `GET /api/training/status/<profile_id>` returns `{"status":"failed","error":"<msg>"}` — failure visible post-run
- WebSocket at `ws://localhost:8000/ws/training/<profile_id>` — live stream during training
- Structured logs to stdout: `{"event":"training_start",...}` on start

## Deviations

1. **WebSocket router split**: The task plan said "registered on the `app` directly" or "include_router handles it automatically". In practice, including a router with `/api/training` prefix makes the WS route `/api/training/ws/training/{id}`. Fix: introduced `ws_router` (no prefix) for the WS endpoint. Both routers imported and registered in `main.py`.

2. **Test fixture fixes**: Two bugs in T01's `test_training.py` fixtures needed repair:
   - `asyncio.get_event_loop()` → `asyncio.run()` in `sync_client` (Python 3.11 compat)
   - `mock_manager` fixture lacked `monkeypatch`+`RVC_ROOT` setup needed for the router's env check to pass

## Known Issues

None.

## Files Created/Modified

- `backend/app/routers/training.py` — new; REST + WebSocket training router
- `backend/app/main.py` — updated; added training_router + training_ws_router registration
- `backend/tests/test_training.py` — updated; fixed sync_client (asyncio.run) and mock_manager (monkeypatch + RVC_ROOT)
