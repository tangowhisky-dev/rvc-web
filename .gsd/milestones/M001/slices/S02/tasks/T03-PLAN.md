---
estimated_steps: 5
estimated_files: 3
---

# T03: Wire Training REST + WebSocket Router into FastAPI

**Slice:** S02 — Training Pipeline with Live Log Streaming
**Milestone:** M001

## Description

Create `backend/app/routers/training.py` with the four REST endpoints and the WebSocket endpoint, then register the router in `main.py`. This wires the `TrainingManager` from T02 into HTTP/WebSocket surfaces. After this task, all 8 contract tests in `test_training.py` should pass.

Key design points:
- `POST /api/training/start` body: `{profile_id: str, epochs: int = 200, save_every: int = 10}` — exposes configurable epoch count for integration testing.
- WebSocket route at `/ws/training/{profile_id}` — registered on the `app` directly (FastAPI WebSocket routes work on routers too but must be careful about prefix).
- The WebSocket handler polls for the job to appear (up to 30s) to handle the race where a client connects before `start_job` finishes setup.
- Profile `sample_path` is stored relative to `rvc-web/` CWD; must be made absolute before passing to `start_job`.

## Steps

1. Create `backend/app/routers/training.py`:
   - Import `TrainingManager`, `manager` from `backend.app.training`
   - Import `get_db` from `backend.app.db`
   - Define Pydantic request models: `StartTrainingRequest(profile_id: str, epochs: int = 200, save_every: int = 10)`, `CancelTrainingRequest(profile_id: str)`
   - Define Pydantic response models: `StartTrainingResponse(job_id: str)`, `StatusResponse(phase: str, progress_pct: int, status: str, error: Optional[str])`
   - `APIRouter` with `prefix="/api/training"`, `tags=["training"]`

2. Implement `POST /start`:
   - Parse `StartTrainingRequest`
   - Open `get_db()`, `SELECT * FROM profiles WHERE id = ?` — return 404 if not found
   - Return 400 if `profile.sample_path is None`
   - Derive `sample_dir = os.path.dirname(os.path.abspath(profile["sample_path"]))` 
   - Derive `rvc_root = os.environ.get("RVC_ROOT", "")` — return 400 with `{"detail":"RVC_ROOT not set"}` if empty
   - Call `manager.start_job(profile_id, sample_dir, rvc_root, total_epoch=request.epochs, save_every=request.save_every)` — catch `ValueError` → return 409
   - Return `StartTrainingResponse(job_id=job.job_id)`
   - Emit structured log: `{"event":"training_start","profile_id":...,"job_id":...}`

3. Implement `POST /cancel`:
   - Parse `CancelTrainingRequest`
   - Call `manager.cancel_job(profile_id)` — if job not found, return 404
   - Return `{"ok": True}`

4. Implement `GET /status/{profile_id}`:
   - `job = manager.get_job(profile_id)` — return 404 if None
   - Return `StatusResponse(phase=job.phase, progress_pct=job.progress_pct, status=job.status, error=job.error)`

5. Implement WebSocket endpoint `WS /ws/training/{profile_id}` — add directly to the router (FastAPI supports WebSocket on Router since v0.95):
   - `await websocket.accept()`
   - Poll for job: loop up to 300 iterations × 100ms = 30s timeout: `job = manager.get_job(profile_id)`. If not found after timeout, send `{type:"error","message":"job not found"}` and close.
   - Once job found: loop `msg = await asyncio.wait_for(job.queue.get(), timeout=60.0)` → `await websocket.send_json(msg)`. On `type=="done"` or `type=="error"`, send the message then break.
   - Close websocket on exit.

6. In `backend/app/main.py`: add `from backend.app.routers.training import router as training_router` and `app.include_router(training_router)`. Also register the WebSocket route — if the WebSocket endpoint is on the training router, the include_router call handles it automatically.

## Must-Haves

- [ ] `POST /api/training/start` returns 404 for unknown profile, 400 for missing sample_path, 400 for missing RVC_ROOT, 409 for double-start, 200 with job_id on success
- [ ] `POST /api/training/cancel` returns 404 if no job exists for profile, 200 on success
- [ ] `GET /api/training/status/{profile_id}` returns 404 if no job, 200 with `{phase, progress_pct, status, error}` if job exists
- [ ] WebSocket handler polls up to 30s for job appearance before giving up
- [ ] WebSocket handler closes gracefully on `done` or `error` message type
- [ ] Router registered in `main.py` — endpoints reachable at `/api/training/*` and `/ws/training/*`
- [ ] `epochs` and `save_every` fields in start request body (for integration test override)

## Verification

```bash
# All contract tests must pass
conda run -n rvc pytest backend/tests/test_training.py -v

# Full suite still green
conda run -n rvc pytest backend/tests/ -v

# Confirm routing (server must be running in separate terminal)
# curl -s http://localhost:8000/api/training/status/unknown → 404
# curl -s -X POST http://localhost:8000/api/training/cancel -H "Content-Type: application/json" -d '{"profile_id":"unknown"}' → 404
```

## Observability Impact

- Signals added/changed:
  - `POST /api/training/start` emits `{"event":"training_start",...}` structured log to stdout
  - `GET /api/training/status/:profile_id` is the primary runtime inspection surface
  - WebSocket stream provides real-time phase visibility
- How a future agent inspects this: `GET /api/training/status/<profile_id>` at any time; WebSocket client to watch live log stream
- Failure state exposed: `GET /api/training/status` returns `{status:"failed","error":"<last error>"}` — inspectable without a WebSocket client

## Inputs

- `backend/app/training.py` (T02) — `manager` singleton, `TrainingJob`, `TrainingManager`
- `backend/app/db.py` — `get_db()` context manager
- `backend/tests/test_training.py` (T01) — contract test assertions to satisfy
- S01 pattern: `backend/app/routers/profiles.py` — how routes are structured and registered

## Expected Output

- `backend/app/routers/training.py` — REST + WebSocket router, complete
- `backend/app/main.py` — updated with training router import and registration
- All 8 contract tests in `backend/tests/test_training.py` passing
- Full test suite (17 S01 tests + 8 new training tests) — 25/25 green
