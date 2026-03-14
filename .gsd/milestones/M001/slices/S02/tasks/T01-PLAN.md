---
estimated_steps: 4
estimated_files: 1
---

# T01: Write Failing Contract Tests for Training Endpoints and WebSocket

**Slice:** S02 — Training Pipeline with Live Log Streaming
**Milestone:** M001

## Description

Create `backend/tests/test_training.py` with contract tests for all training API surfaces: `POST /api/training/start`, `POST /api/training/cancel`, `GET /api/training/status/:profile_id`, and `WS /ws/training/:profile_id`. Tests use mocked subprocesses so no real RVC subprocesses run. All tests must fail at this point (import or routing errors) — that failure proves the tests are real and will constrain the implementation in T02–T03.

The WebSocket tests use `starlette.testclient.TestClient.websocket_connect()` (synchronous API) since `httpx` does not support WebSocket. The mock strategy: patch `backend.app.training.manager` at module scope so `start_job` and `cancel_job` return controlled responses without spawning processes.

## Steps

1. Create `backend/tests/test_training.py`. Import `TestClient` from `starlette.testclient` (for WebSocket support) alongside the existing `httpx` async client.
2. Write a `sync_client` pytest fixture using `starlette.testclient.TestClient(app)` — needed for WebSocket tests. The existing `client` fixture (httpx async) can be reused for REST tests.
3. Write a `mock_manager` fixture that patches `backend.app.routers.training.manager` (after router exists) with a `MagicMock`. The mock's `start_job` returns a fake `TrainingJob` with `job_id="test-job-001"` and an `asyncio.Queue` pre-populated with a `{type:"done", message:"ok", phase:"train"}` message. The mock's `cancel_job` returns None. The mock's `get_job` returns the fake job.
4. Write the following tests:
   - `test_start_training_success` — POST `/api/training/start` with a valid `profile_id` in DB → 200 `{job_id}`.
   - `test_start_training_unknown_profile` — POST with unknown `profile_id` → 404.
   - `test_start_training_no_sample` — POST with a profile that has `sample_path=None` → 400.
   - `test_start_training_conflict` — POST twice with manager raising `ValueError("job already running")` on second call → 409.
   - `test_cancel_training` — POST `/api/training/cancel` → 200 `{ok: true}`.
   - `test_get_status_active_job` — GET `/api/training/status/<profile_id>` → 200 with `{phase, progress_pct, status}` fields present.
   - `test_get_status_no_job` — GET `/api/training/status/unknown` → 404.
   - `test_websocket_receives_messages` — `sync_client.websocket_connect("/ws/training/<profile_id>")`, receive messages until `type=="done"`, assert at least one message received with `type`, `message`, `phase` keys present.
5. Confirm all tests fail (router not yet wired): `conda run -n rvc pytest backend/tests/test_training.py -v 2>&1 | head -30` — should show collection errors or import errors.

## Must-Haves

- [ ] `TestClient` from starlette imported alongside httpx client — both fixtures present
- [ ] All 8 tests written with real assertions (not just `assert True`)
- [ ] WebSocket test uses `TestClient.websocket_connect()`, not httpx
- [ ] Mock patches the router-level `manager` reference (not the training module directly)
- [ ] All tests currently fail with import/routing errors (confirms they're real)

## Verification

```bash
conda run -n rvc pytest backend/tests/test_training.py -v 2>&1 | head -40
```
Expected: all tests show ERROR or FAILED — no passes.

## Observability Impact

- Signals added/changed: None (test file only)
- How a future agent inspects this: `conda run -n rvc pytest backend/tests/test_training.py -v` — any pass in T01 indicates a problem with test isolation
- Failure state exposed: None (test scaffolding only)

## Inputs

- `backend/tests/conftest.py` — existing `client` fixture pattern with monkeypatch isolation; anyio_backend convention
- `backend/app/main.py` — FastAPI `app` object to wrap in TestClient
- S01 pattern: `async with get_db() as db` for DB setup in test fixtures

## Expected Output

- `backend/tests/test_training.py` — 8 contract tests, all currently failing with import/routing errors
