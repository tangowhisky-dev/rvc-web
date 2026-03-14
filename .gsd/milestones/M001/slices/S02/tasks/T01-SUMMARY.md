---
id: T01
parent: S02
milestone: M001
provides:
  - backend/tests/test_training.py — 8 contract tests for training API (currently all failing, as intended)
key_files:
  - backend/tests/test_training.py
key_decisions:
  - D021 already captured: starlette TestClient for WS tests; httpx async client for REST tests
patterns_established:
  - mock_manager fixture patches backend.app.routers.training.manager (router-level reference)
  - sync_client fixture wraps starlette TestClient with same DB isolation as async client fixture
  - _make_fake_job() helper builds a MagicMock TrainingJob with pre-populated asyncio.Queue
observability_surfaces:
  - none (test scaffolding only)
duration: 20m
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T01: Write Failing Contract Tests for Training Endpoints and WebSocket

**8 contract tests written in `backend/tests/test_training.py`; all fail with AttributeError (router module not yet created) — confirming tests are real and will constrain T02–T03.**

## What Happened

Created `backend/tests/test_training.py` with 8 contract tests covering all training API surfaces. The tests are intentionally failing at this stage to prove they are real constraints on the yet-to-be-written implementation.

Two fixtures were added alongside the existing conftest pattern:
- `sync_client` — wraps `starlette.testclient.TestClient(app)` with the same temp-DB isolation as the async `client` fixture. Required because `httpx` does not support WebSocket.
- `mock_manager` — patches `backend.app.routers.training.manager` via `unittest.mock.patch`. Returns a fake `TrainingJob` (MagicMock) with a pre-populated `asyncio.Queue` containing a `{type:"done", message:"ok", phase:"train"}` message so the WebSocket consumer terminates cleanly.

Tests written:
1. `test_start_training_success` — POST with valid profile → 200 `{job_id}`
2. `test_start_training_unknown_profile` — POST with unknown profile → 404
3. `test_start_training_no_sample` — POST with NULL sample_path profile → 400
4. `test_start_training_conflict` — POST twice with ValueError on second → 409
5. `test_cancel_training` — POST cancel → 200 `{ok: true}`
6. `test_get_status_active_job` — GET status → 200 with `{phase, progress_pct, status}`
7. `test_get_status_no_job` — GET status for unknown → 404
8. `test_websocket_receives_messages` — WS connect → receive until `type=="done"`, assert keys present

## Verification

```
conda run -n rvc pytest backend/tests/test_training.py -v 2>&1 | tail -12
```

Result: **8 errors, 0 passed** — all fail at fixture setup with:
- `AttributeError: <module 'backend.app.routers'> does not have the attribute 'training'` (tests 1–7)
- `RuntimeError: There is no current event loop in thread 'MainThread'` (test 8 — secondary error that emerges once patch target fails)

Both error types confirm the router module does not yet exist. No test passes.

## Diagnostics

None — this is test scaffolding only. When T02–T03 implement the router and manager, run:

```bash
conda run -n rvc pytest backend/tests/test_training.py -v
```

Any passing test during T01 would indicate a test isolation problem.

## Deviations

None — task executed exactly as planned. The `sync_client` fixture uses `asyncio.get_event_loop().run_until_complete(db_module.init_db())` instead of being an `async` fixture, because starlette's `TestClient` is synchronous and cannot be yielded from an `async` pytest fixture without a loop context mismatch.

## Known Issues

The WebSocket test's secondary error (`no current event loop`) is a consequence of the missing patch target, not a bug in the test itself. Once `backend.app.routers.training` exists, the patch will resolve and the real asyncio fixture logic will run correctly.

## Files Created/Modified

- `backend/tests/test_training.py` — NEW: 8 contract tests for training API; all currently failing
