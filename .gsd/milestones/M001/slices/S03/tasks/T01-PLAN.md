---
estimated_steps: 4
estimated_files: 1
---

# T01: Write 10 contract tests for realtime API (all initially failing)

**Slice:** S03 — Realtime Voice Conversion & Waveform Visualization
**Milestone:** M001

## Description

Write `backend/tests/test_realtime.py` with 10 contract tests covering all five realtime API endpoints. All tests must fail until T02–T03 create the implementation. This is the contract-first gate that constrains T02 and T03 — the passing condition for the slice is these 10 tests turning green.

Mirror the `test_training.py` fixture structure exactly: `sync_client` fixture wraps `starlette.testclient.TestClient` for WebSocket tests; `mock_manager` patches `backend.app.routers.realtime.manager` at router-module scope so no real audio hardware is touched.

## Steps

1. Create `backend/tests/test_realtime.py` with module docstring and imports (`asyncio`, `uuid`, `unittest.mock`, `pytest`, `httpx`, `starlette.testclient.TestClient`).

2. Write shared helpers:
   - `_make_fake_session(session_id)` → returns a `MagicMock` with `session_id`, `profile_id`, `status="active"`, `input_device=2`, `output_device=1`, `error=None`, `_stop_event` (threading.Event that is not set), and `_waveform_deque` pre-populated with 2 fake waveform messages followed by a sentinel `None` to trigger the done close (the WS handler should be written to send `{type:"done"}` when deque yields None or stop event fires).

3. Write `sync_client` fixture: wraps the FastAPI `app` in `starlette.testclient.TestClient`.

4. Write `mock_manager` fixture: patches `backend.app.routers.realtime.manager` using `unittest.mock.patch` as a context manager (or `@pytest.fixture` with `with patch(...) as mock: yield mock`). Sets `mock.active_session.return_value = None` by default (inactive state). Must also monkeypatch a trained profile into the DB so 404 vs 400 tests work correctly.

5. Write the 10 contract tests:

   **REST (via async `client` fixture + `mock_manager`):**
   - `test_start_returns_session_id`: mock `active_session()` → None, `start_session()` returns fake session; POST `/api/realtime/start` with valid trained profile → 200 with `session_id` key.
   - `test_start_404_unknown_profile`: POST with nonexistent profile_id → 404.
   - `test_start_400_profile_not_trained`: insert profile with status=`untrained` into DB; POST → 400 with message containing "not trained".
   - `test_start_400_no_index_file`: mock `start_session()` to raise a 400-style exception (`HTTPException(400, "no index file found")`); POST → 400.
   - `test_start_409_already_active`: mock `active_session()` returns a fake session; POST → 409.
   - `test_stop_200_ok`: mock `get_session()` returns fake session; POST `/api/realtime/stop` with valid `session_id` → 200 `{ok: true}`.
   - `test_stop_404_unknown`: mock `get_session()` returns None; POST `/api/realtime/stop` → 404.
   - `test_params_200_ok`: mock `get_session()` returns fake session with mock `rvc` attribute; POST `/api/realtime/params` → 200 `{ok: true}`.
   - `test_status_inactive`: mock `active_session()` → None; GET `/api/realtime/status` → 200 `{active: false}`.

   **WebSocket (via `sync_client` fixture):**
   - `test_ws_streams_waveform_and_done`: pre-populate the fake session's `_waveform_deque` with 2 waveform messages; connect to `WS /ws/realtime/{session_id}`; assert first 2 received messages have `type` in (`waveform_in`, `waveform_out`); assert final message has `type == "done"`.

6. Confirm all tests are syntactically valid by running pytest in collect-only mode.

## Must-Haves

- [ ] All 10 tests collected without syntax errors or fixture errors
- [ ] All 10 tests fail for implementation-missing reasons (ImportError, 404, or assertion failure — not pytest collection errors)
- [ ] Mock strategy uses `patch('backend.app.routers.realtime.manager')` at router scope
- [ ] WS test uses `starlette.testclient.TestClient.websocket_connect()` (not httpx)
- [ ] `sync_client` and `mock_manager` fixtures are clearly separated (reuse pattern from test_training.py)

## Verification

```bash
# Collect-only — must show 10 tests with no errors
conda run -n rvc pytest backend/tests/test_realtime.py --collect-only

# Full run — all 10 must fail (not error)
conda run -n rvc pytest backend/tests/test_realtime.py -v 2>&1 | tail -15
# Expected: 10 FAILED (ImportError or 404 on missing router)
```

## Observability Impact

- Signals added/changed: None (test file only)
- How a future agent inspects this: `conda run -n rvc pytest backend/tests/test_realtime.py -v` shows exactly which contracts are unmet
- Failure state exposed: Each failing test names the exact API contract that is not yet satisfied

## Inputs

- `backend/tests/conftest.py` — async `client` fixture and `anyio_backend` session fixture; reuse as-is
- `backend/tests/test_training.py` — `sync_client`, `mock_manager`, `_make_fake_job` patterns to mirror exactly
- `S03-PLAN.md` Must-Haves — exact list of 10 test scenarios
- `S03-RESEARCH.md` → Test Strategy section — mock strategy and WS pre-population pattern

## Expected Output

- `backend/tests/test_realtime.py` — 10 contract tests, all collected, all failing for implementation-missing reasons (not collection or fixture errors)
