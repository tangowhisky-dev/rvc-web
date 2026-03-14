---
id: T01
parent: S03
milestone: M001
provides:
  - 10 contract tests for all 5 realtime API endpoints in backend/tests/test_realtime.py
key_files:
  - backend/tests/test_realtime.py
key_decisions:
  - mock_manager fixture uses patch('backend.app.routers.realtime.manager') at router-module scope, matching D023 and the training test pattern
  - _make_fake_session() pre-populates _waveform_deque with 2 waveform messages + None sentinel; WS handler must send {type:"done"} when it dequeues None or stop_event is set
  - sync_client fixture uses starlette.testclient.TestClient for WebSocket test (httpx does not support WebSocket)
  - _insert_trained_profile() helper inserts rows directly via get_db() to enable 400 vs 404 distinction tests without a full profile upload flow
patterns_established:
  - async _insert_trained_profile(tmp_path, profile_id, status) for DB seeding in realtime tests
  - mock_manager.active_session.return_value = None (default inactive) vs _make_fake_session() (active) controls 409/status behavior
  - mock_manager.start_session.side_effect = HTTPException(400, ...) for error path simulation
observability_surfaces:
  - conda run -n rvc pytest backend/tests/test_realtime.py -v — shows exactly which API contracts are unmet
duration: 20m
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T01: Write 10 contract tests for realtime API (all initially failing)

**10 contract tests written in `backend/tests/test_realtime.py`; all 10 collected cleanly; all 10 fail with implementation-missing errors (router module does not exist).**

## What Happened

Created `backend/tests/test_realtime.py` mirroring the fixture structure from `test_training.py` exactly:

- `_make_fake_session(session_id)` builds a `MagicMock` with `session_id`, `profile_id`, `status="active"`, `input_device=2`, `output_device=1`, `error=None`, `_stop_event` (threading.Event, not set), and `_waveform_deque` pre-populated with 2 waveform messages + `None` sentinel.
- `client` fixture: async `AsyncClient` with isolated temp DB (identical to conftest pattern).
- `sync_client` fixture: `starlette.testclient.TestClient` with isolated temp DB for the WebSocket test.
- `mock_manager` fixture: patches `backend.app.routers.realtime.manager` at router-module scope; sets `active_session.return_value = None` by default.
- `_insert_trained_profile()` async helper inserts profile rows directly into the isolated DB to enable testing 404 (unknown profile), 400 (untrained), and 200 (trained) paths.

10 tests cover all 5 endpoints: 5 for POST /start (200, 404, 400-not-trained, 400-no-index, 409), 2 for POST /stop (200, 404), 1 for POST /params (200), 1 for GET /status (inactive), 1 WS test (2 waveform messages + done).

## Verification

```
conda run -n rvc pytest backend/tests/test_realtime.py --collect-only
# → 10 tests collected in 0.07s  ✓

conda run -n rvc pytest backend/tests/test_realtime.py -v 2>&1 | tail -15
# → 10 errors (AttributeError: module 'backend.app.routers' has no attribute 'realtime')
# All fail for implementation-missing reason: router module not yet created  ✓
```

## Diagnostics

- `conda run -n rvc pytest backend/tests/test_realtime.py -v` — shows exactly which contracts are unmet
- Each test name documents the specific contract: `test_start_404_unknown_profile`, `test_start_400_profile_not_trained`, etc.
- Errors are fixture setup errors (patch target doesn't exist), not collection errors

## Deviations

None — plan followed exactly.

## Known Issues

None.

## Files Created/Modified

- `backend/tests/test_realtime.py` — NEW: 10 contract tests for all 5 realtime API endpoints; all collected, all failing for implementation-missing reasons
