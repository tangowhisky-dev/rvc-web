---
estimated_steps: 5
estimated_files: 2
---

# T03: Implement REST + WebSocket router and wire into main.py

**Slice:** S03 — Realtime Voice Conversion & Waveform Visualization
**Milestone:** M001

## Description

Create `backend/app/routers/realtime.py` with the five realtime endpoints and wire both its routers into `backend/app/main.py`. This task closes the loop between the audio engine (T02) and the test contracts (T01) — after T03, all 10 contract tests must be green and the full suite must be 35 passed.

Follow the exact two-router pattern from D023: `router = APIRouter(prefix='/api/realtime')` for REST; `ws_router = APIRouter()` (no prefix) for `WS /ws/realtime/{session_id}`.

## Steps

1. **Create `backend/app/routers/realtime.py`** with imports and two routers:
   ```python
   router = APIRouter(prefix="/api/realtime", tags=["realtime"])
   ws_router = APIRouter(tags=["realtime"])
   ```

2. **Define Pydantic request/response models**:
   - `StartSessionRequest(profile_id, input_device_id, output_device_id, pitch=0.0, index_rate=0.75, protect=0.33)`
   - `StopSessionRequest(session_id)`
   - `UpdateParamsRequest(session_id, pitch=None, index_rate=None, protect=None)` — all optional
   - `StartSessionResponse(session_id)`
   - `StatusResponse(active, session_id, profile_id, input_device, output_device)` — all nullable except `active`

3. **Implement five endpoints**:

   **POST `/start`**:
   - Read `RVC_ROOT` from env; 400 if not set
   - Call `await manager.start_session(...)` inside try/except:
     - `ValueError` with "not trained" → `HTTPException(400, ...)`
     - `ValueError` with "no index file" → `HTTPException(400, ...)`
     - `ValueError` with "session already active" → `HTTPException(409, ...)`
     - `Exception` for profile not found (check DB result) → `HTTPException(404, ...)`
   - Return `StartSessionResponse(session_id=session.session_id)`
   - Log `{"event":"realtime_start", "session_id":..., "profile_id":...}` structured JSON

   **POST `/stop`**:
   - `session = manager.get_session(request.session_id)`; 404 if None
   - Call `manager.stop_session(request.session_id)` — synchronous (no await; stop_session sets event and joins thread)
   - Return `{"ok": True}`

   **POST `/params`**:
   - `session = manager.get_session(request.session_id)`; 404 if None
   - Call `manager.update_params(request.session_id, request.pitch, request.index_rate, request.protect)`
   - Return `{"ok": True}`

   **GET `/status`**:
   - `session = manager.active_session()`
   - If None: return `{"active": False, "session_id": None, "profile_id": None, "input_device": None, "output_device": None}`
   - Else: return `{"active": True, "session_id": session.session_id, "profile_id": session.profile_id, "input_device": session.input_device, "output_device": session.output_device}`

   **WS `/ws/realtime/{session_id}` (on `ws_router`)**:
   - Accept WebSocket
   - Poll up to 30s for session to become active (mirrors training WS pattern)
   - Deliver waveform messages: use `asyncio.sleep(0.05)` poll loop; drain `_waveform_deque` (pop from left with `popleft()` — `deque` is thread-safe for single producer/consumer)
   - When `session._stop_event.is_set()` and deque is empty: send `{"type": "done"}` and close
   - Catch `WebSocketDisconnect`: log and return cleanly
   - Target ~20fps: sleep 50ms between deque polls

4. **Update `backend/app/main.py`**: add imports for `realtime_router` and `realtime_ws_router` from `backend.app.routers.realtime`; add two `app.include_router(...)` calls after training routers.

5. **Run full test suite** and fix any failures.

## Must-Haves

- [ ] Two-router pattern (D023): `router` prefixed `/api/realtime`, `ws_router` unprefixed
- [ ] POST `/start` returns 404/400/409 matching T01 test expectations exactly
- [ ] WS endpoint on `ws_router` at `/ws/realtime/{session_id}` (not on prefixed router)
- [ ] WS uses `deque.popleft()` for thread-safe consumption (not queue.get — waveform_deque is a collections.deque not asyncio.Queue)
- [ ] WS sends `{type:"done"}` before closing when stop_event is set
- [ ] Both routers registered in `main.py`
- [ ] All 10 contract tests pass after this task
- [ ] Full suite 35 passed, 1 skipped (no regressions)

## Verification

```bash
# All 10 realtime contract tests pass
conda run -n rvc pytest backend/tests/test_realtime.py -v
# Expected: 10 passed

# Full suite — no regressions
conda run -n rvc pytest backend/tests/ -v
# Expected: 35 passed, 1 skipped

# Smoke check routes are registered
cd /Users/tango16/code/rvc-web
conda run -n rvc python -c "
from backend.app.main import app
routes = [r.path for r in app.routes]
assert '/api/realtime/start' in routes, routes
assert '/api/realtime/stop' in routes, routes
assert '/api/realtime/status' in routes, routes
assert '/ws/realtime/{session_id}' in routes, routes
print('All routes registered:', [r for r in routes if 'realtime' in r])
"
```

## Observability Impact

- Signals added/changed:
  - `{"event":"realtime_start", "session_id":..., "profile_id":...}` — logged at POST /start
  - `{"event":"ws_realtime_disconnect", "session_id":...}` — logged on WebSocket client disconnect
  - `GET /api/realtime/status` — live session state surface; inspectable at any time without a live WS connection
- How a future agent inspects this: `curl -s http://localhost:8000/api/realtime/status` returns `{active, session_id, profile_id, input_device, output_device}` — fast no-auth diagnostic
- Failure state exposed: 409 response body names the conflict; 400 body names the missing prerequisite (no index file / not trained); session error accessible via status endpoint after T02 error propagation

## Inputs

- `backend/app/realtime.py` (T02 output) — `manager` singleton; `start_session`, `stop_session`, `get_session`, `active_session`, `update_params` methods; `RealtimeSession._waveform_deque`, `_stop_event`
- `backend/tests/test_realtime.py` (T01 output) — exact HTTP status codes and response body shapes the router must satisfy
- `backend/app/routers/training.py` — two-router pattern template (D023)
- `backend/app/main.py` — router registration pattern
- `S03-RESEARCH.md` → API Contract section — endpoint shapes and error codes

## Expected Output

- `backend/app/routers/realtime.py` — REST + WebSocket router (~130 lines): 5 endpoints, Pydantic models, structured logging
- `backend/app/main.py` — updated with realtime_router + realtime_ws_router registration
- Contract test result: `conda run -n rvc pytest backend/tests/test_realtime.py -v` → **10 passed**
- Full suite result: `conda run -n rvc pytest backend/tests/ -v` → **35 passed, 1 skipped**
