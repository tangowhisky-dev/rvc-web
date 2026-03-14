---
id: T04
parent: S01
milestone: M001
provides:
  - backend/app/routers/devices.py — GET /api/devices route calling sounddevice.query_devices()
  - backend/app/main.py — devices router registered
  - backend/tests/test_devices.py — 7 concrete assertions all passing
  - Full pytest suite (17/17) green — slice-level stopping condition met
key_files:
  - backend/app/routers/devices.py
  - backend/app/main.py
  - backend/tests/test_devices.py
key_decisions:
  - synchronous sounddevice.query_devices() used directly (not wrapped in run_in_executor) — confirmed safe, <1ms, blocking thread risk negligible
patterns_established:
  - DeviceOut Pydantic model enforces response schema; integer enumerate() index used as stable device ID (not name — names can contain non-ASCII)
  - PortAudio errors propagate as HTTP 500 with error string — explicit and actionable
observability_surfaces:
  - GET /api/devices — real-time OS audio device state; call at any time to confirm BlackHole 2ch is present before configuring realtime VC session in S03
  - HTTP 500 with PortAudio error string if sounddevice fails (e.g. PortAudio not installed)
duration: <5min
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T04: Implement devices endpoint and complete contract test suite

**`GET /api/devices` implemented via sounddevice.query_devices(); all 17 contract tests pass (7 devices + 10 profiles); slice S01 verification fully green.**

## What Happened

Created `backend/app/routers/devices.py` with an `APIRouter(prefix="/api")` and a single `GET /devices` route. The route calls `sd.query_devices()` synchronously, enumerates the result, and maps each entry to a `DeviceOut` Pydantic model with fields `id` (int index), `name` (str), `is_input` (bool), `is_output` (bool). Registered the router in `backend/app/main.py` alongside the profiles router.

The 7 stub tests in `test_devices.py` were already fully written from T01 — no changes needed; they passed immediately once the route existed.

## Verification

```
conda run -n rvc pytest backend/tests/ -v
# → 17 passed in 0.20s (7 devices + 10 profiles)

curl -s http://localhost:8000/health
# → {"status":"ok","db_path":"backend/data/rvc.db","rvc_root":"not set"}

curl -s http://localhost:8000/api/devices | python3 -m json.tool
# → [{id:0,name:"iPhone Microphone",...}, {id:1,name:"BlackHole 2ch",is_input:true,is_output:true}, ...]
```

All slice-level verification checks passed:
- `pytest backend/tests/` — 17/17 green ✓
- `GET /health` → `{"status":"ok"}` ✓
- `GET /api/devices` → list with BlackHole 2ch at id=1 ✓

## Diagnostics

- `curl http://localhost:8000/api/devices` — confirm BlackHole 2ch is present with its integer device ID before configuring realtime VC session in S03
- Device IDs are PortAudio-assigned indices — stable for a given OS audio configuration
- If PortAudio fails: HTTP 500 with `sounddevice.PortAudioError` message string propagated directly

## Deviations

none

## Known Issues

none

## Files Created/Modified

- `backend/app/routers/devices.py` — new; GET /api/devices route with DeviceOut model
- `backend/app/main.py` — added devices router import and include_router call
