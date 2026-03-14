---
estimated_steps: 4
estimated_files: 4
---

# T04: Implement devices endpoint and complete contract test suite

**Slice:** S01 — Backend Foundation & Voice Sample Library
**Milestone:** M001

## Description

`GET /api/devices` exposes the system audio device list consumed by S03's realtime VC UI. This task implements the route by calling `sounddevice.query_devices()` synchronously (confirmed safe, <1ms) and mapping each device to the boundary contract shape `{id, name, is_input, is_output}`. The devices router is registered in `main.py`. `test_devices.py` is completed with concrete assertions. After this task the full `pytest` suite must be green — this is the slice-level stopping condition.

## Steps

1. Implement `backend/app/routers/devices.py` with `APIRouter(prefix="/api")`:
   - `GET /devices` — call `sounddevice.query_devices()`; iterate over the returned list of dicts; map each to `DeviceOut(id=int(i), name=d["name"], is_input=d["max_input_channels"]>0, is_output=d["max_output_channels"]>0)`; return the list.
   - Note: `sounddevice.query_devices()` returns a `sounddevice.DeviceList` — iterate with `enumerate` to get integer indices as device IDs.
2. Register the devices router in `backend/app/main.py`: `app.include_router(devices_router)`.
3. Complete `backend/tests/test_devices.py`: use `TestClient`; assert response is 200, body is a non-empty list, every item has `id` (int), `name` (str), `is_input` (bool), `is_output` (bool); assert at least one device name contains `"BlackHole"` (confirming sounddevice is queried on the real system, not mocked).
4. Run `conda run -n rvc pytest backend/tests/ -v` from `rvc-web/` and confirm all tests in both `test_profiles.py` and `test_devices.py` pass.

## Must-Haves

- [ ] `GET /api/devices` returns a list with at least one entry
- [ ] Every entry has `id: int`, `name: str`, `is_input: bool`, `is_output: bool`
- [ ] `sounddevice.query_devices()` called without arguments (returns all devices)
- [ ] Device IDs are stable integer indices (not names — names can contain non-ASCII)
- [ ] Devices router registered in `main.py`
- [ ] `test_devices.py` contains assertions that BlackHole 2ch appears in the device list
- [ ] Full `pytest` suite (profiles + devices) passes green

## Verification

- `conda run -n rvc pytest backend/tests/ -v` → all tests pass, zero failures
- `conda run -n rvc uvicorn backend.app.main:app --port 8000 &` then `curl -s http://localhost:8000/api/devices | python3 -m json.tool` → formatted JSON list with BlackHole 2ch visible

## Observability Impact

- Signals added/changed: `GET /api/devices` is a real-time diagnostic surface — calling it at any time shows the current OS audio device state (useful when BlackHole disappears from device list after system audio changes)
- How a future agent inspects this: `curl http://localhost:8000/api/devices` to confirm BlackHole 2ch is present with expected device id before configuring realtime session in S03
- Failure state exposed: If `sounddevice` fails (e.g. PortAudio not found), the exception propagates as HTTP 500 with the PortAudio error string — this is explicit and actionable

## Inputs

- `backend/app/main.py` — profiles router already registered from T03; CORS and lifespan in place
- `backend/tests/test_devices.py` — stub assertions from T01 (currently failing)
- Research finding: `sounddevice.query_devices()` confirmed working in `rvc` env; device 1 = BlackHole 2ch

## Expected Output

- `backend/app/routers/devices.py` — GET /devices route implementation
- `backend/app/main.py` — devices router registered (updated)
- `backend/tests/test_devices.py` — all assertions implemented and passing
- Full `pytest` suite green: every test in `backend/tests/` passes
