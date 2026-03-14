# S01: Backend Foundation & Voice Sample Library — UAT

**Milestone:** M001
**Written:** 2026-03-14

## UAT Type

- UAT mode: artifact-driven
- Why this mode is sufficient: S01 is a pure backend slice — no UI, no human experience surface. The acceptance criteria are fully covered by the `pytest` contract test suite (17 tests, all green) plus live curl verification against a running uvicorn server. No human interaction or visual inspection is required to confirm correctness.

## Preconditions

1. The `rvc` conda environment is active (or commands are prefixed with `conda run -n rvc`).
2. No other process is bound to port 8000.
3. A test audio file exists at `/tmp/test.wav` (or any `.wav` — the server accepts any filename).
4. BlackHole 2ch is installed as a macOS audio device.
5. Run from `rvc-web/` as the working directory.

## Smoke Test

```bash
cd /Users/tango16/code/rvc-web
conda run -n rvc pytest backend/tests/ -v
# Expected: 17 passed
```

## Test Cases

### 1. Full pytest contract suite

```bash
conda run -n rvc pytest backend/tests/ -v
```
**Expected:** `17 passed` — no failures, no errors, no skips.

### 2. Server health check

```bash
conda run -n rvc uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
sleep 3
curl -s http://localhost:8000/health
```
**Expected:** `{"status":"ok","db_path":"backend/data/rvc.db","rvc_root":"..."}` — HTTP 200.

### 3. Create voice profile with file upload

```bash
curl -s -X POST \
  -F "name=test_voice" \
  -F "file=@/tmp/test.wav" \
  http://localhost:8000/api/profiles
```
**Expected:** JSON object with `id` (32 hex chars), `name="test_voice"`, `status="untrained"`, `created_at` (ISO timestamp), `sample_path` containing the profile ID.

### 4. List profiles

```bash
curl -s http://localhost:8000/api/profiles
```
**Expected:** JSON array containing the profile created in test 3. Count ≥ 1.

### 5. Get profile by ID

```bash
PROFILE_ID=$(curl -s http://localhost:8000/api/profiles | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")
curl -s http://localhost:8000/api/profiles/$PROFILE_ID
```
**Expected:** Same JSON object as creation response. HTTP 200.

### 6. Get unknown profile returns 404

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/profiles/doesnotexist
```
**Expected:** `404`

### 7. Delete profile

```bash
PROFILE_ID=$(curl -s http://localhost:8000/api/profiles | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")
curl -s -o /dev/null -w "%{http_code}" -X DELETE http://localhost:8000/api/profiles/$PROFILE_ID
```
**Expected:** `204`

### 8. Deleted profile removed from list

```bash
curl -s http://localhost:8000/api/profiles
```
**Expected:** Empty JSON array `[]` (assuming only one profile was created).

### 9. Audio device list includes BlackHole 2ch

```bash
curl -s http://localhost:8000/api/devices | python3 -c "
import sys, json
devices = json.load(sys.stdin)
bh = [d for d in devices if 'BlackHole' in d['name']]
assert bh, 'BlackHole not found'
assert bh[0]['is_input'] == True
assert bh[0]['is_output'] == True
print('BlackHole OK:', bh[0])
"
```
**Expected:** Prints `BlackHole OK: {id:1, name:'BlackHole 2ch', is_input:True, is_output:True}` — no assertion error.

## Edge Cases

### Missing file field returns 422

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST \
  -F "name=no_file_voice" \
  http://localhost:8000/api/profiles
```
**Expected:** `422` (FastAPI validation error — `file` field required).

### Delete unknown profile returns 404

```bash
curl -s -o /dev/null -w "%{http_code}" -X DELETE \
  http://localhost:8000/api/profiles/ffffffffffffffffffffffffffffffff
```
**Expected:** `404`

### SQLite schema inspection

```bash
sqlite3 backend/data/rvc.db ".schema profiles"
```
**Expected:** Schema shows all 7 columns: `id`, `name`, `status`, `sample_path`, `model_path`, `index_path`, `created_at`. No missing columns.

## Failure Signals

- Any pytest failure → route logic or DB interaction is broken; check conftest isolation and router registration in main.py
- `curl http://localhost:8000/health` returns connection refused → uvicorn failed to start; check conda env and port 8000 availability
- `GET /api/devices` returns `[]` or HTTP 500 → PortAudio or sounddevice issue; check `conda run -n rvc python -c "import sounddevice; print(sounddevice.query_devices())"`
- `GET /api/devices` response missing BlackHole → BlackHole 2ch is not installed or not recognized by PortAudio; run macOS Audio MIDI Setup to confirm
- `POST /api/profiles` returns HTTP 500 on file upload → check write permissions on `data/samples/`; confirm `init_db()` created the directory
- `sqlite3 backend/data/rvc.db ".schema profiles"` shows no table → `init_db()` did not run; check lifespan wiring in `main.py`

## Requirements Proved By This UAT

- **R001 (Voice Sample Library)** — upload, list, get-by-id, and delete voice profiles all verified against a live server. Files stored on disk under `data/samples/<profile_id>/`. SQLite rows persisted with correct schema. Status field initialized as `"untrained"`. Requirements R001 is now **validated**.

## Not Proven By This UAT

- R002 (Training Pipeline) — no training subprocesses run in S01; training status transitions (`untrained → training → trained`) not exercised
- R003 (Realtime Voice Conversion) — no audio loop started; conversion not exercised
- R004 (Live Waveform Visualization) — no WebSocket waveform data produced
- R005 (Audio Device Selection) — device list is verified present but device selection for a realtime session is S03 scope
- R006–R008 (Audio pipeline modularity, start/stop scripts, BlackHole setup guide) — S03/S04 scope
- Training-to-device round-trip: BlackHole 2ch appears in `GET /api/devices` but actual audio routing to BlackHole is not tested here

## Notes for Tester

- The pytest suite is fully hermetic — it uses temp files and a temp SQLite DB. Running `pytest backend/tests/` never modifies `backend/data/rvc.db` or `data/samples/`.
- Clean up after manual curl tests by running `sqlite3 backend/data/rvc.db "DELETE FROM profiles;"` and `rm -rf data/samples/*/`.
- `RVC_ROOT` is expected to be `"not set"` in S01 — this is correct. The `/health` endpoint will show `"rvc_root":"not set"` until S02 sets it.
- BlackHole 2ch device ID is `1` on this machine. Device IDs are PortAudio-assigned and stable for a given OS audio configuration but should be reconfirmed if the macOS audio configuration changes.
