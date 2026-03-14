# S01: Backend Foundation & Voice Sample Library

**Goal:** FastAPI server runs from `rvc-web/`, SQLite schema is initialized, REST API for creating/listing/getting/deleting voice profiles is live, file upload stores audio to disk and records to the DB, and `GET /api/devices` returns system audio devices — all verified by `pytest` contract tests and `curl`.
**Demo:** `conda run -n rvc pytest backend/tests/` passes all assertions; `curl -F "file=@sample.wav" -F "name=test" http://localhost:8000/api/profiles` returns a JSON profile with an ID; `curl http://localhost:8000/api/devices` returns a list that includes BlackHole 2ch.

## Must-Haves

- `aiosqlite`, `websockets`, `pytest`, and `pytest-asyncio` installed into the `rvc` conda env
- `backend/app/main.py` — FastAPI app with CORS for `localhost:3000` and lifespan startup (creates DB + directories)
- SQLite schema: `profiles(id TEXT PK, name TEXT, status TEXT, sample_path TEXT, model_path TEXT, index_path TEXT, created_at TEXT)`
- `POST /api/profiles` — accepts multipart upload, saves audio to `data/samples/<profile_id>/`, inserts row, returns `{id, name, status, created_at, sample_path}`
- `GET /api/profiles` — returns list of all profiles
- `GET /api/profiles/:id` — returns single profile or 404
- `DELETE /api/profiles/:id` — removes DB row and sample directory; returns 204
- `GET /api/devices` — returns `[{id, name, is_input, is_output}]` from sounddevice
- `RVC_ROOT` env var documented and loaded (for S02/S03 forward compatibility)
- `pytest` contract tests in `backend/tests/` covering all endpoints (happy path + error paths)
- Server starts cleanly with `conda run -n rvc uvicorn backend.app.main:app --host 0.0.0.0 --port 8000` from `rvc-web/`

## Proof Level

- This slice proves: contract
- Real runtime required: yes (uvicorn must start; file I/O and SQLite must work)
- Human/UAT required: no (pytest + curl cover all verification)

## Verification

- `cd /Users/tango16/code/rvc-web && conda run -n rvc pytest backend/tests/ -v` — all tests pass
- `conda run -n rvc uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &` then:
  - `curl -s http://localhost:8000/health` → `{"status":"ok"}`
  - `curl -s -X POST -F "name=test_voice" -F "file=@/tmp/test.wav" http://localhost:8000/api/profiles` → JSON with `id`, `name`, `status:"untrained"`
  - `curl -s http://localhost:8000/api/profiles` → JSON array containing the created profile
  - `curl -s http://localhost:8000/api/devices` → JSON array with BlackHole 2ch entry

## Observability / Diagnostics

- Runtime signals: structured JSON startup log (app version, DB path, RVC_ROOT, data dir) emitted on lifespan startup; per-request errors logged with profile_id and operation name
- Inspection surfaces: `GET /health` endpoint returns `{"status":"ok","db_path":"...","rvc_root":"..."}` for liveness checks; SQLite DB at `backend/data/rvc.db` queryable with `sqlite3` CLI; uploaded files at `data/samples/<profile_id>/`
- Failure visibility: FastAPI HTTPException with `detail` field naming the failing profile_id; file-not-found and DB errors surfaced as 404/500 with structured JSON detail; startup failures logged to stderr before uvicorn accepts requests
- Redaction constraints: none (no secrets in S01)

## Integration Closure

- Upstream surfaces consumed: nothing (first slice)
- New wiring introduced in this slice: `backend/app/main.py` FastAPI app entry point; SQLite DB initialization on startup; `data/samples/` directory convention; `RVC_ROOT` env var loading pattern; `GET /api/devices` sounddevice integration
- What remains before the milestone is truly usable end-to-end: S02 (training pipeline + WebSocket log streaming), S03 (realtime VC loop + waveform WebSocket), S04 (Next.js frontend + scripts)

## Tasks

- [x] **T01: Install missing packages and create backend directory skeleton** `est:20m`
  - Why: `aiosqlite`, `pytest`, `pytest-asyncio`, and `websockets` are absent from the `rvc` env; without them nothing compiles. The full `backend/` tree must exist before any code is written.
  - Files: `backend/app/__init__.py`, `backend/app/main.py` (stub), `backend/app/db.py` (stub), `backend/app/routers/__init__.py`, `backend/app/routers/profiles.py` (stub), `backend/app/routers/devices.py` (stub), `backend/tests/__init__.py`, `backend/tests/test_profiles.py` (failing stubs), `backend/tests/test_devices.py` (failing stubs), `.env.example`
  - Do: Run `conda run -n rvc pip install aiosqlite websockets pytest pytest-asyncio`; verify imports succeed; create the full directory tree; write a minimal `main.py` that starts without error (no routes yet); write stub test files with the assertions that must pass by end of S01 (they should fail now).
  - Verify: `conda run -n rvc python -c "import aiosqlite, websockets, pytest"` exits 0; `conda run -n rvc uvicorn backend.app.main:app --port 8000 &` starts without error
  - Done when: Package imports succeed; directory tree exists; stub test files exist with correct assertions (currently failing); uvicorn starts on port 8000

- [x] **T02: Implement SQLite schema init and DB helpers** `est:30m`
  - Why: All profile CRUD routes depend on an initialized DB and reusable async helpers (get_db, row_to_dict); building these first keeps the route implementations thin.
  - Files: `backend/app/db.py`, `backend/app/main.py` (lifespan startup wires DB init)
  - Do: Implement `init_db()` async function that creates `backend/data/rvc.db` and the `profiles` table with the exact schema from the boundary contract; implement `get_db()` async context manager; wire `init_db()` into FastAPI lifespan; emit a structured startup log line with DB path and data directories; add `GET /health` returning `{"status":"ok","db_path":"...","rvc_root":"..."}`
  - Verify: `conda run -n rvc python -c "import asyncio; from backend.app.db import init_db; asyncio.run(init_db())"` succeeds; `sqlite3 backend/data/rvc.db ".schema"` shows the `profiles` table; `curl http://localhost:8000/health` returns `{"status":"ok",...}`
  - Done when: `backend/data/rvc.db` is created on startup with correct schema; `/health` endpoint responds

- [x] **T03: Implement voice profile CRUD routes and wire to DB** `est:45m`
  - Why: This is the core R001 delivery — POST/GET/DELETE profile endpoints backed by real SQLite data.
  - Files: `backend/app/routers/profiles.py`, `backend/app/main.py` (router registered), `backend/app/db.py` (any additional helpers)
  - Do: Implement `POST /api/profiles` (multipart: `name` field + `file` UploadFile); generate `uuid.uuid4().hex` as profile_id; `os.makedirs(f"data/samples/{profile_id}", exist_ok=True)`; stream file to disk with `aiofiles`; INSERT into profiles with `status="untrained"`; enforce 200MB upload limit (raise 413 if exceeded); implement `GET /api/profiles` (SELECT all, return list); implement `GET /api/profiles/{id}` (SELECT one, 404 if missing); implement `DELETE /api/profiles/{id}` (DELETE row + `shutil.rmtree` sample dir, 204 if found, 404 if missing); register profiles router in `main.py` with prefix `/api`
  - Verify: All `test_profiles.py` assertions pass via `conda run -n rvc pytest backend/tests/test_profiles.py -v`
  - Done when: All profile CRUD tests pass; a real uploaded file appears on disk under `data/samples/<id>/`; re-running GET returns the persisted profile

- [x] **T04: Implement devices endpoint and complete contract test suite** `est:30m`
  - Why: `GET /api/devices` is required by S03's boundary contract; the test suite must be fully green before S01 can close.
  - Files: `backend/app/routers/devices.py`, `backend/app/main.py` (router registered), `backend/tests/test_devices.py`
  - Do: Implement `GET /api/devices` by calling `sounddevice.query_devices()` synchronously (safe, <1ms); map each device to `{id: int, name: str, is_input: bool, is_output: bool}` using `max_input_channels > 0` and `max_output_channels > 0`; register devices router; fill in `test_devices.py` with assertions that response is a non-empty list and every entry has the required keys; run the full test suite and confirm all pass
  - Verify: `conda run -n rvc pytest backend/tests/ -v` all green; `curl http://localhost:8000/api/devices` response contains an entry with `"name"` matching `"BlackHole"`
  - Done when: Full pytest suite passes; `/api/devices` returns sounddevice data including BlackHole 2ch

## Files Likely Touched

- `backend/app/__init__.py`
- `backend/app/main.py`
- `backend/app/db.py`
- `backend/app/routers/__init__.py`
- `backend/app/routers/profiles.py`
- `backend/app/routers/devices.py`
- `backend/tests/__init__.py`
- `backend/tests/test_profiles.py`
- `backend/tests/test_devices.py`
- `backend/data/rvc.db` (runtime artifact)
- `data/samples/` (runtime artifact directory)
- `.env.example`
