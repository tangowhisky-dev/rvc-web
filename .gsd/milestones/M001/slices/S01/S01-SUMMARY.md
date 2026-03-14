---
id: S01
parent: M001
milestone: M001
provides:
  - FastAPI backend (backend/app/main.py) with CORS for localhost:3000 and lifespan startup
  - SQLite DB (backend/data/rvc.db) initialized on startup with 7-column profiles schema
  - GET /health endpoint — liveness + config inspection
  - POST /api/profiles — multipart upload, 200MB limit, aiofiles streaming, DB persistence
  - GET /api/profiles — list all profiles
  - GET /api/profiles/{id} — single profile or 404
  - DELETE /api/profiles/{id} — removes DB row + sample directory, 204/404
  - GET /api/devices — sounddevice.query_devices() mapped to {id, name, is_input, is_output}
  - backend/tests/ contract test suite — 17/17 passing
  - .env.example documenting RVC_ROOT, DATABASE_URL, DATA_DIR
requires: []
affects:
  - S02
  - S03
key_files:
  - backend/app/main.py
  - backend/app/db.py
  - backend/app/routers/profiles.py
  - backend/app/routers/devices.py
  - backend/tests/conftest.py
  - backend/tests/test_profiles.py
  - backend/tests/test_devices.py
  - .env.example
key_decisions:
  - D013 — aiosqlite per-request context manager (no shared pool)
  - D014 — TestClient + temp SQLite DB + DATA_DIR env var for hermetic test isolation
  - D015 — Profile IDs as uuid.uuid4().hex (32 hex chars, no dashes)
  - D016 — JSON startup log + /health endpoint for agent-readable observability
patterns_established:
  - get_db() async context manager sets row_factory=aiosqlite.Row for named column access
  - init_db() is idempotent via CREATE TABLE IF NOT EXISTS + os.makedirs(exist_ok=True)
  - Startup emits a single structured JSON line to stdout (event, db_path, rvc_root, data_dir)
  - Test isolation via monkeypatch on db_module.DB_PATH + DATA_DIR env var — no lifespan override needed
  - aiofiles 64KiB chunked streaming with mid-stream size check and cleanup on limit breach
  - anyio fixtures (not asyncio) to avoid asyncio-mode=STRICT event loop conflicts
  - _samples_root() helper in profiles.py reads DATA_DIR; falls back to "data/samples" in production
  - DeviceOut Pydantic model enforces response schema; enumerate() index used as stable device ID
observability_surfaces:
  - GET /health → {"status":"ok","db_path":"...","rvc_root":"..."} — liveness + config check
  - stdout JSON startup log on every lifespan startup with db_path, rvc_root, data_dir
  - sqlite3 backend/data/rvc.db ".schema profiles" — schema inspection
  - sqlite3 backend/data/rvc.db "SELECT id, name, status, created_at FROM profiles;" — live profile inventory
  - ls data/samples/ — confirm uploaded file directories
  - GET /api/devices — real-time OS audio device list; verify BlackHole 2ch present before S03
  - GET /api/profiles — machine-readable profile list at any time
drill_down_paths:
  - .gsd/milestones/M001/slices/S01/tasks/T01-SUMMARY.md
  - .gsd/milestones/M001/slices/S01/tasks/T02-SUMMARY.md
  - .gsd/milestones/M001/slices/S01/tasks/T03-SUMMARY.md
  - .gsd/milestones/M001/slices/S01/tasks/T04-SUMMARY.md
duration: ~2 hours (4 tasks)
verification_result: passed
completed_at: 2026-03-14
---

# S01: Backend Foundation & Voice Sample Library

**FastAPI server initialized with SQLite schema, full voice profile CRUD (upload, list, get, delete), and GET /api/devices — all 17 contract tests passing; curl-verified against a live server.**

## What Happened

S01 was executed across four sequential tasks. T01 confirmed the four missing Python packages (`aiosqlite`, `websockets`, `pytest`, `pytest-asyncio`) were already installed from a prior partial attempt, created the directory skeleton, and wrote 17 contract test stubs (10 profiles, 7 devices) that served as the acceptance criteria for the remaining tasks.

T02 built the database layer: `backend/app/db.py` with `init_db()` (idempotent schema creation), `get_db()` async context manager with `row_factory=aiosqlite.Row`, and lifespan wiring in `main.py`. Added `GET /health` returning db_path and rvc_root. The `backend/data/rvc.db` SQLite file is now created automatically on every server startup.

T03 implemented the four voice profile CRUD endpoints in `backend/app/routers/profiles.py`. File uploads stream in 64KiB chunks via `aiofiles` with a running byte counter; files exceeding 200MB are rejected with 413 and the partial file is cleaned up. A `conftest.py` fixture achieved per-test DB isolation by redirecting `db_module.DB_PATH` via monkeypatch and redirecting uploads via a `DATA_DIR` env var — the production `backend/data/rvc.db` is never touched during tests. All 10 profile contract tests passed.

T04 added `backend/app/routers/devices.py` with `GET /api/devices`, mapping `sounddevice.query_devices()` to a list of `DeviceOut` models. The 7 device test stubs written in T01 passed immediately. Full suite: 17/17 green.

## Verification

All slice-level verification checks passed against a live uvicorn server:

```
# Full test suite
conda run -n rvc pytest backend/tests/ -v
# → 17 passed in 0.19s

# Health check
curl -s http://localhost:8000/health
# → {"status":"ok","db_path":"backend/data/rvc.db","rvc_root":"not set"}

# Profile upload
curl -s -X POST -F "name=test_voice" -F "file=@/tmp/test.wav" http://localhost:8000/api/profiles
# → {"id":"4a444ac8...","name":"test_voice","status":"untrained","created_at":"2026-03-14T17:26:23Z","sample_path":"data/samples/4a444ac8.../test.wav"}

# Profile list
curl -s http://localhost:8000/api/profiles
# → [{"id":"4a444ac8...",...}]

# Device list (BlackHole 2ch confirmed at id=1)
curl -s http://localhost:8000/api/devices
# → [{id:0,"iPhone Microphone",...}, {id:1,"BlackHole 2ch",is_input:true,is_output:true}, ...]
```

## Requirements Advanced

- R001 (Voice Sample Library) — fully implemented: upload, list, get, delete profiles; SQLite persistence; file storage at `data/samples/<profile_id>/`; status field present

## Requirements Validated

- R001 — validated by: 10 passing contract tests + live curl verification (upload, list, delete confirmed end-to-end with real SQLite writes and disk I/O)

## New Requirements Surfaced

- none

## Requirements Invalidated or Re-scoped

- none

## Deviations

- `POST /api/profiles` returns HTTP 200 instead of 201. The test stubs written in T01 asserted 200, so the route was implemented to match. This is cosmetic (200 vs 201 has no behavioral impact); the deviation is recorded but not treated as a defect.
- `backend/app/db.py` was not created by T01 (the T01 plan listed it as a stub to be created; T02 created it from scratch instead). No impact — T02 was explicitly designed to own the DB layer.

## Known Limitations

- `RVC_ROOT` is loaded from environment but not validated at startup — if missing, S02/S03 features will fail at runtime rather than at startup. This is intentional (S01 doesn't use RVC_ROOT).
- The 200MB upload limit is enforced in-process during streaming, not via a Nginx/proxy-level limit. This is acceptable for a local single-user app but would be wrong for a shared deployment.
- `GET /api/devices` calls `sounddevice.query_devices()` synchronously on the async event loop thread. This is confirmed safe (<1ms, no blocking I/O) for a local tool with a single user; not suitable for high-concurrency deployments.

## Follow-ups

- S02 must install `aiofiles` into the `rvc` env (needed for streaming in profiles.py — verify it's present).
- S03 should verify the device ID returned by `GET /api/devices` for BlackHole 2ch (id=1) matches the PortAudio index expected by `sounddevice.OutputStream`. Device IDs are stable for a given OS audio configuration but should be confirmed live.
- A `RETURNING` clause on the SQLite INSERT would allow removing the SELECT-after-INSERT pattern in `POST /api/profiles`, but this is a minor optimization — defer to a later refactor pass.

## Files Created/Modified

- `backend/app/__init__.py` — package init
- `backend/app/main.py` — FastAPI app with CORS, lifespan (init_db + startup log), GET /health, router registrations
- `backend/app/db.py` — DB_PATH constant, init_db(), get_db() async context manager
- `backend/app/routers/__init__.py` — package init
- `backend/app/routers/profiles.py` — POST/GET/GET-by-id/DELETE /api/profiles with aiofiles streaming + structured logging
- `backend/app/routers/devices.py` — GET /api/devices with DeviceOut Pydantic model
- `backend/tests/__init__.py` — package init
- `backend/tests/conftest.py` — async ASGI client fixture with monkeypatched DB and DATA_DIR isolation
- `backend/tests/test_profiles.py` — 10 profile contract tests (all passing)
- `backend/tests/test_devices.py` — 7 device contract tests (all passing)
- `backend/data/rvc.db` — runtime artifact: SQLite DB with profiles table
- `data/samples/` — runtime artifact: directory for uploaded audio files
- `.env.example` — documents RVC_ROOT, DATABASE_URL, DATA_DIR

## Forward Intelligence

### What the next slice should know

- The profiles router uses `DATA_DIR` env var to locate the samples root. In production this defaults to `"data/samples"` (relative to CWD = `rvc-web/`). S02's training subprocess will need to know this path when locating the uploaded audio file for preprocessing — read `profile.sample_path` from the DB, it's always relative to `rvc-web/`.
- `RVC_ROOT` is read via `os.environ.get("RVC_ROOT", "")` in `main.py`. S02 must set this env var before starting the server (or export it in the shell) — the training subprocesses depend on it to locate `infer/modules/train/preprocess.py` etc.
- Profile status field values in the DB are plain strings: `"untrained"`, `"training"`, `"trained"`, `"failed"`. No Enum — S02 can UPDATE the status column directly with a simple SQL statement.
- The `get_db()` context manager does not hold connections open across requests. Each DB operation (INSERT, SELECT, DELETE) opens and closes its own connection. This is fine for CRUD but S02's training status updates (which may fire mid-subprocess) should open their own connection via `get_db()` in a short async block.

### What's fragile

- `aiofiles` streaming assumes the process has write permission to `data/samples/`. If the directory is missing or read-only, the server will return HTTP 500 with the OSError message. The `init_db()` startup call creates `data/samples/` but only if it doesn't exist — if it gets deleted while the server is running, the next upload will fail.
- `sounddevice.query_devices()` can raise `PortAudioError` if PortAudio is not initialized (e.g. running headless without an audio subsystem). In tests this is mocked by the test fixture; in a production-like CI environment without audio hardware it would need a mock.

### Authoritative diagnostics

- `conda run -n rvc pytest backend/tests/ -v` — fastest end-to-end verification; run after any backend change
- `sqlite3 backend/data/rvc.db "SELECT id, name, status FROM profiles;"` — live profile state, always accurate
- `curl -s http://localhost:8000/health` — confirms server is up and reports RVC_ROOT config
- `curl -s http://localhost:8000/api/devices` — confirms BlackHole 2ch device ID before S03 configuration

### What assumptions changed

- Packages were already installed from a prior partial attempt — T01 did not need to `pip install` anything. No impact on the plan, just a faster T01.
- T01 plan described `backend/app/db.py` as a "(stub)" to be created in T01, but the file was not written in T01 and was cleanly owned by T02 instead. No gap — just a minor planning terminology ambiguity.
