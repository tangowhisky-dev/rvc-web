---
id: T02
parent: S01
milestone: M001
provides:
  - backend/app/db.py with init_db(), get_db(), and DB_PATH
  - FastAPI lifespan wired to init_db() with structured JSON startup log
  - GET /health endpoint returning status, db_path, rvc_root
  - backend/data/rvc.db created with correct 7-column profiles schema
  - data/samples/ directory created on startup
key_files:
  - backend/app/db.py
  - backend/app/main.py
key_decisions:
  - D013 (aiosqlite per-request context manager) already recorded
  - D016 (JSON startup log + /health) already recorded
patterns_established:
  - get_db() async context manager sets row_factory=aiosqlite.Row for named column access
  - init_db() is idempotent via CREATE TABLE IF NOT EXISTS + os.makedirs(exist_ok=True)
  - Startup log is a single JSON line to stdout (event, db_path, rvc_root, data_dir)
observability_surfaces:
  - GET /health → {"status":"ok","db_path":"backend/data/rvc.db","rvc_root":"..."}
  - stdout JSON startup log on every lifespan startup
  - sqlite3 backend/data/rvc.db ".schema profiles" for schema inspection
duration: ~15m
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T02: Implement SQLite schema init and DB helpers

**Wrote `backend/app/db.py` with `init_db()` and `get_db()`; wired lifespan in `main.py`; added `GET /health`; DB and directories created correctly on startup.**

## What Happened

Created `backend/app/db.py` from scratch (the T01 skeleton did not include it). Implemented:

- `DB_PATH = "backend/data/rvc.db"` constant
- `init_db()`: creates `backend/data/` and `data/samples/` directories (exist_ok=True), then connects to SQLite and runs `CREATE TABLE IF NOT EXISTS profiles (...)` with all 7 required columns, then commits
- `get_db()`: async context manager using `@asynccontextmanager` that opens an `aiosqlite.connect(DB_PATH)` connection, sets `row_factory = aiosqlite.Row`, and yields it — per D013

Updated `backend/app/main.py`:
- Added import of `init_db` and `DB_PATH` from `backend.app.db`
- Lifespan now `await init_db()` on startup, then emits `json.dumps({event, db_path, rvc_root, data_dir})` to stdout
- Added `GET /health` route returning `{"status":"ok","db_path":DB_PATH,"rvc_root":os.environ.get("RVC_ROOT","not set")}`

## Verification

All T02 must-haves verified:

```
# init_db() runs and prints ok
conda run -n rvc python -c "import asyncio; from backend.app.db import init_db; asyncio.run(init_db()); print('ok')"
→ ok

# Schema correct (all 7 columns)
sqlite3 backend/data/rvc.db ".schema profiles"
→ CREATE TABLE profiles (id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'untrained',
    sample_path TEXT, model_path TEXT, index_path TEXT, created_at TEXT NOT NULL)

# Idempotent — second call does not error
conda run -n rvc python -c "... asyncio.run(init_db()); asyncio.run(init_db()); print('idempotent ok')"
→ idempotent ok

# Server starts and /health responds
uvicorn backend.app.main:app --port 8000 + curl -s http://localhost:8000/health
→ {"status":"ok","db_path":"backend/data/rvc.db","rvc_root":"not set"}
```

Partial slice verification:
- 2/17 tests pass (the 2 that don't depend on T03/T04 routes)
- 15 tests still fail with 404 — expected, routes not yet implemented

## Diagnostics

- `GET /health` → liveness + config check
- `sqlite3 backend/data/rvc.db ".schema"` → verify schema
- Startup stdout line: `{"event":"startup","db_path":"backend/data/rvc.db","rvc_root":"not set","data_dir":"data/samples"}`
- Missing write permission on `backend/data/` will raise `PermissionError` at startup before any request is served

## Deviations

None. The T01 skeleton did not produce `backend/app/db.py` (only a stub comment in `main.py`), but this was expected per the T01 description ("db.py (stub)" — the file was not actually written). T02 created it from scratch as planned.

## Known Issues

None.

## Files Created/Modified

- `backend/app/db.py` — new file: `DB_PATH`, `init_db()`, `get_db()` async context manager
- `backend/app/main.py` — updated: lifespan wires `init_db()`, JSON startup log, `GET /health` endpoint added
- `backend/data/rvc.db` — runtime artifact: SQLite DB with profiles table created
- `data/samples/` — runtime artifact: directory created by `init_db()`
