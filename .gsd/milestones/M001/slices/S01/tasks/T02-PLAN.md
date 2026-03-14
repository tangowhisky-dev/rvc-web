---
estimated_steps: 4
estimated_files: 3
---

# T02: Implement SQLite schema init and DB helpers

**Slice:** S01 — Backend Foundation & Voice Sample Library
**Milestone:** M001

## Description

Before any route can read or write profiles, the SQLite database must be initialized with the correct schema and a reusable async DB connection helper must exist. This task wires `init_db()` into the FastAPI lifespan so the DB and required directories are created automatically when the server starts. A `GET /health` endpoint is added to give future agents a cheap liveness check that also reports current configuration (DB path, RVC_ROOT). A structured startup log line is emitted so the operator can confirm the server is properly configured at boot.

## Steps

1. Write `backend/app/db.py` with:
   - `DB_PATH = "backend/data/rvc.db"` constant
   - `init_db()` async function: `os.makedirs("backend/data", exist_ok=True)`, `os.makedirs("data/samples", exist_ok=True)`, then `async with aiosqlite.connect(DB_PATH) as db: await db.execute(CREATE TABLE IF NOT EXISTS profiles ...)` with the exact schema from the boundary contract
   - `get_db()` async context manager that yields an `aiosqlite.Connection` with `row_factory = aiosqlite.Row`
2. Update `backend/app/main.py` lifespan to `await init_db()` on startup; emit a JSON startup log line to stdout with keys `event`, `db_path`, `rvc_root`, `data_dir`.
3. Add `GET /health` route directly in `main.py` returning `{"status":"ok","db_path": DB_PATH, "rvc_root": os.environ.get("RVC_ROOT","not set")}`.
4. Confirm DB schema is created correctly by running the schema introspection command.

## Must-Haves

- [ ] `profiles` table created with columns: `id TEXT PRIMARY KEY`, `name TEXT NOT NULL`, `status TEXT NOT NULL DEFAULT 'untrained'`, `sample_path TEXT`, `model_path TEXT`, `index_path TEXT`, `created_at TEXT NOT NULL`
- [ ] `init_db()` is idempotent (`CREATE TABLE IF NOT EXISTS`)
- [ ] `backend/data/` and `data/samples/` directories created on startup
- [ ] `GET /health` returns `{"status":"ok","db_path":"...","rvc_root":"..."}`
- [ ] Structured startup log line emitted to stdout

## Verification

- `conda run -n rvc python -c "import asyncio; from backend.app.db import init_db; asyncio.run(init_db()); print('ok')"` → prints `ok`
- `sqlite3 backend/data/rvc.db ".schema profiles"` → shows the correct CREATE TABLE statement with all 7 columns
- Start server: `conda run -n rvc uvicorn backend.app.main:app --port 8000 &` then `curl -s http://localhost:8000/health` → `{"status":"ok",...}`
- Run twice: second `init_db()` call does not error (idempotent)

## Observability Impact

- Signals added/changed: Structured JSON startup log line on every server boot (event, db_path, rvc_root, data_dir)
- How a future agent inspects this: `GET /health` for liveness; `sqlite3 backend/data/rvc.db ".schema"` for schema verification; startup stdout for config confirmation
- Failure state exposed: Missing `backend/data/` write permissions will raise on startup (before any request); missing `RVC_ROOT` is visible in `/health` response as `"not set"` rather than a silent misconfiguration

## Inputs

- `backend/app/main.py` — minimal stub from T01 (uvicorn starts, CORS configured, empty lifespan)
- Boundary contract schema: `profiles(id TEXT PK, name TEXT, status TEXT, sample_path TEXT, model_path TEXT, index_path TEXT, created_at TEXT)` from `M001-ROADMAP.md`
- `DECISIONS.md` D001 (aiosqlite per-request connections for simplicity)

## Expected Output

- `backend/app/db.py` — `init_db()`, `get_db()`, `DB_PATH`
- `backend/app/main.py` — lifespan wires `init_db()`, `/health` endpoint, startup log
- `backend/data/rvc.db` created on server start with correct schema
- `data/samples/` directory created on server start
