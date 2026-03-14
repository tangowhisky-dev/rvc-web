---
estimated_steps: 5
estimated_files: 10
---

# T01: Install missing packages and create backend directory skeleton

**Slice:** S01 — Backend Foundation & Voice Sample Library
**Milestone:** M001

## Description

The `rvc` conda environment is missing `aiosqlite`, `websockets`, `pytest`, and `pytest-asyncio` — none of the async DB code, WebSocket routes, or contract tests can run without them. This task installs those packages first, then creates the full `backend/` directory tree and writes stub test files that encode the exact assertions the slice must satisfy. The stubs should fail at this point — that is intentional and correct. A minimal `main.py` is written so uvicorn can start without errors before any routes are implemented.

## Steps

1. Run `conda run -n rvc pip install aiosqlite websockets pytest pytest-asyncio` and verify with `conda run -n rvc python -c "import aiosqlite, websockets, pytest, pytest_asyncio; print('ok')"`.
2. Create the full directory tree: `backend/app/`, `backend/app/routers/`, `backend/tests/`, `backend/data/` — all with `__init__.py` where needed.
3. Write `backend/app/main.py` as a minimal FastAPI app (no routes yet) with CORS middleware allowing `http://localhost:3000`, and an empty lifespan handler — enough to start uvicorn cleanly.
4. Write `backend/tests/test_profiles.py` with the full set of expected assertions for all profile CRUD endpoints (these will fail until T03 implements the routes). Include: POST creates profile and file on disk, GET lists it, GET by id returns it, DELETE removes it and returns 204, DELETE unknown id returns 404, POST without file returns 422.
5. Write `backend/tests/test_devices.py` with assertions that `GET /api/devices` returns a non-empty list with `id`, `name`, `is_input`, `is_output` keys (will fail until T04).
6. Write `.env.example` documenting `RVC_ROOT` and `DATABASE_URL` for forward compatibility.

## Must-Haves

- [ ] `aiosqlite`, `websockets`, `pytest`, `pytest-asyncio` importable in `rvc` env
- [ ] `backend/app/main.py` starts with uvicorn without errors
- [ ] `backend/tests/test_profiles.py` exists with concrete assertions (currently failing)
- [ ] `backend/tests/test_devices.py` exists with concrete assertions (currently failing)
- [ ] `.env.example` documents `RVC_ROOT`

## Verification

- `conda run -n rvc python -c "import aiosqlite, websockets, pytest, pytest_asyncio; print('ok')"` → prints `ok`
- `conda run -n rvc pytest backend/tests/ --co -q` → shows collected tests (no import errors, some failures expected)
- `cd /Users/tango16/code/rvc-web && conda run -n rvc uvicorn backend.app.main:app --port 8000 --host 0.0.0.0` starts without crashing

## Observability Impact

- Signals added/changed: None yet (stub only)
- How a future agent inspects this: `conda run -n rvc pytest backend/tests/ --co -q` lists all planned test assertions
- Failure state exposed: Import errors from missing packages will surface immediately on test collection; stub failures confirm tests are wired

## Inputs

- Research findings: `aiosqlite`, `websockets`, `pytest`, `pytest-asyncio` are absent from `rvc` env
- Boundary contract from `M001-ROADMAP.md` S01→S02 section: exact API shape, status values, field names
- `DECISIONS.md` D001 (aiosqlite), D002 (FastAPI), D012 (project layout)

## Expected Output

- `backend/app/__init__.py` (empty)
- `backend/app/main.py` (minimal FastAPI app, CORS configured, lifespan stub)
- `backend/app/routers/__init__.py` (empty)
- `backend/tests/__init__.py` (empty)
- `backend/tests/test_profiles.py` (full assertion stubs — failing)
- `backend/tests/test_devices.py` (full assertion stubs — failing)
- `.env.example` (documents `RVC_ROOT`, `DATABASE_URL`)
- All four packages importable in `rvc` env
