---
estimated_steps: 6
estimated_files: 4
---

# T03: Implement voice profile CRUD routes and wire to DB

**Slice:** S01 — Backend Foundation & Voice Sample Library
**Milestone:** M001

## Description

This task implements the four voice profile endpoints that form the R001 core: POST (upload + create), GET list, GET by id, and DELETE. Each route uses `get_db()` from `db.py`, saves audio to `data/samples/<profile_id>/` via `aiofiles`, and enforces a 200MB upload limit. The router is registered in `main.py` so the routes are reachable. After this task the `test_profiles.py` test suite must be fully green.

## Steps

1. Implement `backend/app/routers/profiles.py` with an `APIRouter(prefix="/api/profiles")`:
   - `POST /` — multipart body with `name: str` form field and `file: UploadFile`; generate `profile_id = uuid.uuid4().hex`; enforce 200MB limit (check `file.size` or stream with counter, raise `HTTPException(413)` if exceeded); `os.makedirs(f"data/samples/{profile_id}", exist_ok=True)`; stream file to disk with `aiofiles.open(..., "wb")`; `created_at = datetime.utcnow().isoformat()`; INSERT into profiles; return `ProfileOut` (id, name, status, created_at, sample_path) with status 201.
   - `GET /` — SELECT all from profiles, return list of `ProfileOut`.
   - `GET /{profile_id}` — SELECT by id; raise `HTTPException(404)` if not found; return `ProfileOut`.
   - `DELETE /{profile_id}` — SELECT to confirm existence (404 if missing); DELETE from profiles; `shutil.rmtree(f"data/samples/{profile_id}", ignore_errors=True)`; return `Response(status_code=204)`.
2. Define a `ProfileOut` Pydantic model (or TypedDict) with fields: `id`, `name`, `status`, `created_at`, `sample_path`.
3. Register the profiles router in `backend/app/main.py`: `app.include_router(profiles_router)`.
4. Use FastAPI `TestClient` in `test_profiles.py` — no running server needed. Override `lifespan` so `init_db()` uses a temp SQLite DB (via `tmp_path` fixture or `TEST_DB_PATH` env var override) to keep test isolation.
5. Ensure the test for POST verifies the file actually exists on disk at the returned `sample_path`.
6. Run full test suite and confirm `test_profiles.py` passes.

## Must-Haves

- [ ] `POST /api/profiles` creates a DB row and writes the audio file to `data/samples/<id>/`
- [ ] File upload streamed to disk with `aiofiles` (not `file.read()` into memory for large files)
- [ ] 200MB upload limit enforced — returns HTTP 413 if exceeded
- [ ] `profile_id` is `uuid.uuid4().hex` (no dashes)
- [ ] `GET /api/profiles` returns all profiles as a list
- [ ] `GET /api/profiles/{id}` returns 404 for unknown id
- [ ] `DELETE /api/profiles/{id}` removes DB row AND `data/samples/<id>/` directory; returns 204
- [ ] `DELETE /api/profiles/{id}` returns 404 for unknown id
- [ ] Test isolation: tests use a temp DB, not the production `backend/data/rvc.db`
- [ ] All `test_profiles.py` assertions pass

## Verification

- `conda run -n rvc pytest backend/tests/test_profiles.py -v` → all tests green
- Manual spot check: POST a real file, then `ls data/samples/` to confirm directory and file exist on disk
- `sqlite3 backend/data/rvc.db "SELECT * FROM profiles;"` (after a manual curl POST) shows the inserted row

## Observability Impact

- Signals added/changed: Per-operation errors logged with `profile_id` and operation name via Python `logging`; HTTPException `detail` field includes `profile_id` for all 404s
- How a future agent inspects this: `sqlite3 backend/data/rvc.db "SELECT id, name, status, created_at FROM profiles;"` to see all profiles; `ls data/samples/` to verify file storage; GET `/api/profiles` endpoint
- Failure state exposed: 404 detail names the missing `profile_id`; 413 detail names the size limit; file write failures surface as 500 with the OS error message

## Inputs

- `backend/app/db.py` — `get_db()`, `DB_PATH`, `init_db()` from T02
- `backend/app/main.py` — lifespan, CORS, stub router registration point from T02
- `backend/tests/test_profiles.py` — stub test file with assertion skeletons from T01

## Expected Output

- `backend/app/routers/profiles.py` — full CRUD router implementation
- `backend/app/main.py` — profiles router registered
- `backend/tests/test_profiles.py` — all assertions implemented and passing
- Real audio files stored under `data/samples/<profile_id>/` during test runs (in temp dir)
