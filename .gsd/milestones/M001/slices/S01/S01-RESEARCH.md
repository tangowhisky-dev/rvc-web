# S01: Backend Foundation & Voice Sample Library — Research

**Date:** 2026-03-14

## Summary

S01 builds the FastAPI backend skeleton inside `rvc-web/backend/` and wires up a SQLite-backed REST API for voice profile management (create/list/get/delete) with file upload. This is a pure application layer with no RVC inference—the only RVC dependency explored here is confirming import mechanics and env var requirements so S02/S03 don't break.

The environment is in good shape: FastAPI 0.135.1, uvicorn 0.41.0, python-multipart, aiofiles, and httpx are all present in the `rvc` conda env. The two missing packages are **`aiosqlite`** and **`websockets`** — both must be pip-installed before any async DB or WebSocket code runs. Starlette's built-in WebSocket support works without the `websockets` package for server-side use, so the WebSocket gap only matters if a test client needs it.

The project directory (`rvc-web/`) currently has only the `Retrieval-based-Voice-Conversion-WebUI/` subdirectory — no `backend/` or `frontend/` scaffolding exists yet. The slice must create the full directory tree from scratch.

## Recommendation

Build the backend as a standalone FastAPI app at `rvc-web/backend/app/main.py`, running from `rvc-web/` as CWD (not from inside the RVC repo). Use `RVC_ROOT` env var to point at the RVC repo — only needed by S02/S03, not S01. Install `aiosqlite` and `websockets` at the top of S01 execution before writing any async code. Use `aiosqlite` directly with raw SQL (no ORM) — the schema is simple and SQLAlchemy adds friction with no benefit here. Store the SQLite file at `backend/data/rvc.db` (writable relative path). Store uploaded audio under `data/samples/<profile_id>/`.

For the devices endpoint (`GET /api/devices`), call `sounddevice.query_devices()` synchronously and return the full list — no background threading needed.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Async SQLite | `aiosqlite` | Already decided (D001); avoids blocking the event loop |
| File upload handling | `fastapi.UploadFile` + `aiofiles.open` | `python-multipart` is already installed; `aiofiles` is already installed |
| CORS for Next.js | `fastapi.middleware.cors.CORSMiddleware` | One-line setup; no external package needed |
| Audio device listing | `sounddevice.query_devices()` | Already installed; returns full device metadata dict |
| WebSocket (server) | Starlette built-in `fastapi.WebSocket` | No `websockets` package needed server-side; confirmed working in env |

## Existing Code and Patterns

- `Retrieval-based-Voice-Conversion-WebUI/.env` — Defines `rmvpe_root`, `weight_root`, `index_root`. The backend must load this file (or replicate the vars) before any RVC import. For S01, no RVC imports happen, so this is only a forward concern.
- `Retrieval-based-Voice-Conversion-WebUI/configs/config.py` — `Config` singleton calls `arg_parse()` which reads `sys.argv`; calling it from a FastAPI route would blow up. **Never instantiate `Config` from the FastAPI process directly** — it must be run as a subprocess or from a dedicated worker process. This does not affect S01 (no training/inference).
- `Retrieval-based-Voice-Conversion-WebUI/infer/lib/rtrvc.py` — Imports `fairseq`, `faiss`, `torch`, reads `os.environ["rmvpe_root"]`. Loads `assets/hubert/hubert_base.pt` via a relative path — so any process that imports `rtrvc` must have CWD = `Retrieval-based-Voice-Conversion-WebUI/`. For S01 we do **not** import rtrvc; document this constraint for S02.
- `Retrieval-based-Voice-Conversion-WebUI/web.py` — Shows the subprocess invocation pattern for `preprocess.py`, including exact arg order: `inp_root sr n_p exp_dir noparallel per`. Useful reference for S02.
- `Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2/` — Contains `f0G48k.pth`, `f0D48k.pth`, etc. **Never write here.** Fine-tuned output goes to `assets/weights/`.
- `Retrieval-based-Voice-Conversion-WebUI/logs/` — Training artifacts dir. Already has a `mute/` subdirectory from prior runs. Our training exp dirs will land here as `logs/<profile_id>/`.

## Constraints

- `aiosqlite` and `websockets` are not installed in the `rvc` env — must `pip install` before any code that uses them.
- `pytest` is also not installed — needed for contract verification tests. Install alongside `aiosqlite`.
- FastAPI's `TestClient` (via `httpx`) is available — use it for synchronous tests without needing a running server.
- `Config` singleton in `configs/config.py` calls `arg_parse()` at construction — cannot be imported from within the FastAPI app without poisoning `sys.argv`. Only import RVC modules from subprocess workers or a dedicated process that owns `sys.argv`.
- `rtrvc.py` must be imported with CWD = `Retrieval-based-Voice-Conversion-WebUI/` and `rmvpe_root` env var set. S01 does not import it, but the FastAPI app must support an `RVC_ROOT` env var for S02/S03 to use.
- `sounddevice` works from the `rvc` env on macOS — confirmed: device 1 = BlackHole 2ch, device 2 = MacBook Pro Microphone. Device indices can shift if system audio config changes; always query dynamically.
- CORS must allow `http://localhost:3000` (Next.js dev server).
- File uploads must be saved asynchronously using `aiofiles` to avoid blocking the event loop.
- SQLite DB path must be a stable, relative-to-CWD path — store at `backend/data/rvc.db` and create the directory on startup.

## Common Pitfalls

- **`Config` singleton import** — Importing `from configs.config import Config` anywhere inside the FastAPI process triggers `arg_parse()` which reads `sys.argv` and will interpret uvicorn args as its own flags. Never do this in the FastAPI app. Keep RVC imports isolated to subprocess workers.
- **`aiosqlite` connection as a module-level singleton** — aiosqlite connections are not thread-safe for concurrent writes; use `async with aiosqlite.connect(DB_PATH) as db` per request for writes, or a single shared connection with proper locking. For this single-user tool, per-request connections are simplest and safe.
- **`UploadFile` not consumed before response** — Must `await file.read()` or stream via `aiofiles` before returning the response. FastAPI closes the upload stream after the response is sent.
- **`profile_id` as UUID** — Use `uuid.uuid4().hex` (no dashes) as the profile ID to keep directory names safe on all filesystems.
- **Missing `data/samples/<profile_id>/` directory** — Must `os.makedirs(..., exist_ok=True)` before writing the uploaded file. Failure to do so raises `FileNotFoundError` with no helpful message to the API consumer.
- **Blocking sounddevice call on async route** — `sounddevice.query_devices()` is synchronous but fast (<1ms). It's safe to call directly from an async route without `run_in_executor`. For long-running audio ops in S03, use `asyncio.get_event_loop().run_in_executor(None, ...)`.
- **Config.load_config_json uses `configs/inuse/`** — The `inuse/` subdir already exists and is populated. If the backend's CWD is `rvc-web/` (not inside the RVC repo), this path won't resolve. Confirm the backend's CWD is always `rvc-web/` and that RVC subprocesses are launched with `cwd=RVC_ROOT`.

## Open Risks

- **`aiosqlite` install may conflict** — `pip install aiosqlite websockets pytest pytest-asyncio` into the `rvc` env has a small chance of version conflicts with the large conda env. Run the install early and verify imports before writing app code.
- **`websockets` package version** — FastAPI/Starlette's WebSocket support uses the `websockets` library internally for the test client but not for the server itself. The `TestClient.websocket_connect()` call requires `websockets` installed. Check that the installed version is compatible with Starlette 0.52.1.
- **SQLite concurrency during training** — S02 will update profile `status` from a background task while S01 routes are reading. With `aiosqlite`, concurrent readers are fine; a write lock is held briefly. For this single-user app this is acceptable, but if the training worker is a subprocess (not a coroutine), it needs its own `aiosqlite` connection — or communicate status updates back to the FastAPI process via a queue/event.
- **File size limits** — FastAPI has no built-in upload size limit. A 30-minute voice sample at 48kHz WAV could be ~500MB. Add a reasonable limit (e.g. 200MB) or stream to disk in chunks to avoid memory exhaustion.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| FastAPI | (none checked) | Built into GSD context; no external skill needed |
| Next.js | (none checked) | S04 concern; not needed for S01 |
| aiosqlite | (none checked) | Simple enough; official docs sufficient |

## Sources

- Confirmed `aiosqlite` and `websockets` missing from `rvc` env via `conda run -n rvc pip list`
- Confirmed `fastapi 0.135.1`, `uvicorn 0.41.0`, `python-multipart`, `aiofiles`, `httpx` present
- Confirmed FastAPI WebSocket works server-side without `websockets` package (Starlette built-in)
- Confirmed `sounddevice.query_devices()` returns: 0=iPhone Mic, 1=BlackHole 2ch, 2=MacBook Pro Mic, 3=MacBook Pro Speakers, 4=Microsoft Teams Audio
- Confirmed `rtrvc.py` uses `os.environ["rmvpe_root"]` and relative path `assets/hubert/hubert_base.pt` — must run from RVC repo root
- Confirmed `configs/config.py` `Config` singleton calls `arg_parse()` at init — unsafe to import from FastAPI process
- Confirmed `configs/inuse/` directory exists and is populated
- Confirmed `assets/rmvpe/rmvpe.pt` and `rmvpe.onnx` present
- Confirmed `assets/pretrained_v2/` has `f0G48k.pth`, `f0D48k.pth` — base models intact
- Confirmed `assets/weights/` directory exists (empty target for fine-tuned models)
- Confirmed `logs/mute/` exists — training artifact directory pattern established
