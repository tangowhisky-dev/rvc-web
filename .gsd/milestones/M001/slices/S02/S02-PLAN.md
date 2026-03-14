# S02: Training Pipeline with Live Log Streaming

**Goal:** POST to start training for a profile, watch preprocess → feature extract → train logs stream via WebSocket in a test client, and find a real `.pth` in `assets/weights/` when training completes.

**Demo:** `POST /api/training/start {"profile_id": "..."}` triggers a four-phase subprocess chain; a WebSocket client connecting to `ws://localhost:8000/ws/training/<profile_id>` receives streamed `{type, message, phase}` JSON lines in real time; after completion, `Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth` exists on disk; profile status is `trained` in SQLite.

## Must-Haves

- `POST /api/training/start` accepts `{profile_id}`, validates the profile exists with a sample, returns `{job_id}`, returns 409 if a job is already running
- `POST /api/training/cancel` body `{profile_id}` → kills subprocess tree, updates status to `failed`
- `GET /api/training/status/:profile_id` → `{phase, progress_pct, status}` from in-memory job state
- `WS /ws/training/:profile_id` → streams `{type, message, phase}` JSON lines; sends `{type:"done"}` on completion
- Phases 1–3 (preprocess, f0 extraction, feature extraction) stream stdout → asyncio.Queue → WebSocket
- Phase 4 (train.py) is started as subprocess; `train.log` is tailed asynchronously → asyncio.Queue → WebSocket
- Phase 5 (FAISS index) runs in-process in a thread executor after training completes, emitting log messages to the queue
- Profile status updated to `training` on start, `trained` on success, `failed` on error or cancel
- `assets/weights/rvc_finetune_active.pth` produced at end of a real smoke-test run (`-te 1 -se 1`)
- Contract tests (WebSocket shape, status endpoint, start/cancel) pass without a running subprocess
- Integration test smoke-runs 1 training epoch and asserts `.pth` exists

## Proof Level

- This slice proves: integration (real subprocess chain; real `.pth` produced)
- Real runtime required: yes — a real 1-epoch training run must complete to prove `.pth` production
- Human/UAT required: no — test assertions cover all acceptance criteria

## Verification

Contract tests (fast, no real subprocesses):
```
conda run -n rvc pytest backend/tests/test_training.py -v
```
All tests should pass in < 5 seconds with mocked subprocesses.

Integration smoke test (real run — may take 5–30 minutes):
```
conda run -n rvc pytest backend/tests/test_training_integration.py -v -s
```
- Starts real training for a 5-second WAV file with `-te 1 -se 1`
- Asserts `Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth` exists
- Asserts at least one log message received over WebSocket during training

Observability verification:
```
# Confirm status surface works
curl -s http://localhost:8000/api/training/status/<profile_id>
# Expected: {"phase":"train","progress_pct":N,"status":"training"}

# Confirm SQLite status update
sqlite3 backend/data/rvc.db "SELECT id, name, status FROM profiles WHERE status='trained';"
```

## Observability / Diagnostics

- Runtime signals:
  - Structured JSON log on training start: `{"event":"training_start","profile_id":"...","job_id":"...","rvc_root":"..."}`
  - Structured JSON log on phase transition: `{"event":"training_phase","phase":"...","profile_id":"..."}`
  - Structured JSON log on completion/failure: `{"event":"training_done"|"training_failed","profile_id":"...","elapsed_s":N}`
  - WebSocket messages carry `{type, message, phase}` — inspectable in any WS test client
- Inspection surfaces:
  - `GET /api/training/status/:profile_id` → `{phase, progress_pct, status}` from in-memory state
  - `GET /health` — existing endpoint confirms `rvc_root` is set
  - `sqlite3 backend/data/rvc.db "SELECT id,name,status FROM profiles;"` — live profile status
  - `ls Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/` — phase artifact directories
  - `ls Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth` — final model presence
  - `tail -f Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/train.log` — live training log
- Failure visibility:
  - `TrainingJob.error` field (last error message string, preserved after failure)
  - `GET /api/training/status/:profile_id` returns `{status:"failed","error":"..."}` with last error
  - All subprocess stderr is merged with stdout (stderr=STDOUT) and streamed as `{type:"log"}` messages — errors appear in the WebSocket stream in real time
  - Profile status in SQLite set to `"failed"` with DB write, persisted across restarts
- Redaction constraints: none (no secrets in training pipeline)

## Integration Closure

- Upstream surfaces consumed:
  - `backend/app/db.py` — `get_db()` for profile reads and status updates
  - `backend/app/routers/profiles.py` — profile `sample_path` and `status` field conventions
  - S01 patterns: `_samples_root()`, `DATA_DIR` env var, `DB_PATH` monkeypatch isolation
  - `Retrieval-based-Voice-Conversion-WebUI/web.py` — `click_train()` logic for filelist.txt + config.json construction
  - `Retrieval-based-Voice-Conversion-WebUI/infer/lib/train/utils.py` — `get_hparams()` for arg reference
- New wiring introduced in this slice:
  - `backend/app/training.py` — `TrainingManager` singleton (in-memory job registry)
  - `backend/app/routers/training.py` — REST + WebSocket router registered in `main.py`
  - `RVC_ROOT` env var now required at runtime (validated at training start, not server start)
  - asyncio subprocess chain + log-file tailing + in-process FAISS index build
- What remains before the milestone is truly usable end-to-end:
  - S03: Realtime VC session with audio device selection and waveform streaming
  - S04: Next.js frontend UI connecting to all three backend features + start/stop scripts

## Tasks

- [x] **T01: Write failing contract tests for training endpoints and WebSocket** `est:45m`
  - Why: Defines the acceptance criteria for the training API before any implementation. Tests fail until T02–T03 build the real code. Forces the contract to be explicit before we code to it.
  - Files: `backend/tests/test_training.py`
  - Do: Write pytest tests covering: `POST /api/training/start` happy path (mock subprocess), 409 on double-start, 404 for unknown profile, `POST /api/training/cancel`, `GET /api/training/status/:profile_id`, WebSocket connection to `/ws/training/:profile_id` receives JSON messages with correct shape `{type, message, phase}`. Use `TestClient.websocket_connect()` from starlette (sync) for the WebSocket tests. Mock the TrainingManager's `start_job` and `cancel_job` at the module level so no subprocess is spawned. Also add a test that verifies a `{type:"done"}` message is eventually received when the mock job completes. Confirm all tests fail with ImportError or 404 (expected — router not yet wired).
  - Verify: `conda run -n rvc pytest backend/tests/test_training.py -v 2>&1 | grep -E "FAILED|ERROR|passed"` — expect all failures, no passes
  - Done when: Tests exist, all fail with import/routing errors (not logic errors)

- [x] **T02: Implement TrainingManager and subprocess chain (phases 1–4)** `est:2h`
  - Why: Core of the slice — the asyncio machinery that runs preprocess → f0 extract → feature extract → train.py as subprocesses, streams their output to a shared queue, and tails the train.log file for phase 4. This is the hardest task and stands alone.
  - Files: `backend/app/training.py`
  - Do:
    1. Define `TrainingJob` dataclass: `job_id`, `profile_id`, `phase`, `progress_pct`, `status` (`training|done|failed|cancelled`), `error`, `queue: asyncio.Queue`, `_proc: Optional[asyncio.subprocess.Process]`
    2. Define `TrainingManager` class with `Dict[str, TrainingJob]` keyed by `profile_id`, methods: `start_job(profile_id, sample_path, rvc_root) -> TrainingJob`, `cancel_job(profile_id)`, `get_job(profile_id) -> Optional[TrainingJob]`, `active_job() -> Optional[TrainingJob]`
    3. `start_job` validates no active job (raises ValueError "job already running"), creates job, launches `asyncio.create_task(_run_pipeline(job, ...))`, returns job
    4. `_run_pipeline` coroutine: chains phases in order, each updating `job.phase`. On any phase failure, puts `{type:"error"}` to queue, sets `job.status="failed"`, updates DB status. On success, puts `{type:"done"}`, sets `job.status="done"`, updates DB to `trained`.
    5. Phase 1 (preprocess): `_run_subprocess(job, cmd, env, cwd, phase="preprocess")` — streams stdout+stderr line by line to queue as `{type:"log", message:line, phase:"preprocess"}`. Returns True on exit code 0, False otherwise.
    6. Phase 2a (f0 extract): same pattern, `phase="extract_f0"`. Env must include `rmvpe_root=<rvc_root>/assets/rmvpe`.
    7. Phase 2b (feature extract): same pattern, `phase="extract_feature"`. Env includes `PYTORCH_ENABLE_MPS_FALLBACK=1`.
    8. Pre-phase 3 setup: build `filelist.txt` from feature dirs (adapt `click_train()` logic from `web.py`), copy/write `config.json` from `configs/inuse/v2/48k.json`. Both must exist before train.py is started.
    9. Phase 3 (train): start subprocess but DO NOT read its stdout (it produces none). Poll `logs/rvc_finetune_active/train.log` every 200ms with aiofiles; wait up to 10s for file to appear (log file created by mp.Process child, there's a race). Stream new lines to queue. Exit when process terminates.
    10. Phase 4 (FAISS index): `await asyncio.get_event_loop().run_in_executor(None, _build_index, rvc_root)` — adapts `train_index()` from `web.py`: loads npy files from `logs/rvc_finetune_active/3_feature768/`, builds IVF FAISS index, saves to `logs/rvc_finetune_active/added_IVF*_Flat*.index`. Emits log messages to queue.
    11. `cancel_job`: kills subprocess via `os.killpg(os.getpgid(proc.pid), signal.SIGTERM)` if proc is alive, sets status `cancelled`, puts `{type:"error", message:"cancelled"}` to queue, updates DB to `failed`.
    12. All subprocess commands: use `sys.executable` for the Python path, `cwd=rvc_root`.
    13. Export a module-level `manager = TrainingManager()` singleton.
  - Verify: `conda run -n rvc python -c "from backend.app.training import manager; print('OK')"` — no import error
  - Done when: Module imports cleanly; `TrainingManager` instantiates without error; `_build_index` function exists

- [x] **T03: Wire training REST + WebSocket router into FastAPI** `est:45m`
  - Why: Connects the `TrainingManager` to HTTP endpoints and WebSocket handler so the contract tests can pass and the integration test can exercise the live pipeline end-to-end.
  - Files: `backend/app/routers/training.py`, `backend/app/main.py`
  - Do:
    1. Create `backend/app/routers/training.py` with FastAPI `APIRouter` prefix `/api/training` and a separate WebSocket route at `/ws/training/{profile_id}`.
    2. `POST /api/training/start`: read `{profile_id}` from JSON body, look up profile via `get_db()`, return 404 if not found, return 400 if `sample_path` is None, call `manager.start_job(...)`, return 409 with `{"detail":"job already running"}` on ValueError, return 200 `{job_id}`.
    3. `POST /api/training/cancel`: read `{profile_id}`, call `manager.cancel_job(profile_id)`, return 200 `{"ok": true}`.
    4. `GET /api/training/status/{profile_id}`: look up job in manager, return `{phase, progress_pct, status, error}` or 404 if no job.
    5. `WS /ws/training/{profile_id}`: accept WebSocket, wait for job in manager (up to 30s polling 100ms intervals), then loop `await job.queue.get()` and `websocket.send_json(msg)`. On `type=="done"` or `type=="error"`, send final message, then close.
    6. Register router in `main.py` (import + `app.include_router`). Also register the WebSocket route on the app directly (not on the router prefix) since WebSocket routes need to be registered on the app.
    7. Emit structured JSON startup log addition: include `rvc_root` validation warning if not set.
  - Verify: `conda run -n rvc pytest backend/tests/test_training.py -v` — all tests pass
  - Done when: All contract tests in `test_training.py` pass; `GET /api/training/status/unknown` returns 404

- [x] **T04: Write and run integration smoke test (1 epoch)** `est:30m` (setup) + real run time
  - Why: Proves the real subprocess chain works end-to-end — MPS/CPU stability, log file tailing, `.pth` production, FAISS index build, and DB status update. This is the M001 roadmap requirement for retiring the "MPS training stability" risk.
  - Files: `backend/tests/test_training_integration.py`
  - Do:
    1. Write `test_training_integration.py` — marked with `@pytest.mark.integration` so it's skipped by default unless explicitly invoked. Skip if `RVC_ROOT` env var is not set or if a real WAV sample is not available.
    2. Test setup: create a real profile via `POST /api/profiles` with a short 5-second WAV file (generate one using `scipy.io.wavfile` or use any existing wav in the test assets). Record the `profile_id`.
    3. Start training via `POST /api/training/start {profile_id}`, override `total_epoch=1, save_every_epoch=1` (pass as query params or expose as body fields in the router — add `epochs` and `save_every` optional fields to the start request body, defaulting to 200 and 10 respectively).
    4. Connect to `WS /ws/training/<profile_id>` using `TestClient.websocket_connect()`. Collect messages until `type=="done"` or `type=="error"` is received (with a 30-minute timeout).
    5. Assert: at least one `{type:"log"}` message received. Assert: final message is `{type:"done"}`. Assert: `<rvc_root>/assets/weights/rvc_finetune_active.pth` exists. Assert: profile status in DB is `trained`.
    6. Run the test: `RVC_ROOT=<path> conda run -n rvc pytest backend/tests/test_training_integration.py -v -s -m integration`.
    7. If the test reveals any subprocess command errors, fix them in `training.py` and re-run.
  - Verify: `ls Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth` — file exists after test completes
  - Done when: Integration test passes; `.pth` file exists on disk; `sqlite3 backend/data/rvc.db "SELECT status FROM profiles WHERE status='trained';"` returns at least one row

## Files Likely Touched

- `backend/app/training.py` — NEW: TrainingManager, TrainingJob, subprocess chain, log tailing, FAISS index build
- `backend/app/routers/training.py` — NEW: REST endpoints + WebSocket handler
- `backend/app/main.py` — add training router import and registration
- `backend/tests/test_training.py` — NEW: contract tests (mocked subprocesses)
- `backend/tests/test_training_integration.py` — NEW: real 1-epoch smoke test
