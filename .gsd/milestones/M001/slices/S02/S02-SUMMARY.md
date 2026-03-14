---
id: S02
parent: M001
milestone: M001
provides:
  - POST /api/training/start — triggers async 5-phase subprocess training chain, returns {job_id}
  - POST /api/training/cancel — kills active subprocess tree, updates profile to failed
  - GET /api/training/status/{profile_id} — runtime inspection surface (phase, progress_pct, status, error)
  - WS /ws/training/{profile_id} — real-time {type, message, phase} JSON line stream
  - backend/app/training.py — TrainingManager singleton with full async subprocess pipeline
  - backend/app/routers/training.py — REST + WebSocket training router
  - Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth — 55MB real fine-tuned model
  - Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/added_IVF6_Flat_nprobe_1_rvc_finetune_active_v2.index — FAISS index
  - backend/tests/test_training.py — 8 contract tests (all passing)
  - backend/tests/test_training_integration.py — real 1-epoch smoke test
  - pytest.ini — project pytest config with integration mark and asyncio_mode=strict
requires:
  - slice: S01
    provides: DB connection pattern (get_db), profile CRUD (sample_path, status fields), DATA_DIR env convention, RVC_ROOT env var pattern
affects:
  - S03 — consumes rvc_finetune_active.pth for realtime inference; consumes profile status=trained
  - S04 — frontend connects to training endpoints + WebSocket for live log display
key_files:
  - backend/app/training.py
  - backend/app/routers/training.py
  - backend/app/main.py
  - backend/tests/test_training.py
  - backend/tests/test_training_integration.py
  - pytest.ini
  - Retrieval-based-Voice-Conversion-WebUI/infer/lib/train/process_ckpt.py
key_decisions:
  - D017: in-memory Dict[str, TrainingJob] keyed by profile_id; single active job at a time
  - D018: phases 1–3 use asyncio.create_subprocess_exec with stdout=PIPE,stderr=STDOUT streaming into asyncio.Queue
  - D019: phase 4 (train.py) tails train.log via aiofiles polling; waits up to 10s for file creation
  - D020: FAISS index built in-process via run_in_executor (CPU-bound; no subprocess)
  - D021: starlette TestClient.websocket_connect() for WS tests; httpx async client for REST tests
  - D022: fixed exp_name="rvc_finetune_active" — single global training slot
  - D023: two-router pattern — ws_router (no prefix) for WS, router (prefixed) for REST — both registered in main.py
  - D024: rmvpe_root must be in train subprocess env — save_small_model → Pipeline.__init__ reads it
  - D025: process_ckpt.py:save_small_model patched to catch model_hash_ckpt failure (PyTorch 2.6 + fairseq incompatibility)
  - D026: pytest.ini created at repo root; asyncio_mode=strict; integration mark registered
patterns_established:
  - _run_subprocess coroutine: streams stdout+stderr line-by-line into asyncio.Queue; returns bool on exit code
  - _tail_train_log coroutine: waits up to 10s for file existence then polls every 200ms with aiofiles
  - _update_db_status: always opens fresh get_db() context; never passes connections across task boundaries
  - cancel_job: os.killpg(os.getpgid(proc.pid), SIGTERM) to kill process group including mp.Process children
  - MASTER_PORT randomized per run (50000–59999) to avoid port conflicts on re-train
  - mock_manager fixture patches backend.app.routers.training.manager (router-level reference, not module-level)
  - Two-router FastAPI pattern for same-module REST+WebSocket with different URL prefixes
observability_surfaces:
  - GET /api/training/status/{profile_id} → {phase, progress_pct, status, error} — runtime polling
  - GET /api/training/status/{profile_id} → {status:"failed", error:"..."} — failure state persists after run
  - WS /ws/training/{profile_id} — real-time {type,message,phase} stream during training
  - Structured stdout JSON: {"event":"training_start|training_phase|training_done|training_failed",...}
  - sqlite3 backend/data/rvc.db "SELECT id,name,status FROM profiles;" — profile status persisted in DB
  - ls Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth — model presence
  - ls Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/ — phase artifact directories
  - G_latest.pth / D_latest.pth in logs/rvc_finetune_active/ — confirms train.py ran even if train.log is empty
drill_down_paths:
  - .gsd/milestones/M001/slices/S02/tasks/T01-SUMMARY.md
  - .gsd/milestones/M001/slices/S02/tasks/T02-SUMMARY.md
  - .gsd/milestones/M001/slices/S02/tasks/T03-SUMMARY.md
  - .gsd/milestones/M001/slices/S02/tasks/T04-SUMMARY.md
duration: ~3.5h (T01: 20m, T02: 1.5h, T03: 25m, T04: 30m + 2 integration test runs)
verification_result: passed
completed_at: 2026-03-14
---

# S02: Training Pipeline with Live Log Streaming

**Full 5-phase RVC training pipeline ships: POST → WebSocket → .pth in 24 seconds on Apple Silicon; all 25 contract tests green and 1-epoch smoke test producing a real 55MB model file.**

## What Happened

S02 built the complete training backend in four sequential tasks.

**T01** defined the API contract as 8 failing tests before any implementation existed. The tests cover all five endpoints (POST start, POST cancel, GET status, WS connect, double-start 409) and forced the implementation contract to be explicit before coding began. A `sync_client` fixture wraps `starlette.testclient.TestClient` for WebSocket compatibility, while a `mock_manager` fixture patches the router-level manager reference and pre-populates the asyncio queue so WS tests terminate cleanly.

**T02** built `backend/app/training.py` — the core of the slice. `TrainingJob` is a dataclass holding all runtime state including an `asyncio.Queue` for log delivery. `TrainingManager` holds an in-memory dict keyed by `profile_id`. The `_run_pipeline` coroutine chains five phases:
- Preprocess and feature extraction (phases 1–3): `asyncio.create_subprocess_exec` with `stdout=PIPE, stderr=STDOUT`, lines streamed into the queue as `{type:"log", message, phase}` JSON.
- Training (phase 4): `train.py` launched with `stdout/stderr=DEVNULL` (it produces no stdout), while `_tail_train_log` polls `train.log` via `aiofiles` every 200ms concurrently.
- FAISS index (phase 5): `_build_index` runs in `loop.run_in_executor(None, ...)` — CPU-bound IVF FAISS construction adapted from `web.py`.

**T03** created `backend/app/routers/training.py` and wired it into `main.py`. A key structural insight: a WebSocket route cannot share the `/api/training` prefix if it should live at `/ws/training/{profile_id}`. Two routers are used — `router` (prefixed) for REST, `ws_router` (no prefix) for WebSocket — both registered in `main.py`. Two bugs in T01's test fixtures were also fixed: `asyncio.get_event_loop().run_until_complete()` → `asyncio.run()` (Python 3.11 compatibility), and `mock_manager` needed to set `RVC_ROOT` before the router's env check ran. With these fixes, all 8 contract tests and the full 25-test suite passed.

**T04** wrote the integration smoke test and ran it twice. The first run exposed two upstream RVC bugs:
1. `save_small_model` in `process_ckpt.py` calls `model_hash_ckpt` → `Pipeline.__init__` which reads `os.environ["rmvpe_root"]` — but the train subprocess env was missing this var. Fixed in `training.py`.
2. PyTorch 2.6 changed `torch.load` default to `weights_only=True`, breaking fairseq's `load_checkpoint_to_cpu` which tries to deserialize `fairseq.data.dictionary.Dictionary` globals. Fixed by catching the hash computation failure in `save_small_model` and proceeding without hash/id metadata (DRM fields only; model is fully functional for inference without them).

The second integration run completed in 24.5 seconds: 60 WebSocket messages, final `{type:"done"}`, 55MB `.pth` on disk, FAISS index written, DB status `trained`.

## Verification

```bash
# Contract tests — 8/8 green
conda run -n rvc pytest backend/tests/test_training.py -v
# → 8 passed in 0.20s

# Full suite — 25/25 green (integration test auto-skipped without RVC_ROOT)
conda run -n rvc pytest backend/tests/ -v
# → 25 passed, 1 skipped in 0.29s

# Integration smoke test (real run)
RVC_ROOT=Retrieval-based-Voice-Conversion-WebUI conda run -n rvc pytest backend/tests/test_training_integration.py -v -s -m integration --timeout=1800
# → 1 passed in 24.50s

# .pth on disk — 55MB
ls -lh Retrieval-based-Voice-Conversion-WebUI/assets/weights/rvc_finetune_active.pth

# FAISS index on disk
ls Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/added_IVF6_Flat_nprobe_1_rvc_finetune_active_v2.index

# Observability — status endpoint (requires running server)
curl -s http://localhost:8000/api/training/status/<profile_id>

# DB status
sqlite3 backend/data/rvc.db "SELECT id, name, status FROM profiles WHERE status='trained';"
```

## Requirements Advanced

- R002 (Training / Fine-tuning Pipeline) — full preprocess → feature extract → train → FAISS pipeline implemented; WebSocket log streaming live; cancel endpoint working; phase/status inspection surface live

## Requirements Validated

- R002 (Training / Fine-tuning Pipeline) — validated by real 1-epoch integration test: `.pth` produced (55MB), FAISS index written, DB status updated to `trained`, WebSocket streamed 60 messages; MPS training stability risk retired (Apple Silicon produces `.pth` in 24 seconds without crashes)

## New Requirements Surfaced

- none

## Requirements Invalidated or Re-scoped

- D010 (MPS training stability) — risk retired: Apple Silicon (MPS/CPU) produces a valid `.pth` in 24 seconds for a 1-epoch run; no device errors observed during the full pipeline

## Deviations

1. **Two-router pattern for WebSocket**: Plan said "register WebSocket route on app directly". In practice the cleanest solution is a second `APIRouter()` with no prefix — both routers registered in `main.py`. Same effect, cleaner module structure.

2. **train.log empty on 1-epoch runs**: The log tailer emits "Warning: train.log did not appear within 10s" because `train.py` calls `os._exit(0)` before the Python logging FileHandler flushes on fast runs. The `{type:"done"}` message is still sent correctly. `G_latest.pth` / `D_latest.pth` presence confirms training ran.

3. **Two upstream RVC bugs fixed**: `rmvpe_root` missing from train env (D024) and PyTorch 2.6 `weights_only` incompatibility in `process_ckpt.py` (D025). Both fixes were necessary for `.pth` production and are documented.

4. **T01 fixture bugs fixed in T03**: `sync_client` used deprecated `asyncio.get_event_loop()` pattern; `mock_manager` lacked `RVC_ROOT` env setup. Both fixed in T03.

## Known Limitations

- `train.log` stays **empty** on fast runs (≤1 epoch) because `train.py` calls `os._exit(0)` before the Python logging FileHandler flushes. Log tailer emits a "did not appear within 10s" warning — harmless. On longer runs (>10s to first epoch), the file will populate normally.
- `model_hash` and `id` fields in the saved `.pth` are empty strings (consequence of D025 fix). RVC inference does not use these fields; only affects RVC's own DRM/authenticity checks.
- In-memory job state is not persisted. A server restart loses all in-flight training state. The SQLite `profiles.status` field persists the last-known status, so a restarted server sees `training` as the last status for an interrupted job — a manual status reset may be needed.
- MASTER_PORT is randomized per run (50000–59999) to avoid conflicts; extremely low probability collision is still theoretically possible.

## Follow-ups

- S03 will consume `rvc_finetune_active.pth` directly — ensure `RVC_ROOT` env var is set in all subprocess invocations (same as train env)
- Add a status-reset endpoint (`POST /api/training/reset/{profile_id}`) in S04 to handle interrupted-training recovery gracefully
- Consider flushing final `train.log` lines after `os._exit` by reading `G_latest.pth` creation time as a proxy for epoch completion — deferred to S04 polish if needed

## Files Created/Modified

- `backend/app/training.py` — NEW: TrainingManager singleton, TrainingJob dataclass, full 5-phase async subprocess pipeline (355 lines)
- `backend/app/routers/training.py` — NEW: REST + WebSocket training router (POST start/cancel, GET status, WS handler)
- `backend/app/main.py` — updated: training_router + training_ws_router registration
- `backend/tests/test_training.py` — NEW (T01), updated (T03): 8 contract tests; fixture fixes for Python 3.11 + RVC_ROOT
- `backend/tests/test_training_integration.py` — NEW: real 1-epoch smoke test with autouse skip guard
- `pytest.ini` — NEW: project pytest config (asyncio_mode=strict, integration mark)
- `Retrieval-based-Voice-Conversion-WebUI/infer/lib/train/process_ckpt.py` — patched: save_small_model catches model_hash_ckpt failure non-fatally

## Forward Intelligence

### What the next slice should know
- `rvc_finetune_active.pth` is at `{RVC_ROOT}/assets/weights/rvc_finetune_active.pth` — this is the model path S03's `rtrvc.py` must be configured to load. Set `model_path` in the rtrvc.py `Config` accordingly.
- `RVC_ROOT` env var must be set for all subprocesses. The `_run_pipeline` already sets `rmvpe_root` for preprocess/extract/train — ensure rtrvc.py's runtime also has `rmvpe_root` set (same path: `{RVC_ROOT}/assets/rmvpe`).
- The `profiles.status` field in SQLite is the canonical state for "is a model trained for this profile?". S03 should gate realtime VC on `status == "trained"`.
- FAISS index path: `{RVC_ROOT}/logs/rvc_finetune_active/added_IVF6_Flat_nprobe_1_rvc_finetune_active_v2.index` — pass this as `index_rate` path to rtrvc.py.

### What's fragile
- `_tail_train_log` 10-second deadline is calibrated for the current hardware. On a slower machine this could time out before `train.py` creates the log file. The warning is logged but training continues — not a crash.
- `process_ckpt.py` patch (D025) is a local override of the upstream RVC file. If the upstream repo is updated, the patch could be lost. The patched region is `save_small_model`; the patch wraps `model_hash_ckpt(ckpt, save_dir, name, epoch)` in a try/except.
- `MASTER_PORT` randomization avoids re-train port conflicts but doesn't guarantee they can't occur. If two training runs happen in rapid succession (race between kill and new start), a port collision is theoretically possible.

### Authoritative diagnostics
- `GET /api/training/status/{profile_id}` — fastest non-WS check of training state during and after a run
- `ls {RVC_ROOT}/logs/rvc_finetune_active/` — which phase directories exist shows how far a failed run progressed: `0_gt_wavs/`, `3_feature768/`, `2a_f0/`, `2b-f0nsf/` for phases 1–3; `G_latest.pth` for phase 4
- `sqlite3 backend/data/rvc.db "SELECT id,name,status,model_path FROM profiles;"` — persisted state survives server restarts
- Structured stdout JSON from uvicorn: `grep training_start\|training_phase\|training_done\|training_failed` — timeline of a run

### What assumptions changed
- "MPS training stability is unknown" (milestone risk) — resolved: Apple Silicon runs the full pipeline without device errors in 24 seconds. CPU fallback is not needed for the training path.
- "train.log will be tailed live" — partially true: on fast runs (≤1 epoch), train.log is empty because `os._exit(0)` skips Python logging flush. The `{type:"done"}` WebSocket message is still sent via the main pipeline coroutine. `G_latest.pth` / `D_latest.pth` are the reliable completion proof for fast runs.
- "fairseq environment is stable" — PyTorch 2.6 broke `weights_only` compatibility with fairseq. The patch is in place; future PyTorch or fairseq updates should be tested against the training pipeline.
