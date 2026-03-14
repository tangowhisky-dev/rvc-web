---
id: T02
parent: S02
milestone: M001
provides:
  - backend/app/training.py — TrainingManager singleton, TrainingJob dataclass, full async subprocess pipeline
key_files:
  - backend/app/training.py
key_decisions:
  - D017: in-memory Dict[str, TrainingJob] keyed by profile_id; single active job at a time
  - D018: phases 1–3 use asyncio.create_subprocess_exec with stdout=PIPE,stderr=STDOUT
  - D019: phase 4 (train.py) tails train.log via aiofiles polling; waits up to 10s for file creation
  - D020: FAISS index built in-process via run_in_executor (CPU-bound)
  - D022: fixed exp_name="rvc_finetune_active" for single training slot
patterns_established:
  - _run_subprocess coroutine: streams stdout+stderr line-by-line into asyncio.Queue; returns bool on exit code
  - _tail_train_log coroutine: waits for file existence (10s deadline), then polls every 200ms with aiofiles
  - _update_db_status: always opens fresh get_db() context; never passes connections across task boundaries
  - cancel_job: os.killpg(os.getpgid(proc.pid), SIGTERM) to kill process group including mp.Process children
  - MASTER_PORT randomized per run (50000–59999) to avoid port conflicts on re-train
observability_surfaces:
  - stdout structured JSON on training_start, training_phase, training_done, training_failed events
  - job.status field ("training"|"done"|"failed"|"cancelled") inspectable from manager.get_job()
  - job.error field preserves last error message after failure
  - asyncio.Queue drainable by WebSocket handler (T03) for real-time log streaming
duration: ~1.5h
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T02: Implement TrainingManager and Subprocess Chain (Phases 1–5)

**Built `backend/app/training.py`: complete TrainingManager singleton with async 5-phase subprocess pipeline (preprocess → F0 → feature → train+log-tail → FAISS index), all streaming into asyncio.Queue.**

## What Happened

Implemented `backend/app/training.py` from scratch following the task plan and research notes exactly. Key design decisions executed:

1. **`TrainingJob` dataclass** — all required fields: `job_id`, `profile_id`, `phase`, `progress_pct`, `status`, `error`, `queue: asyncio.Queue`, `_proc: Optional[asyncio.subprocess.Process]`.

2. **`_run_subprocess`** — uses `asyncio.create_subprocess_exec` with `stdout=PIPE, stderr=STDOUT`. Streams lines into `job.queue` as `{type:"log", message, phase}`. Returns bool on exit code.

3. **`_tail_train_log`** — polls for log file existence with a 10-second deadline (train.log is created by an `mp.Process` child of train.py, so there's a race). Once file exists, opens with `aiofiles`, seeks to EOF, polls every 200ms for new lines. Stops when `job.status != "training"`.

4. **`_build_filelist`** — adapts `click_train()` from `web.py`. Builds intersection of stems across 4 subdirs (`0_gt_wavs`, `3_feature768`, `2a_f0`, `2b-f0nsf`). Appends 2 mute padding entries. Shuffles and writes `filelist.txt`.

5. **`_write_config`** — idempotent: skips copy if `config.json` already exists. Uses `shutil.copy2` from `configs/inuse/v2/48k.json`.

6. **`_build_index`** — sync function (for `run_in_executor`). Adapts `train_index()` from `web.py`. Loads `.npy` files, concatenates, shuffles. Skips MiniBatchKMeans for typical short samples (`shape[0] <= 2e5`). Builds `IVF{n_ivf},Flat` FAISS index at dim=768. Returns path to `added_IVF*.index`.

7. **`_run_pipeline`** — full orchestration coroutine. Chains all phases with proper env vars (`rmvpe_root` for F0, `PYTORCH_ENABLE_MPS_FALLBACK=1` for feature extract, `MASTER_ADDR/MASTER_PORT` for train). Phase 3 (train.py): subprocess started with `DEVNULL` stdout/stderr, then `asyncio.create_task(_tail_train_log(...))` runs concurrently. After `proc.wait()`, gives tail task 0.5s to drain before cancelling. Phase 4 FAISS runs in `loop.run_in_executor(None, _build_index, rvc_root)`. On success, updates DB to `trained`, emits `{type:"done"}`. On any failure, sets `job.status="failed"`, emits `{type:"error"}`, updates DB to `"failed"`.

8. **`TrainingManager`** — `_jobs: Dict[str, TrainingJob]` keyed by `profile_id`. `start_job` raises `ValueError("job already running")` if any job has `status=="training"`. Creates job, launches `asyncio.create_task(_run_pipeline(...))`, returns job immediately. `cancel_job` uses `os.killpg(os.getpgid(proc.pid), SIGTERM)` to kill process group. `get_job` and `active_job` are simple dict lookups.

9. **`manager = TrainingManager()`** — module-level singleton exported at bottom of file.

## Verification

```
# Import check
conda run -n rvc python -c "from backend.app.training import manager, TrainingJob, TrainingManager; print('OK')"
# Result: OK

# Type/defaults check
conda run -n rvc python -c "
from backend.app.training import manager, TrainingJob
import asyncio
job = TrainingJob(job_id='x', profile_id='y')
print('job.phase:', job.phase)      # idle
print('job.status:', job.status)    # training
print('queue type:', type(job.queue))  # asyncio.Queue
print('manager type:', type(manager))  # TrainingManager
print('OK')
"
# Result: all values correct, OK printed

# Must-have checks (all 11 passing)
# - TrainingJob fields including queue and _proc: PASSED
# - TrainingJob defaults: PASSED
# - TrainingManager methods present: PASSED
# - _build_index is sync (for run_in_executor): PASSED
# - _run_subprocess is async: PASSED
# - _tail_train_log is async: PASSED
```

Contract tests still fail as expected (router module `backend.app.routers.training` not yet created — T03 task):
```
conda run -n rvc pytest backend/tests/test_training.py -v
# 8 errors at setup of mock_manager fixture (ImportError on router module)
```
This is the expected state for an intermediate T02 — tests will pass after T03 wires the router.

## Diagnostics

- `from backend.app.training import manager` — confirms module is importable
- `manager.get_job(profile_id)` — returns `TrainingJob` or `None`; inspect `.status`, `.phase`, `.error`
- `job.queue` — asyncio.Queue containing `{type, message, phase}` dicts; drain via WebSocket handler (T03)
- Structured stdout logs: `{"event":"training_start|training_phase|training_done|training_failed", ...}` — grep from uvicorn stdout
- Phase artifacts in `Retrieval-based-Voice-Conversion-WebUI/logs/rvc_finetune_active/` — inspect which subdirs exist to determine how far a failed run got
- `train.log` at `logs/rvc_finetune_active/train.log` — written by train.py FileHandler; tail for live status

## Deviations

- **Phase 3 env vars**: The task plan specified building `MASTER_ADDR/MASTER_PORT` in the train subprocess env. Implemented as specified, but noted that `train.py` sets these itself too — the explicit env values ensure correctness regardless.
- **`_emit_phase` helper**: Added a small `_emit_phase` closure inside `_run_pipeline` to DRY up phase transitions (sets `job.phase`, puts `{type:"phase"}` to queue, and emits structured log). Not in the plan but aligns with the observability requirements.
- **`asyncio.sleep(0.5)` drain window**: After `train_proc.wait()`, added 500ms before cancelling the tail task so final log lines from the train run are captured. Not explicitly in the plan but required to avoid losing the last log entries.

## Known Issues

- None. The module is complete and all must-haves verified.

## Files Created/Modified

- `backend/app/training.py` — NEW: complete TrainingManager module (355 lines); all phases, helpers, and singleton
- `.gsd/milestones/M001/slices/S02/S02-PLAN.md` — marked T02 `[x]`
