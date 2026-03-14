---
estimated_steps: 8
estimated_files: 1
---

# T02: Implement TrainingManager and Subprocess Chain (Phases 1–5)

**Slice:** S02 — Training Pipeline with Live Log Streaming
**Milestone:** M001

## Description

Build `backend/app/training.py` — the core of S02. Contains the `TrainingJob` dataclass, `TrainingManager` singleton, and the full async subprocess pipeline that chains preprocess → F0 extraction → feature extraction → train.py (with log-file tailing) → FAISS index build (in-process). All phases funnel log lines into an `asyncio.Queue`, which the WebSocket handler (T03) drains.

This task does not wire any HTTP routes — it only implements the business logic so it can be unit-tested and imported cleanly.

The most important constraints from research:
- Phases 1–3: `asyncio.create_subprocess_exec` with `stdout=PIPE, stderr=STDOUT`, stream lines to queue.
- Phase 4 (train.py): no stdout. Start subprocess, then poll `logs/rvc_finetune_active/train.log` via `aiofiles` every 200ms. Wait up to 10s for the log file to appear (it's created by a child `mp.Process`, not the subprocess itself).
- Phase 5 (FAISS): run in thread executor (`run_in_executor`) — calls faiss directly, is CPU-bound.
- Pre-phase 4 setup: write `config.json` (copy from `configs/inuse/v2/48k.json`) and build `filelist.txt` using the same logic as `click_train()` in `web.py`.
- All subprocess commands: `cwd=rvc_root`; use `sys.executable` for the Python interpreter path.
- `rmvpe_root` env var required for phase 2a.
- Cancel: kill subprocess process group with `os.killpg(os.getpgid(proc.pid), signal.SIGTERM)`, set status → `cancelled`, update DB.

## Steps

1. Define `TrainingJob` dataclass with fields: `job_id: str`, `profile_id: str`, `phase: str = "idle"`, `progress_pct: int = 0`, `status: str = "training"`, `error: Optional[str] = None`, `queue: asyncio.Queue = field(default_factory=asyncio.Queue)`, `_proc: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)`.

2. Implement `_run_subprocess(job, args, env, cwd, phase) -> bool` coroutine:
   - `proc = await asyncio.create_subprocess_exec(*args, stdout=PIPE, stderr=STDOUT, env=env, cwd=cwd)`
   - Store `job._proc = proc`
   - `async for line in proc.stdout: await job.queue.put({"type":"log","message":line.decode().rstrip(),"phase":phase})`
   - `await proc.wait(); return proc.returncode == 0`

3. Implement `_tail_train_log(job, log_path) -> None` coroutine:
   - Wait up to 10s for `log_path` to exist (poll every 200ms)
   - Open with `aiofiles.open(log_path, "r")` and seek to end
   - Poll every 200ms: `line = await f.readline()`; if non-empty, put to queue; if job is no longer in `training` status, break

4. Implement `_build_filelist(rvc_root, exp_dir)` sync function:
   - Adapt `click_train()` logic from `Retrieval-based-Voice-Conversion-WebUI/web.py`
   - `gt_wavs_dir = f"{exp_dir}/0_gt_wavs"`, `feature_dir = f"{exp_dir}/3_feature768"`, `f0_dir = f"{exp_dir}/2a_f0"`, `f0nsf_dir = f"{exp_dir}/2b-f0nsf"`
   - Build `names` as intersection of stems in all four dirs
   - Build `opt` list of pipe-delimited lines per name (with f0 fields)
   - Append 2 mute entries: `f"{rvc_root}/logs/mute/0_gt_wavs/mute48k.wav|...` (use sr=48k, fea_dim=768)
   - Shuffle opt; write to `f"{exp_dir}/filelist.txt"`

5. Implement `_write_config(rvc_root, exp_dir)` sync function:
   - `config_save_path = f"{exp_dir}/config.json"`
   - If file already exists, return (idempotent)
   - Copy from `f"{rvc_root}/configs/inuse/v2/48k.json"` to `config_save_path` using `shutil.copy2`

6. Implement `_build_index(rvc_root) -> str` sync function (runs in thread executor):
   - Adapt `train_index()` from `web.py`
   - `exp_dir = f"{rvc_root}/logs/rvc_finetune_active"`, `feature_dir = f"{exp_dir}/3_feature768"`
   - Load all `.npy` files, `np.concatenate`, build FAISS `IVF<n_ivf>,Flat` index (dim=768)
   - Save `trained_IVF*` index and `added_IVF*` index to `exp_dir`
   - Return the `added_IVF*` path as string
   - Skip kmeans step if `big_npy.shape[0] <= 2e5` (typical for short samples)

7. Implement `_run_pipeline(job, sample_dir, rvc_root, total_epoch, save_every)` coroutine:
   - Phase 1 preprocess: build cmd using `sys.executable`, `rvc_root/infer/modules/train/preprocess.py`, args from research. `inp_root = sample_dir` (abs path). `exp_dir = f"{rvc_root}/logs/rvc_finetune_active"`. Run `_run_subprocess`. On failure: set error, put `{type:"error"}`, update DB, return.
   - Phase 2a f0 extract: build cmd, env with `rmvpe_root`. Run `_run_subprocess`. On failure: same error handling.
   - Phase 2b feature extract: build cmd, env with `PYTORCH_ENABLE_MPS_FALLBACK=1`. Run `_run_subprocess`. On failure: same error handling.
   - Pre-phase 3: call `_write_config(rvc_root, exp_dir)` and `_build_filelist(rvc_root, exp_dir)` synchronously.
   - Phase 3 train: build cmd with all required args (`-e rvc_finetune_active -sr 48k -f0 1 -bs 4 -te <total_epoch> -se <save_every> -pg assets/pretrained_v2/f0G48k.pth -pd assets/pretrained_v2/f0D48k.pth -l 1 -c 0 -sw 0 -v v2 -a ""`). Set `job.phase = "train"`. Start subprocess with `MASTER_ADDR=localhost`, `MASTER_PORT` = random port (50000–59999). Do NOT read stdout. Launch `asyncio.create_task(_tail_train_log(...))`. `await proc.wait()`. Cancel tail task. Check return code.
   - Phase 4 FAISS: put `{type:"phase","message":"Building FAISS index","phase":"index"}`. `loop.run_in_executor(None, _build_index, rvc_root)`. Put result as log message.
   - On full success: update profile status to `trained` in DB (via `get_db()`). Put `{type:"done","message":"Training complete","phase":"done"}`. Set `job.status = "done"`.
   - `_update_db_status(profile_id, status)` helper: opens fresh `get_db()` connection, runs UPDATE, commits.

8. Implement `TrainingManager` class:
   - `_jobs: Dict[str, TrainingJob] = {}` — keyed by profile_id
   - `start_job(profile_id, sample_dir, rvc_root, total_epoch=200, save_every=10) -> TrainingJob`: check if any job has `status=="training"` → raise `ValueError("job already running")`. Create job, store in `_jobs`, `asyncio.create_task(_run_pipeline(...))`, return job.
   - `cancel_job(profile_id)`: get job, kill proc, set `job.status = "cancelled"`, put error message, call `_update_db_status(profile_id, "failed")`.
   - `get_job(profile_id) -> Optional[TrainingJob]`: return `_jobs.get(profile_id)`.
   - `active_job() -> Optional[TrainingJob]`: first job with status `"training"`, or None.

9. Export `manager = TrainingManager()` at module level.

## Must-Haves

- [ ] `TrainingJob` dataclass with all fields including `queue` and `_proc`
- [ ] `_run_subprocess` streams stdout+stderr line by line (not buffered)
- [ ] `_tail_train_log` waits up to 10s for file creation before starting tail
- [ ] Phase 3 train.py subprocess: stdout NOT read (it has none); log tailing via file polling
- [ ] `_build_filelist` uses intersection of 4 subdirectory stems + appends mute entries
- [ ] `_write_config` is idempotent (skip if file exists)
- [ ] `_build_index` runs in thread executor (not blocking the event loop)
- [ ] MASTER_PORT is randomized for each run (avoids port conflicts on re-train)
- [ ] `cancel_job` kills process group (not just the direct child)
- [ ] `_update_db_status` uses a fresh `get_db()` context (does not pass connections across task boundaries)
- [ ] `manager` singleton exported at module level

## Verification

```bash
# Import check
conda run -n rvc python -c "from backend.app.training import manager, TrainingJob, TrainingManager; print('OK')"

# Basic type check
conda run -n rvc python -c "
from backend.app.training import manager, TrainingJob
import asyncio
job = TrainingJob(job_id='x', profile_id='y')
print('job.phase:', job.phase)
print('job.status:', job.status)
print('queue type:', type(job.queue))
print('manager type:', type(manager))
print('OK')
"
```

## Observability Impact

- Signals added/changed:
  - `{"event":"training_start","profile_id":"...","job_id":"...","rvc_root":"..."}` — emitted to stdout on `start_job`
  - `{"event":"training_phase","phase":"...","profile_id":"..."}` — emitted on each phase transition
  - `{"event":"training_done","profile_id":"...","elapsed_s":N}` or `{"event":"training_failed","profile_id":"...","error":"..."}` — emitted on completion
- How a future agent inspects this: tail stdout of the uvicorn process; or `GET /api/training/status/:profile_id` (wired in T03)
- Failure state exposed: `job.error` field preserved after failure; `job.status` set to `"failed"` or `"cancelled"`; last error message emitted to queue as `{type:"error"}` message visible to any connected WebSocket client

## Inputs

- `S02-RESEARCH.md` — all subprocess commands, arg signatures, pitfalls, and env var requirements
- `Retrieval-based-Voice-Conversion-WebUI/web.py` — `click_train()` for filelist.txt/config.json logic, `train_index()` for FAISS index build
- `backend/app/db.py` — `get_db()` context manager for DB status updates

## Expected Output

- `backend/app/training.py` — complete, importable module with `manager` singleton, all phases implemented
