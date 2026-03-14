"""TrainingManager and async subprocess pipeline for RVC fine-tuning.

Provides:
  - TrainingJob: dataclass holding per-job state, asyncio.Queue, and subprocess ref
  - TrainingManager: singleton orchestrating one active training job at a time
  - manager: module-level TrainingManager singleton (import this)

Pipeline phases:
  1. Preprocess       — infer/modules/train/preprocess.py (stdout captured)
  2a. F0 extraction   — infer/modules/train/extract_f0_print.py (stdout captured)
  2b. Feature extract — infer/modules/train/extract_feature_print.py (stdout captured)
  3. Train            — infer/modules/train/train.py (no stdout; tail train.log file)
  4. FAISS index      — in-process via run_in_executor (CPU-bound)

All phases funnel JSON messages into TrainingJob.queue:
  {"type": "log",   "message": "...",  "phase": "preprocess"}
  {"type": "phase", "message": "...",  "phase": "..."}
  {"type": "done",  "message": "Training complete", "phase": "done"}
  {"type": "error", "message": "...",  "phase": "..."}

The WebSocket handler (T03) drains the queue and forwards to connected clients.
"""

import asyncio
import json
import multiprocessing
import os
import random
import shutil
import signal
import sys
import time
import uuid
from asyncio import subprocess as asyncio_subprocess
from dataclasses import dataclass, field
from random import shuffle
from typing import Dict, Optional

import aiofiles
import numpy as np

from backend.app.db import get_db


# ---------------------------------------------------------------------------
# TrainingJob
# ---------------------------------------------------------------------------

@dataclass
class TrainingJob:
    """Holds all runtime state for a single training run."""

    job_id: str
    profile_id: str
    phase: str = "idle"
    progress_pct: int = 0
    status: str = "training"
    error: Optional[str] = None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _proc: Optional[asyncio_subprocess.Process] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

async def _update_db_status(profile_id: str, status: str) -> None:
    """Open a fresh DB connection and update profile status.

    Always uses a fresh get_db() context — never shares connections across
    task/coroutine boundaries (S01 pattern).
    """
    async with get_db() as db:
        await db.execute(
            "UPDATE profiles SET status = ? WHERE id = ?",
            (status, profile_id),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

async def _run_subprocess(
    job: TrainingJob,
    args: list,
    env: dict,
    cwd: str,
    phase: str,
) -> bool:
    """Run a subprocess, stream stdout+stderr lines into job.queue.

    Returns True on exit code 0, False otherwise.
    All output lines are put as {"type":"log","message":line,"phase":phase}.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio_subprocess.PIPE,
        stderr=asyncio_subprocess.STDOUT,
        env=env,
        cwd=cwd,
    )
    job._proc = proc

    async for raw_line in proc.stdout:
        line = raw_line.decode(errors="replace").rstrip()
        if line:
            await job.queue.put({"type": "log", "message": line, "phase": phase})

    await proc.wait()
    job._proc = None
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Log-file tailer for train.py
# ---------------------------------------------------------------------------

async def _tail_train_log(job: TrainingJob, log_path: str) -> None:
    """Poll train.log for new lines and put them into job.queue.

    Waits up to 10 seconds for the log file to appear (it is created by a
    child mp.Process inside train.py, so there is a race between subprocess
    start and file creation).

    Polls every 200ms. Stops when job status is no longer "training".
    """
    # Wait up to 10s for log file to appear
    deadline = time.monotonic() + 10.0
    while not os.path.exists(log_path):
        if time.monotonic() >= deadline:
            await job.queue.put({
                "type": "log",
                "message": "Warning: train.log did not appear within 10s; skipping tail",
                "phase": "train",
            })
            return
        await asyncio.sleep(0.2)

    async with aiofiles.open(log_path, "r") as f:
        # Seek to end so we don't replay pre-existing content on re-runs
        await f.seek(0, 2)  # SEEK_END

        while job.status == "training":
            line = await f.readline()
            if line:
                stripped = line.rstrip()
                if stripped:
                    await job.queue.put({
                        "type": "log",
                        "message": stripped,
                        "phase": "train",
                    })
            else:
                await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# filelist.txt builder (adapts click_train() from web.py)
# ---------------------------------------------------------------------------

def _build_filelist(rvc_root: str, exp_dir: str) -> None:
    """Build logs/rvc_finetune_active/filelist.txt from feature directories.

    Adapts the filelist construction logic from click_train() in web.py.
    Uses v2 feature dimensions (768) and 48k sample rate with f0 enabled.

    exp_dir: absolute path to logs/rvc_finetune_active/
    """
    gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
    feature_dir = f"{exp_dir}/3_feature768"
    f0_dir = f"{exp_dir}/2a_f0"
    f0nsf_dir = f"{exp_dir}/2b-f0nsf"

    # Build intersection of stems across all four directories
    def _stems(d: str) -> set:
        if not os.path.isdir(d):
            return set()
        return {name.split(".")[0] for name in os.listdir(d)}

    names = (
        _stems(gt_wavs_dir)
        & _stems(feature_dir)
        & _stems(f0_dir)
        & _stems(f0nsf_dir)
    )

    spk_id = 0  # speaker id; fixed for single-speaker fine-tuning
    fea_dim = 768  # v2 feature dimension
    sr_tag = "48k"

    opt = []
    for name in names:
        opt.append(
            "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
            % (
                gt_wavs_dir,
                name,
                feature_dir,
                name,
                f0_dir,
                name,
                f0nsf_dir,
                name,
                spk_id,
            )
        )

    # Append 2 mute padding entries (required by train.py DataLoader)
    for _ in range(2):
        opt.append(
            "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|"
            "%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
            % (rvc_root, sr_tag, rvc_root, fea_dim, rvc_root, rvc_root, spk_id)
        )

    shuffle(opt)

    filelist_path = f"{exp_dir}/filelist.txt"
    with open(filelist_path, "w") as f:
        f.write("\n".join(opt))


# ---------------------------------------------------------------------------
# config.json writer (idempotent)
# ---------------------------------------------------------------------------

def _write_config(rvc_root: str, exp_dir: str) -> None:
    """Copy configs/inuse/v2/48k.json into exp_dir/config.json (idempotent).

    Skips the copy if config.json already exists so re-runs are safe.
    train.py reads config_save_path = exp_dir/config.json at startup via
    get_hparams(); the file must exist before train.py is launched.
    """
    config_save_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_save_path):
        return
    src = os.path.join(rvc_root, "configs", "inuse", "v2", "48k.json")
    shutil.copy2(src, config_save_path)


# ---------------------------------------------------------------------------
# FAISS index builder (runs in thread executor)
# ---------------------------------------------------------------------------

def _build_index(rvc_root: str) -> str:
    """Build FAISS IVF index from 3_feature768/ npy files.

    Adapts train_index() from web.py. Runs synchronously in a thread
    executor (run_in_executor) because it is CPU-bound numpy/faiss work.

    Returns the path to the saved added_IVF*.index file.
    """
    import faiss  # noqa: PLC0415 — lazy import; faiss is large

    exp_name = "rvc_finetune_active"
    exp_dir = os.path.join(rvc_root, "logs", exp_name)
    feature_dir = os.path.join(exp_dir, "3_feature768")

    if not os.path.isdir(feature_dir):
        raise RuntimeError(f"Feature directory not found: {feature_dir}")

    npy_files = sorted(f for f in os.listdir(feature_dir) if f.endswith(".npy"))
    if not npy_files:
        raise RuntimeError(f"No .npy files found in {feature_dir}")

    npys = [np.load(os.path.join(feature_dir, fname)) for fname in npy_files]
    big_npy = np.concatenate(npys, axis=0)

    # Shuffle
    idx = np.arange(big_npy.shape[0])
    np.random.shuffle(idx)
    big_npy = big_npy[idx]

    # Skip kmeans reduction for typical short samples (shape[0] <= 2e5)
    if big_npy.shape[0] > 2e5:
        from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415
        n_clusters = 10000
        big_npy = (
            MiniBatchKMeans(
                n_clusters=n_clusters,
                verbose=True,
                batch_size=256 * multiprocessing.cpu_count(),
                compute_labels=False,
                init="random",
            )
            .fit(big_npy)
            .cluster_centers_
        )

    np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    n_ivf = max(n_ivf, 1)  # safety floor

    index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)

    trained_path = os.path.join(
        exp_dir,
        f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_v2.index",
    )
    faiss.write_index(index, trained_path)

    # Add all vectors and save the searchable index
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])

    added_path = os.path.join(
        exp_dir,
        f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_v2.index",
    )
    faiss.write_index(index, added_path)

    return added_path


# ---------------------------------------------------------------------------
# Core pipeline coroutine
# ---------------------------------------------------------------------------

async def _run_pipeline(
    job: TrainingJob,
    sample_dir: str,
    rvc_root: str,
    total_epoch: int = 200,
    save_every: int = 10,
) -> None:
    """Orchestrate the full training pipeline for one TrainingJob.

    Chains: preprocess → f0 extract → feature extract → (config+filelist setup)
            → train (with log tailing) → FAISS index build.

    All output is funnelled into job.queue as JSON messages.
    Sets job.status to "done" or "failed" on completion.
    Updates profile status in SQLite via _update_db_status().
    """
    start_ts = time.monotonic()
    exp_name = "rvc_finetune_active"
    exp_dir = os.path.join(rvc_root, "logs", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    n_cpu = multiprocessing.cpu_count()
    python = sys.executable

    # Base environment: inherit everything, then layer overrides
    base_env = os.environ.copy()

    def _emit_phase(phase: str, message: str) -> None:
        """Helper: put a phase-transition message to the queue (fire-and-forget for sync)."""
        job.phase = phase
        job.queue.put_nowait({"type": "phase", "message": message, "phase": phase})
        # Structured observability log
        print(
            json.dumps({"event": "training_phase", "phase": phase, "profile_id": job.profile_id}),
            flush=True,
        )

    async def _fail(phase: str, message: str) -> None:
        job.status = "failed"
        job.error = message
        await job.queue.put({"type": "error", "message": message, "phase": phase})
        await _update_db_status(job.profile_id, "failed")
        elapsed = time.monotonic() - start_ts
        print(
            json.dumps({
                "event": "training_failed",
                "profile_id": job.profile_id,
                "error": message,
                "elapsed_s": round(elapsed, 1),
            }),
            flush=True,
        )

    # ------------------------------------------------------------------
    # Phase 1: Preprocess
    # ------------------------------------------------------------------
    _emit_phase("preprocess", "Preprocessing audio samples")

    preprocess_args = [
        python,
        "infer/modules/train/preprocess.py",
        sample_dir,       # inp_root (directory, not the wav file itself)
        "48000",          # sr
        str(n_cpu),       # n_p
        exp_dir,          # exp_dir (absolute path)
        "False",          # noparallel
        "3.0",            # per (segment duration)
    ]

    ok = await _run_subprocess(job, preprocess_args, base_env, rvc_root, "preprocess")
    if not ok:
        await _fail("preprocess", "Preprocess phase failed (non-zero exit code)")
        return

    # ------------------------------------------------------------------
    # Phase 2a: F0 Extraction
    # ------------------------------------------------------------------
    _emit_phase("extract_f0", "Extracting pitch (F0)")

    f0_env = base_env.copy()
    f0_env["rmvpe_root"] = os.path.join(rvc_root, "assets", "rmvpe")

    f0_args = [
        python,
        "infer/modules/train/extract_f0_print.py",
        exp_dir,   # exp_dir absolute
        str(n_cpu),
        "rmvpe",   # f0method
        "mps",     # device (with PYTORCH_ENABLE_MPS_FALLBACK fallback)
        "False",   # is_half
    ]

    ok = await _run_subprocess(job, f0_args, f0_env, rvc_root, "extract_f0")
    if not ok:
        await _fail("extract_f0", "F0 extraction phase failed (non-zero exit code)")
        return

    # ------------------------------------------------------------------
    # Phase 2b: Feature Extraction (HuBERT)
    # ------------------------------------------------------------------
    _emit_phase("extract_feature", "Extracting features (HuBERT)")

    feat_env = base_env.copy()
    feat_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    feat_args = [
        python,
        "infer/modules/train/extract_feature_print.py",
        "mps",   # device
        "1",     # n_part
        "0",     # i_part
        "",      # i_gpu (empty → use device arg)
        exp_dir, # exp_dir absolute
        "v2",    # version
        "False", # is_half
    ]

    ok = await _run_subprocess(job, feat_args, feat_env, rvc_root, "extract_feature")
    if not ok:
        await _fail("extract_feature", "Feature extraction phase failed (non-zero exit code)")
        return

    # ------------------------------------------------------------------
    # Pre-phase 3 setup: config.json and filelist.txt
    # ------------------------------------------------------------------
    try:
        _write_config(rvc_root, exp_dir)
        _build_filelist(rvc_root, exp_dir)
    except Exception as exc:
        await _fail("setup", f"Pre-train setup failed: {exc}")
        return

    # ------------------------------------------------------------------
    # Phase 3: Train (log-file tailing)
    # ------------------------------------------------------------------
    _emit_phase("train", "Training model")

    master_port = random.randint(50000, 59999)
    train_env = base_env.copy()
    train_env["MASTER_ADDR"] = "localhost"
    train_env["MASTER_PORT"] = str(master_port)
    # save_small_model calls model_hash_ckpt → Pipeline.__init__ which needs rmvpe_root
    train_env["rmvpe_root"] = os.path.join(rvc_root, "assets", "rmvpe")

    train_args = [
        python,
        "infer/modules/train/train.py",
        "-e", exp_name,
        "-sr", "48k",
        "-f0", "1",
        "-bs", "4",
        "-te", str(total_epoch),
        "-se", str(save_every),
        "-pg", "assets/pretrained_v2/f0G48k.pth",
        "-pd", "assets/pretrained_v2/f0D48k.pth",
        "-l", "1",    # save latest only
        "-c", "0",    # no cache GPU
        "-sw", "0",   # no save every weights
        "-v", "v2",
        "-a", "",
    ]

    train_proc = await asyncio.create_subprocess_exec(
        *train_args,
        stdout=asyncio_subprocess.DEVNULL,   # train.py has no stdout
        stderr=asyncio_subprocess.DEVNULL,   # all output goes to train.log
        env=train_env,
        cwd=rvc_root,
    )
    job._proc = train_proc

    log_path = os.path.join(exp_dir, "train.log")

    # Tail the log file concurrently while train.py runs
    tail_task = asyncio.create_task(_tail_train_log(job, log_path))

    await train_proc.wait()
    job._proc = None

    # Cancel the tail task after the process exits (it may still have
    # unread lines — give it a brief moment to drain before cancelling)
    await asyncio.sleep(0.5)
    tail_task.cancel()
    try:
        await tail_task
    except asyncio.CancelledError:
        pass

    if train_proc.returncode != 0:
        await _fail("train", f"Train phase failed (exit code {train_proc.returncode})")
        return

    # ------------------------------------------------------------------
    # Phase 4: FAISS index (CPU-bound, run in thread executor)
    # ------------------------------------------------------------------
    _emit_phase("index", "Building FAISS index")

    loop = asyncio.get_event_loop()
    try:
        added_index_path = await loop.run_in_executor(None, _build_index, rvc_root)
        await job.queue.put({
            "type": "log",
            "message": f"FAISS index saved: {added_index_path}",
            "phase": "index",
        })
    except Exception as exc:
        await _fail("index", f"FAISS index build failed: {exc}")
        return

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------
    job.phase = "done"
    job.progress_pct = 100
    job.status = "done"

    await _update_db_status(job.profile_id, "trained")

    elapsed = time.monotonic() - start_ts
    await job.queue.put({"type": "done", "message": "Training complete", "phase": "done"})

    print(
        json.dumps({
            "event": "training_done",
            "profile_id": job.profile_id,
            "job_id": job.job_id,
            "elapsed_s": round(elapsed, 1),
        }),
        flush=True,
    )


# ---------------------------------------------------------------------------
# TrainingManager
# ---------------------------------------------------------------------------

class TrainingManager:
    """Singleton managing in-memory training job state.

    At most one active job at a time (single-user local tool; D017).
    Jobs are keyed by profile_id in _jobs dict.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, TrainingJob] = {}

    def start_job(
        self,
        profile_id: str,
        sample_dir: str,
        rvc_root: str,
        total_epoch: int = 200,
        save_every: int = 10,
    ) -> TrainingJob:
        """Create and launch a training job for profile_id.

        Raises ValueError("job already running") if any job is currently
        in status=="training" (HTTP 409 in the router layer).

        Returns the created TrainingJob immediately (pipeline runs as a
        background asyncio task).
        """
        # Guard: only one active job at a time
        if any(j.status == "training" for j in self._jobs.values()):
            raise ValueError("job already running")

        job_id = uuid.uuid4().hex
        job = TrainingJob(job_id=job_id, profile_id=profile_id)
        self._jobs[profile_id] = job

        asyncio.create_task(
            _run_pipeline(job, sample_dir, rvc_root, total_epoch, save_every)
        )

        print(
            json.dumps({
                "event": "training_start",
                "profile_id": profile_id,
                "job_id": job_id,
                "rvc_root": rvc_root,
            }),
            flush=True,
        )

        return job

    def cancel_job(self, profile_id: str) -> None:
        """Cancel the training job for profile_id.

        Kills the process group (covers train.py + its mp.Process children).
        Sets job status to "cancelled" and updates DB to "failed".
        Silently no-ops if no job exists for profile_id.
        """
        job = self._jobs.get(profile_id)
        if job is None:
            return

        if job._proc is not None:
            try:
                os.killpg(os.getpgid(job._proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                # Process may have already exited — not an error
                try:
                    job._proc.kill()
                except (ProcessLookupError, OSError):
                    pass

        job.status = "cancelled"
        job.error = "Training cancelled by user"
        job.queue.put_nowait({
            "type": "error",
            "message": "Training cancelled by user",
            "phase": job.phase,
        })

        # Fire-and-forget DB update (best-effort; manager is sync)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_update_db_status(profile_id, "failed"))
            else:
                loop.run_until_complete(_update_db_status(profile_id, "failed"))
        except RuntimeError:
            pass  # No event loop — skip DB update on cancel in tests

    def get_job(self, profile_id: str) -> Optional[TrainingJob]:
        """Return the TrainingJob for profile_id, or None if none exists."""
        return self._jobs.get(profile_id)

    def active_job(self) -> Optional[TrainingJob]:
        """Return the first job with status=="training", or None."""
        for job in self._jobs.values():
            if job.status == "training":
                return job
        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

manager = TrainingManager()
