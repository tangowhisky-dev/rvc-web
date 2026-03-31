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
import glob
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

from backend.app.db import get_db, save_epoch_loss


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
            await job.queue.put(
                {
                    "type": "log",
                    "message": "Warning: train.log did not appear within 10s; skipping tail",
                    "phase": "train",
                }
            )
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
                    # Detect epoch-start lines: "Train Epoch: N [N%]"
                    if "Train Epoch:" in stripped:
                        import re as _re

                        m = _re.search(r"Train Epoch:\s*(\d+)", stripped)
                        if m:
                            epoch_num = int(m.group(1))
                            await job.queue.put(
                                {
                                    "type": "epoch",
                                    "message": f"Epoch {epoch_num} started",
                                    "phase": "train",
                                    "epoch": epoch_num,
                                }
                            )
                            continue  # don't also emit the raw line
                    # Detect epoch-done lines: "====> Epoch: N [timestamp] | (elapsed)"
                    if "====> Epoch:" in stripped:
                        import re as _re

                        m = _re.search(r"====> Epoch:\s*(\d+)", stripped)
                        if m:
                            epoch_num = int(m.group(1))
                            await job.queue.put(
                                {
                                    "type": "epoch_done",
                                    "message": f"Epoch {epoch_num} complete",
                                    "phase": "train",
                                    "epoch": epoch_num,
                                }
                            )
                            continue
                    await job.queue.put(
                        {
                            "type": "log",
                            "message": stripped,
                            "phase": "train",
                        }
                    )
            else:
                await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# filelist.txt builder (adapts click_train() from web.py)
# ---------------------------------------------------------------------------


def _build_filelist(project_root: str, exp_dir: str) -> None:
    """Build logs/rvc_finetune_active/filelist.txt from feature directories.

    Adapts the filelist construction logic from click_train() in web.py.
    Uses v2 feature dimensions (768) and 32k sample rate with f0 enabled.

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
        _stems(gt_wavs_dir) & _stems(feature_dir) & _stems(f0_dir) & _stems(f0nsf_dir)
    )

    spk_id = 0  # speaker id; fixed for single-speaker fine-tuning
    fea_dim = 768  # v2 feature dimension
    sr_tag = "32k"

    # Mute padding entries — absolute paths under logs/mute/ (project root)
    mute_root = os.path.join(project_root, "logs", "mute")

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
            "%s/0_gt_wavs/mute%s.wav|%s/3_feature%s/mute.npy|"
            "%s/2a_f0/mute.wav.npy|%s/2b-f0nsf/mute.wav.npy|%s"
            % (mute_root, sr_tag, mute_root, fea_dim, mute_root, mute_root, spk_id)
        )

    shuffle(opt)

    filelist_path = f"{exp_dir}/filelist.txt"
    with open(filelist_path, "w") as f:
        f.write("\n".join(opt))


# ---------------------------------------------------------------------------
# config.json writer (idempotent)
# ---------------------------------------------------------------------------


def _write_config(
    project_root: str, exp_dir: str, batch_size: int = 8, c_spk: float = 2.0
) -> None:
    """Copy backend/rvc/configs/inuse/v2/32k.json into exp_dir/config.json, patching
    training hyperparameters for MPS performance.

    log_interval=10 (not 1) is the key MPS optimization:
    - With log_interval=1, train.py calls plot_spectrogram_to_numpy 3 times
      per step (126 times/epoch at 42 steps). Each call does a MPS→CPU sync
      (.data.cpu().numpy()) which flushes the Metal command buffer and resets
      the GPU execution pipeline — measured cost: ~15s/epoch.
    - With log_interval=10, that drops to ~12 calls/epoch (~1.5s/epoch).
    - The last batch's loss line is still the one attached to epoch_done
      because _drain_stdout buffers and replaces it on each loss line seen.
      Loss accuracy is unaffected; only spectrogram images are sparser.
    """
    import json as _json

    config_save_path = os.path.join(exp_dir, "config.json")
    src = os.path.join(
        project_root, "backend", "rvc", "configs", "inuse", "v2", "32k.json"
    )

    with open(src) as _f:
        cfg = _json.load(_f)
    # log_interval=10: log every 10 batches instead of every batch.
    # Eliminates ~90% of the matplotlib MPS→CPU syncs that account for ~15s/epoch.
    # Loss values attached to epoch_done still come from the final batch since
    # _drain_stdout buffers and overwrites last_loss_line on each emitted loss line.
    cfg["train"]["log_interval"] = 10
    cfg["train"]["c_spk"] = c_spk
    with open(config_save_path, "w") as _f:
        _json.dump(cfg, _f, indent=2)


# ---------------------------------------------------------------------------
# FAISS index builder — runs as an isolated subprocess to avoid segfault
# ---------------------------------------------------------------------------
# faiss + PyTorch/MPS in the same process causes a native segfault when
# run_in_executor is used (faiss native code conflicts with MPS memory).
# Solution: run the index build as a fully separate subprocess so faiss
# never shares an address space with the uvicorn/PyTorch process.

_INDEX_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "_index_worker.py")


def _build_index(project_root: str) -> str:
    """Build FAISS IVF index by spawning an isolated subprocess.

    Returns the path to the saved added_IVF*.index file.
    Raises RuntimeError on failure.
    """
    import subprocess as _sp
    import sys as _sys

    result = _sp.run(
        [_sys.executable, _INDEX_WORKER_SCRIPT, project_root],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-2000:].strip() or "index worker exited non-zero"
        )

    # Worker prints the added index path as last stdout line
    added_path = result.stdout.strip().splitlines()[-1].strip()
    if not os.path.exists(added_path):
        raise RuntimeError(f"Index worker claimed {added_path} but file not found")
    return added_path


# ---------------------------------------------------------------------------
# Core pipeline coroutine
# ---------------------------------------------------------------------------


async def _run_pipeline(
    job: TrainingJob,
    sample_dir: str,
    project_root: str,
    total_epoch: int = 200,
    save_every: int = 10,
    batch_size: int = 8,
    profile_dir: str = "",
    prior_epochs: int = 0,
    files_changed: bool = False,
    embedder: str = "spin-v2",
    overtrain_threshold: int = 0,
    vocoder: str = "HiFi-GAN",
    c_spk: float = 2.0,
) -> None:
    """Orchestrate the full training pipeline for one TrainingJob.

    Chains: preprocess → f0 extract → feature extract → (config+filelist setup)
            → train (with log tailing) → FAISS index build → copy to profile_dir.

    project_root: absolute path to rvc-web/ (the project root). Replaces the
      old RVC_ROOT / submodule pattern. Layout under project_root:
        assets/              — pretrained weights, hubert, rmvpe
        logs/rvc_finetune_active/ — training workspace (exp_dir)
        backend/rvc/         — RVC Python package + training scripts
        backend/rvc/configs/ — v2/48k.json and inuse/ configs

    files_changed: True when audio files were added or removed since the last
      training run (needs_retraining flag from DB).  Forces a full wipe of the
      feature directories (0_gt_wavs, 1_16k_wavs, 2a_f0, 2b-f0nsf, 3_feature768)
      even for same-profile re-runs, so preprocess.py and extract_f0/feature
      both reprocess from scratch.

      Why this matters: preprocess.py names segment files by sorted position of
      the source audio file (idx0 = index in sorted(listdir(sample_dir))). If a
      new audio file sorts before an existing one, sorted indices shift and
      preprocess.py overwrites 0_gt_wavs segments with new audio — but
      extract_f0 and extract_feature skip files whose .npy already exists.
      The result is stale F0/feature data paired with new audio — silently wrong
      training targets. Wiping the feature dirs when files_changed=True forces
      full re-extraction and eliminates this mismatch.

    prior_epochs: epochs already trained for this profile (from DB). Used to:
      - compute cumulative -te target for train.py so resumed runs continue
        from the right epoch number
      - skip pretrain weights when prior_epochs > 0 (checkpoint in exp_dir
        takes over)

    All output is funnelled into job.queue as JSON messages.
    Sets job.status to "done" or "failed" on completion.
    Updates profile status in SQLite via _update_db_status().
    """
    start_ts = time.monotonic()
    exp_name = "rvc_finetune_active"
    # Training workspace: project_root/logs/rvc_finetune_active/
    exp_dir = os.path.join(project_root, "logs", exp_name)
    # backend/rvc is the cwd for all training subprocess scripts
    rvc_pkg_dir = os.path.join(project_root, "backend", "rvc")

    # ------------------------------------------------------------------
    # Seed exp_dir with prior checkpoints (resume support).
    # If the profile has checkpoints/ from a previous run, copy them into
    # exp_dir before the cleanup pass so train.py auto-resumes from them.
    # We do this BEFORE the cleanup so the files survive the wipe of the
    # non-feature directories.
    # ------------------------------------------------------------------
    ckpt_dir = os.path.join(profile_dir, "checkpoints") if profile_dir else None
    seeded_checkpoint = False
    if ckpt_dir and os.path.isdir(ckpt_dir):
        os.makedirs(exp_dir, exist_ok=True)
        for ckpt_file in ("G_latest.pth", "D_latest.pth"):
            src = os.path.join(ckpt_dir, ckpt_file)
            dst = os.path.join(exp_dir, ckpt_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                seeded_checkpoint = True
                await job.queue.put(
                    {
                        "type": "log",
                        "message": f"Seeded {ckpt_file} from profile checkpoints (resuming from epoch {prior_epochs})",
                        "phase": "setup",
                    }
                )

    # ------------------------------------------------------------------
    # Clean stale training artifacts before each run so we always start
    # fresh (no accumulated checkpoints, tfevents, or stale logs).
    # Preserve the feature extraction outputs (0_gt_wavs, 1_16k_wavs,
    # 2a_f0, 2b-f0nsf, 3_feature768) so re-runs skip those phases fast
    # when the audio hasn't changed.
    # Preserve G_latest.pth / D_latest.pth ONLY if they were just seeded
    # from this profile's own checkpoints/ dir.  If this is a fresh profile
    # (prior_epochs == 0 and nothing was seeded), delete any stale
    # checkpoints left by a previously-deleted profile so train.py starts
    # from pretrained weights instead of inheriting a stranger's weights.
    #
    # Additionally: if exp_dir was built for a *different* profile (detected
    # via a sentinel file), wipe the feature dirs too — they're stale audio.
    # ------------------------------------------------------------------
    sentinel_path = os.path.join(exp_dir, ".profile_id")
    prev_profile_id = None
    if os.path.isfile(sentinel_path):
        try:
            prev_profile_id = open(sentinel_path).read().strip()
        except OSError:
            pass

    different_profile = (
        prev_profile_id is not None and prev_profile_id != job.profile_id
    )

    # If a different profile last used this workspace, treat as completely fresh:
    # wipe feature dirs too so preprocessing reruns for the new audio.
    if different_profile:
        await job.queue.put(
            {
                "type": "log",
                "message": f"Detected stale workspace from profile {prev_profile_id[:8]}… — clearing all cached features",
                "phase": "setup",
            }
        )
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir, ignore_errors=True)
        os.makedirs(exp_dir, exist_ok=True)
        # Re-seed checkpoints after clearing (they were copied before cleanup above)
        if seeded_checkpoint and ckpt_dir:
            for ckpt_file in ("G_latest.pth", "D_latest.pth"):
                src = os.path.join(ckpt_dir, ckpt_file)
                dst = os.path.join(exp_dir, ckpt_file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
    elif files_changed:
        # Audio files were added or removed since last training run.  Wipe ALL
        # feature directories so preprocess + extract_f0 + extract_feature all
        # reprocess from scratch.  Without this, extract_f0/feature skip files
        # whose .npy exists — but those .npy files may now be paired with the
        # WRONG audio segments because sorted(listdir) indices shifted.
        _FEATURE_DIRS = {"0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"}
        _KEEP_FILES = {"G_latest.pth", "D_latest.pth"} if seeded_checkpoint else set()
        await job.queue.put(
            {
                "type": "log",
                "message": "Audio files changed — clearing cached features for full re-extraction",
                "phase": "setup",
            }
        )
        if os.path.isdir(exp_dir):
            for entry in os.listdir(exp_dir):
                if entry in _KEEP_FILES:
                    continue
                full = os.path.join(exp_dir, entry)
                try:
                    if os.path.isdir(full):
                        shutil.rmtree(full, ignore_errors=True)
                    else:
                        os.remove(full)
                except OSError:
                    pass
        os.makedirs(exp_dir, exist_ok=True)
        # Re-seed checkpoints after clearing feature dirs
        if seeded_checkpoint and ckpt_dir:
            for ckpt_file in ("G_latest.pth", "D_latest.pth"):
                src = os.path.join(ckpt_dir, ckpt_file)
                dst = os.path.join(exp_dir, ckpt_file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
    else:
        _KEEP_DIRS = {"0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"}
        _KEEP_FILES = {"G_latest.pth", "D_latest.pth"} if seeded_checkpoint else set()
        if os.path.isdir(exp_dir):
            for entry in os.listdir(exp_dir):
                if entry in _KEEP_DIRS or entry in _KEEP_FILES:
                    continue
                full = os.path.join(exp_dir, entry)
                try:
                    if os.path.isdir(full):
                        shutil.rmtree(full, ignore_errors=True)
                    else:
                        os.remove(full)
                except OSError:
                    pass
        os.makedirs(exp_dir, exist_ok=True)

    # Write sentinel so the next run can detect a profile change.
    try:
        with open(sentinel_path, "w") as f:
            f.write(job.profile_id)
    except OSError:
        pass
    os.makedirs(exp_dir, exist_ok=True)

    # Delete stale .spec.pt cache files inside 0_gt_wavs.
    # These are auto-generated from .wav files by the DataLoader, but if the audio
    # was re-preprocessed (different length segments), old .spec.pt files cause a
    # spec/wav length mismatch that makes `ids_str_max` go negative inside
    # rand_slice_segments_on_last_dim, wrapping around as a long and producing a
    # zero-size tensor — crashing the training loop immediately with RuntimeError.
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    if os.path.isdir(gt_wavs_dir):
        removed_specs = 0
        for fname in os.listdir(gt_wavs_dir):
            if fname.endswith(".spec.pt"):
                try:
                    os.remove(os.path.join(gt_wavs_dir, fname))
                    removed_specs += 1
                except OSError:
                    pass
        if removed_specs:
            await job.queue.put(
                {
                    "type": "log",
                    "message": f"Cleared {removed_specs} stale .spec.pt cache files from 0_gt_wavs",
                    "phase": "setup",
                }
            )

    n_cpu = multiprocessing.cpu_count()
    python = sys.executable

    # Base environment: inherit everything, then layer overrides
    base_env = os.environ.copy()
    base_env["PROJECT_ROOT"] = project_root
    base_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    base_env["PROJECT_ROOT"] = project_root

    def _emit_phase(phase: str, message: str) -> None:
        """Helper: put a phase-transition message to the queue (fire-and-forget for sync)."""
        job.phase = phase
        job.queue.put_nowait({"type": "phase", "message": message, "phase": phase})
        # Structured observability log
        print(
            json.dumps(
                {
                    "event": "training_phase",
                    "phase": phase,
                    "profile_id": job.profile_id,
                }
            ),
            flush=True,
        )

    async def _fail(phase: str, message: str) -> None:
        job.status = "failed"
        job.error = message
        await job.queue.put({"type": "error", "message": message, "phase": phase})
        await _update_db_status(job.profile_id, "failed")
        elapsed = time.monotonic() - start_ts
        print(
            json.dumps(
                {
                    "event": "training_failed",
                    "profile_id": job.profile_id,
                    "error": message,
                    "elapsed_s": round(elapsed, 1),
                }
            ),
            flush=True,
        )

    # ------------------------------------------------------------------
    # Phase 1: Preprocess
    # ------------------------------------------------------------------
    _emit_phase("preprocess", "Preprocessing audio samples")

    preprocess_args = [
        python,
        "infer/modules/train/preprocess.py",
        sample_dir,  # inp_root (directory, not the wav file itself)
        "32000",  # sr
        str(n_cpu),  # n_p
        exp_dir,  # exp_dir (absolute path)
        "False",  # noparallel
        "3.0",  # per (segment duration)
    ]

    ok = await _run_subprocess(
        job, preprocess_args, base_env, rvc_pkg_dir, "preprocess"
    )
    if not ok:
        await _fail("preprocess", "Preprocess phase failed (non-zero exit code)")
        return

    # ------------------------------------------------------------------
    # Resolve compute device for all subprocess phases.
    # Reads RVC_DEVICE exported by start.sh; falls back to auto-detect so
    # the pipeline works even when started manually without start.sh.
    # ------------------------------------------------------------------
    import torch as _torch_detect

    _rvc_device: str = base_env.get("RVC_DEVICE") or (
        "cuda"
        if _torch_detect.cuda.is_available()
        else "mps"
        if _torch_detect.backends.mps.is_available()
        else "cpu"
    )
    # is_half: fp16 on CUDA only (not supported on MPS or CPU)
    _is_half_str: str = "True" if _rvc_device.startswith("cuda") else "False"
    # CUDA device index for tools that accept i_gpu as a separate arg
    _cuda_device_idx: str = base_env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

    await job.queue.put(
        {
            "type": "log",
            "message": f"Using device: {_rvc_device}  is_half={_is_half_str}",
            "phase": "setup",
        }
    )

    # ------------------------------------------------------------------
    # Phase 2a: F0 Extraction
    # ------------------------------------------------------------------
    _emit_phase("extract_f0", "Extracting pitch (F0)")

    f0_env = base_env.copy()
    f0_env["PROJECT_ROOT"] = project_root
    if _rvc_device == "mps":
        f0_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    f0_args = [
        python,
        "infer/modules/train/extract_f0_print.py",
        exp_dir,  # exp_dir absolute
        str(n_cpu),
        "rmvpe",  # f0method
        _rvc_device,  # device — resolved at runtime
        _is_half_str,  # is_half
    ]

    ok = await _run_subprocess(job, f0_args, f0_env, rvc_pkg_dir, "extract_f0")
    if not ok:
        await _fail("extract_f0", "F0 extraction phase failed (non-zero exit code)")
        return

    # ------------------------------------------------------------------
    # Phase 2b: Feature Extraction (embedder)
    # ------------------------------------------------------------------
    # Resolve embedder to an absolute path under assets/.
    # "hubert" is the legacy fairseq .pt fallback (assets/hubert/hubert_base.pt).
    # All others (spin-v2, spin, contentvec) are HuggingFace directories
    # sitting directly under assets/<name>/.
    # The embedder name is stored on the profile and locked after first training.
    if embedder == "hubert":
        embedder_path = os.path.join(project_root, "assets", "hubert", "hubert_base.pt")
    else:
        # spin-v2 → assets/spin-v2, spin → assets/spin, contentvec → assets/contentvec
        embedder_path = os.path.join(project_root, "assets", embedder)

    _emit_phase("extract_feature", f"Extracting features ({embedder})")

    feat_env = base_env.copy()
    feat_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    feat_env["PROJECT_ROOT"] = project_root

    feat_args = [
        python,
        "infer/modules/train/extract_feature_print.py",
        _rvc_device,  # device — resolved at runtime
        "1",  # n_part
        "0",  # i_part
        _cuda_device_idx,  # i_gpu — CUDA device index (empty-safe for MPS/CPU)
        exp_dir,  # exp_dir absolute
        "v2",  # version
        _is_half_str,  # is_half
        embedder_path,  # embedder path (new arg)
    ]

    ok = await _run_subprocess(job, feat_args, feat_env, rvc_pkg_dir, "extract_feature")
    if not ok:
        await _fail(
            "extract_feature", "Feature extraction phase failed (non-zero exit code)"
        )
        return

    # ------------------------------------------------------------------
    # Pre-phase 3: Extract profile-level speaker embedding (if c_spk > 0)
    # ------------------------------------------------------------------
    profile_emb_path = os.path.join(exp_dir, "profile_embedding.pt")
    if c_spk > 0:
        _emit_phase("extract_profile_emb", "Extracting profile speaker embedding")
        gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
        # Use MPS if available (same logic as in train.py)
        emb_device = _rvc_device
        emb_extract_args = [
            python,
            "infer/modules/train/extract_profile_embedding.py",
            gt_wavs_dir,
            profile_emb_path,
            emb_device,
        ]
        await job.queue.put(
            {
                "type": "log",
                "message": f"Extracting profile embedding: {gt_wavs_dir} → {profile_emb_path} (device={emb_device})",
                "phase": "extract_profile_emb",
            }
        )
        ok = await _run_subprocess(
            job, emb_extract_args, base_env, rvc_pkg_dir, "extract_profile_emb"
        )
        if not ok:
            # Non-fatal — speaker embedding extraction failed, disable speaker loss
            await job.queue.put(
                {
                    "type": "log",
                    "message": "WARNING: Profile embedding extraction failed — speaker loss disabled",
                    "phase": "extract_profile_emb",
                }
            )
            c_spk = 0.0
        else:
            await job.queue.put(
                {
                    "type": "log",
                    "message": f"Profile speaker embedding saved to {profile_emb_path}",
                    "phase": "extract_profile_emb",
                }
            )
    else:
        await job.queue.put(
            {
                "type": "log",
                "message": f"Speaker loss disabled (c_spk={c_spk})",
                "phase": "setup",
            }
        )

    # ------------------------------------------------------------------
    # Pre-phase 3 setup: config.json and filelist.txt
    # ------------------------------------------------------------------
    try:
        _write_config(project_root, exp_dir, batch_size=batch_size, c_spk=c_spk)
        _build_filelist(project_root, exp_dir)
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
    train_env["PROJECT_ROOT"] = project_root
    # MPS memory tuning — Apple Silicon only.
    # High watermark 0.0 = no limit; shared memory reduces copy overhead for
    # checkpoint saves and gradient norm computation on unified memory.
    if _rvc_device == "mps":
        train_env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        train_env.setdefault("PYTORCH_MPS_PREFER_SHARED_MEMORY", "1")
        train_env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Checkpoint save schedule: every 25% of the run or every 10 epochs,
    # whichever is smaller — ensures at least 4 saves and no more than
    # one save per 10 epochs on longer runs.
    effective_save_every = max(1, min(10, total_epoch // 4))

    # Cumulative epoch target: train.py resumes from prior_epochs (loaded
    # from G_latest.pth) and runs until it hits this number.
    cumulative_epochs = prior_epochs + total_epoch

    # Pretrain weights — absolute paths into project_root/assets/
    # RefineGAN ships its own pretrained G/D under assets/refinegan/
    if prior_epochs == 0:
        if vocoder == "RefineGAN":
            pretrain_g = os.path.join(project_root, "assets", "refinegan", "f0G32k.pth")
            pretrain_d = os.path.join(project_root, "assets", "refinegan", "f0D32k.pth")
        else:
            pretrain_g = os.path.join(
                project_root, "assets", "pretrained_v2", "f0G32k.pth"
            )
            pretrain_d = os.path.join(
                project_root, "assets", "pretrained_v2", "f0D32k.pth"
            )
    else:
        pretrain_g = ""
        pretrain_d = ""

    await job.queue.put(
        {
            "type": "log",
            "message": (
                f"Training epochs {prior_epochs + 1}–{cumulative_epochs} "
                f"(save every {effective_save_every})"
                + (" [resume]" if prior_epochs > 0 else " [fresh]")
            ),
            "phase": "train",
        }
    )

    # GPU cache: on CUDA/MPS cache dataset to device memory for speed.
    # On CPU set to 0 — there is no GPU memory to cache into and the flag
    # causes confusion (train.py still tries to move tensors to "cuda:0").
    _cache_gpu = "1" if _rvc_device in ("cuda", "mps") else "0"

    # GPU index string for train.py (-g flag).  train.py uses this to set
    # CUDA_VISIBLE_DEVICES internally.  Pass through whatever the caller
    # already exported so multi-GPU Linux works correctly.
    _gpus_str = (
        base_env.get("CUDA_VISIBLE_DEVICES", "0") if _rvc_device == "cuda" else "0"
    )

    train_args = [
        python,
        "infer/modules/train/train.py",
        "-e",
        exp_dir,  # absolute path — avoids cwd-relative ./logs/ resolution in train.py
        "-sr",
        "32k",
        "-f0",
        "1",
        "-bs",
        str(batch_size),
        "-te",
        str(cumulative_epochs),
        "-se",
        str(effective_save_every),
        "-g",
        _gpus_str,  # GPU index(es) — prevents train.py overriding CUDA_VISIBLE_DEVICES
        "-pg",
        pretrain_g,
        "-pd",
        pretrain_d,
        "-l",
        "1",  # save latest only (G_latest.pth / D_latest.pth)
        "-c",
        _cache_gpu,  # cache dataset in GPU memory (0 on CPU to avoid OOM)
        "-sw",
        "0",  # save_small_model at end only (we handle artifacts ourselves)
        "-v",
        "v2",
        "-a",
        "",
        "-voc",
        vocoder,
        "-emb",
        embedder,
    ]

    train_proc = await asyncio.create_subprocess_exec(
        *train_args,
        stdout=asyncio_subprocess.PIPE,  # capture stdout — root logger writes here (basicConfig)
        stderr=asyncio_subprocess.PIPE,  # capture stderr so crashes surface in job error
        env=train_env,
        cwd=rvc_pkg_dir,
    )
    job._proc = train_proc

    log_path = os.path.join(exp_dir, "train.log")

    # Tail train.log concurrently while train.py runs (belt + suspenders —
    # in case the FileHandler works in some configurations)
    tail_task = asyncio.create_task(_tail_train_log(job, log_path))

    # Stream stdout into job queue — the root logger's StreamHandler (basicConfig)
    # writes all INFO/DEBUG lines here. This is the primary log source since the
    # FileHandler's file handle is lost when the logger is pickled into mp.Process.
    # Parse epoch markers to emit structured epoch/epoch_done events.
    async def _drain_stdout() -> None:
        if train_proc.stdout is None:
            return
        import re as _re

        # With log_interval=10, loss lines fire every 10 batches (4-5 times/epoch).
        # Buffer the latest loss line and attach it to epoch_done.
        # All mid-epoch log noise is suppressed.
        last_loss_line: str = ""

        # Overtraining detection state
        # Mirrors the logic in ultimate-rvc's train.py but runs here so we can
        # terminate the subprocess cleanly without modifying train.py.
        _ot_lowest_gen: float = float("inf")
        _ot_consecutive: int = 0
        _ot_min_delta: float = 0.004  # must improve by at least this to count
        _ot_triggered: bool = False

        async for raw_line in train_proc.stdout:
            line = raw_line.decode(errors="replace").rstrip()
            if not line:
                continue
            # Filter faiss loader spam
            if line.startswith("DEBUG:faiss") or line.startswith("INFO:faiss"):
                continue
            # Strip logger prefix "INFO:name:" / "DEBUG:name:"
            clean = _re.sub(r"^(INFO|DEBUG|WARNING|ERROR):[^:]+:", "", line).strip()
            if not clean:
                continue

            # Loss line — buffer silently; shown once at epoch_done
            if "loss_disc=" in clean or "loss_gen=" in clean:
                last_loss_line = clean
                continue

            # "Train Epoch: N [pct%]" — suppress (fires every batch with log_interval=1)
            if "Train Epoch:" in clean:
                continue

            # "[step, lr]" list line — suppress
            if clean.startswith("[") and "," in clean:
                continue

            # Epoch done: "====> Epoch: N [timestamp] | (elapsed)"
            if "====> Epoch:" in clean:
                m = _re.search(r"====> Epoch:\s*(\d+)", clean)
                if m:
                    epoch_num = int(m.group(1))
                    elapsed_m = _re.search(r"\(([^)]+)\)", clean)
                    elapsed = elapsed_m.group(1) if elapsed_m else ""
                    loss_suffix = f" | {last_loss_line}" if last_loss_line else ""
                    # Parse individual loss scalars from the buffered loss line
                    # so the frontend can plot them without parsing log text.
                    losses: dict = {}
                    for key in (
                        "loss_disc",
                        "loss_gen",
                        "loss_fm",
                        "loss_mel",
                        "loss_kl",
                        "loss_spk",
                    ):
                        lm = _re.search(rf"{key}=([0-9.]+)", last_loss_line)
                        if lm:
                            try:
                                losses[key] = float(lm.group(1))
                            except ValueError:
                                pass

                    # Overtraining detection — track generator loss trend.
                    # Only active when overtrain_threshold > 0.
                    ot_label = ""
                    if overtrain_threshold > 0 and "loss_gen" in losses:
                        gen_loss = losses["loss_gen"]
                        if gen_loss < _ot_lowest_gen - _ot_min_delta:
                            _ot_lowest_gen = gen_loss
                            _ot_consecutive = 0
                        else:
                            _ot_consecutive += 1
                        if _ot_consecutive >= overtrain_threshold and not _ot_triggered:
                            _ot_triggered = True
                            ot_label = f" ⚠ overtraining detected ({_ot_consecutive} epochs no improvement)"
                            await job.queue.put(
                                {
                                    "type": "log",
                                    "message": f"Stopping early: generator loss has not improved for {_ot_consecutive} consecutive epochs (threshold={overtrain_threshold}).",
                                    "phase": "train",
                                }
                            )
                            # Terminate train.py cleanly
                            try:
                                train_proc.terminate()
                            except Exception:
                                pass

                    msg: dict = {
                        "type": "epoch_done",
                        "message": f"Epoch {epoch_num} ({elapsed}){loss_suffix}{ot_label}",
                        "phase": "train",
                        "epoch": epoch_num,
                    }
                    if losses:
                        msg["losses"] = losses
                    if overtrain_threshold > 0:
                        msg["overtrain_consecutive"] = _ot_consecutive
                    await job.queue.put(msg)
                    # Persist losses to DB so the chart survives restarts
                    if losses:
                        try:
                            await save_epoch_loss(job.profile_id, epoch_num, losses)
                        except Exception as _e:
                            logger.warning(f"epoch_loss DB write failed: {_e}")
                    last_loss_line = ""
                    continue

            await job.queue.put({"type": "log", "message": clean, "phase": "train"})

    stdout_task = asyncio.create_task(_drain_stdout())

    # Also stream stderr lines into the job queue so crash tracebacks are visible
    # in the UI in real time (train.py child process errors go to stderr).
    async def _drain_stderr() -> None:
        if train_proc.stderr is None:
            return
        async for raw_line in train_proc.stderr:
            line = raw_line.decode(errors="replace").rstrip()
            if line:
                await job.queue.put(
                    {"type": "log", "message": f"[stderr] {line}", "phase": "train"}
                )

    stderr_task = asyncio.create_task(_drain_stderr())

    await train_proc.wait()
    job._proc = None

    # Give the readers a moment to drain remaining lines
    await asyncio.sleep(0.5)

    tail_task.cancel()
    stdout_task.cancel()
    stderr_task.cancel()
    for task in (tail_task, stdout_task, stderr_task):
        try:
            await task
        except asyncio.CancelledError:
            pass

    if train_proc.returncode != 0:
        # SIGTERM from overtraining detection (returncode = -15 on Unix) is a
        # clean intentional stop — treat it as a successful partial run and
        # proceed to index building + artifact copy.
        import signal as _signal

        is_overtrain_stop = overtrain_threshold > 0 and train_proc.returncode in (
            -_signal.SIGTERM,
            _signal.SIGTERM,
        )
        if not is_overtrain_stop:
            # stderr already drained into job queue by _drain_stderr above
            err_msg = f"Train phase failed (exit code {train_proc.returncode}) — see [stderr] lines above"
            await _fail("train", err_msg)
            return
        # Overtraining stop — log it and continue to artifact phase
        await job.queue.put(
            {
                "type": "log",
                "message": "Training stopped early by overtraining detector — proceeding to index build.",
                "phase": "train",
            }
        )
    # On success: stderr was already drained by the concurrent _drain_stderr task

    # ------------------------------------------------------------------
    # Phase 4: FAISS index (CPU-bound, run in thread executor)
    # ------------------------------------------------------------------
    _emit_phase("index", "Building FAISS index")

    loop = asyncio.get_event_loop()
    try:
        added_index_path = await loop.run_in_executor(None, _build_index, project_root)
        await job.queue.put(
            {
                "type": "index_done",
                "message": f"FAISS index built: {os.path.basename(added_index_path)}",
                "phase": "index",
            }
        )
    except Exception as exc:
        await _fail("index", f"FAISS index build failed: {exc}")
        return

    # ------------------------------------------------------------------
    # Phase 5: Persist checkpoints + index to profile directory
    # ------------------------------------------------------------------
    # G_latest.pth / D_latest.pth → profile_dir/checkpoints/  (resume, latest only)
    # assets/weights/{exp}.pth    → profile_dir/model_infer.pth (inference, ~54 MB)
    # added_IVF*.index            → profile_dir/model.index
    #
    # Two files serve two purposes:
    #   - checkpoints/G_latest.pth: full resume checkpoint (431 MB, optimizer
    #     state included). Seeded back into exp_dir before the next training run.
    #   - model_infer.pth: inference-ready fp16 weights (~54 MB) written by
    #     save_small_model() at the end of every completed run. This is what
    #     rtrvc.py / load_synthesizer() expects (has 'weight', 'config', 'f0',
    #     'sr', 'version' keys).
    dest_model_path: Optional[str] = None  # → DB model_path (checkpoint for resume)
    dest_infer_path: Optional[str] = None  # → DB inference_model_path
    dest_index_path: Optional[str] = None

    g_latest = os.path.join(exp_dir, "G_latest.pth")
    d_latest = os.path.join(exp_dir, "D_latest.pth")
    # assets/weights/{exp}.pth is written by save_small_model() inside train.py
    # using PROJECT_ROOT → project_root/assets/weights/
    weights_pth = os.path.join(project_root, "assets", "weights", f"{exp_name}.pth")

    if not os.path.exists(g_latest):
        # Fallback: find any G_{step}.pth; pick the highest step
        g_candidates = sorted(glob.glob(os.path.join(exp_dir, "G_*.pth")))
        if g_candidates:
            g_latest = g_candidates[-1]
            await job.queue.put(
                {
                    "type": "log",
                    "message": f"G_latest.pth not found; using {os.path.basename(g_latest)}",
                    "phase": "index",
                }
            )

    if profile_dir:
        dest_ckpt_dir = os.path.join(profile_dir, "checkpoints")
        os.makedirs(dest_ckpt_dir, exist_ok=True)

        dest_g = os.path.join(dest_ckpt_dir, "G_latest.pth")
        dest_d = os.path.join(dest_ckpt_dir, "D_latest.pth")
        dest_infer_path = os.path.join(profile_dir, "model_infer.pth")
        dest_index_path = os.path.join(profile_dir, "model.index")

        try:
            if os.path.exists(g_latest):
                shutil.copy2(g_latest, dest_g)
                dest_model_path = dest_g
                await job.queue.put(
                    {
                        "type": "log",
                        "message": f"Saved resume checkpoint → {dest_g}",
                        "phase": "index",
                    }
                )
            else:
                await job.queue.put(
                    {
                        "type": "log",
                        "message": f"Warning: no checkpoint found at {g_latest}",
                        "phase": "index",
                    }
                )

            if os.path.exists(d_latest):
                shutil.copy2(d_latest, dest_d)

            if os.path.exists(weights_pth):
                shutil.copy2(weights_pth, dest_infer_path)
                await job.queue.put(
                    {
                        "type": "log",
                        "message": f"Saved inference model → {dest_infer_path}",
                        "phase": "index",
                    }
                )
            else:
                await job.queue.put(
                    {
                        "type": "log",
                        "message": f"Warning: inference weights not found at {weights_pth}",
                        "phase": "index",
                    }
                )
                dest_infer_path = None

            shutil.copy2(added_index_path, dest_index_path)
            await job.queue.put(
                {
                    "type": "log",
                    "message": f"Saved index → {dest_index_path}",
                    "phase": "index",
                }
            )
        except Exception as exc:
            await job.queue.put(
                {
                    "type": "log",
                    "message": f"Warning: failed to copy artifacts to profile dir: {exc}",
                    "phase": "index",
                }
            )
            dest_model_path = None
            dest_infer_path = None
            dest_index_path = None
    else:
        # Legacy / no profile_dir
        dest_model_path = g_latest if os.path.exists(g_latest) else None
        dest_infer_path = weights_pth if os.path.exists(weights_pth) else None
        dest_index_path = added_index_path

    # ------------------------------------------------------------------
    # Success — update DB
    # ------------------------------------------------------------------
    job.phase = "done"
    job.progress_pct = 100
    job.status = "done"

    new_total_epochs = prior_epochs + total_epoch

    async with get_db() as db:
        await db.execute(
            """UPDATE profiles
               SET status = 'trained',
                   model_path = ?,
                   checkpoint_path = ?,
                   index_path = ?,
                   total_epochs_trained = ?,
                   needs_retraining = 0
               WHERE id = ?""",
            (
                dest_infer_path,
                dest_model_path,
                dest_index_path,
                new_total_epochs,
                job.profile_id,
            ),
        )
        await db.commit()

    elapsed = time.monotonic() - start_ts
    await job.queue.put(
        {"type": "done", "message": "Training complete", "phase": "done"}
    )

    print(
        json.dumps(
            {
                "event": "training_done",
                "profile_id": job.profile_id,
                "job_id": job.job_id,
                "elapsed_s": round(elapsed, 1),
            }
        ),
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
        project_root: str,
        total_epoch: int = 200,
        save_every: int = 10,
        batch_size: int = 8,
        profile_dir: str = "",
        prior_epochs: int = 0,
        files_changed: bool = False,
        embedder: str = "spin-v2",
        overtrain_threshold: int = 0,
        vocoder: str = "HiFi-GAN",
        c_spk: float = 2.0,
    ) -> TrainingJob:
        """Create and launch a training job for profile_id.

        profile_dir: absolute path to data/profiles/{id}/ — checkpoints/
        and model.index are stored here on training completion.

        prior_epochs: total epochs already trained for this profile (from
        DB total_epochs_trained). Used to compute cumulative -te target and
        to skip pretrained weights on resume runs.

        files_changed: True when audio files were added or removed since the
        last training run (DB needs_retraining=1). Forces full re-extraction
        of preprocess/F0/feature outputs even for the same profile, preventing
        stale F0 data being paired with new audio segments.

        embedder: feature embedder name, e.g. "spin-v2". Locked after first run.

        overtrain_threshold: stop training when generator loss fails to improve
        for this many consecutive epochs. 0 = disabled.

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
            _run_pipeline(
                job,
                sample_dir,
                project_root,
                total_epoch,
                save_every,
                batch_size,
                profile_dir,
                prior_epochs,
                files_changed,
                embedder,
                overtrain_threshold,
                vocoder,
                c_spk,
            )
        )

        print(
            json.dumps(
                {
                    "event": "training_start",
                    "profile_id": profile_id,
                    "job_id": job_id,
                    "project_root": project_root,
                }
            ),
            flush=True,
        )

        return job

    def cancel_job(self, profile_id: str) -> None:
        """Cancel the training job for profile_id.

        Kills only the subprocess (NOT the process group — killpg would kill
        uvicorn since train.py inherits our process group on macOS).
        Sets job status to "cancelled" and updates DB to "failed".
        Silently no-ops if no job exists for profile_id.
        """
        job = self._jobs.get(profile_id)
        if job is None:
            return

        if job._proc is not None:
            try:
                # Kill only the direct subprocess and its children via psutil,
                # NOT os.killpg which would kill uvicorn's process group too.
                import psutil

                try:
                    parent = psutil.Process(job._proc.pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    parent.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            except ImportError:
                # psutil not available — fall back to direct kill (no killpg)
                try:
                    job._proc.kill()
                except (ProcessLookupError, OSError):
                    pass

        job.status = "cancelled"
        job.error = "Training cancelled by user"
        job.queue.put_nowait(
            {
                "type": "error",
                "message": "Training cancelled by user",
                "phase": job.phase,
            }
        )

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
