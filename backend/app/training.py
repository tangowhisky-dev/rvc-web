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
    total_epoch: int = 0
    batch_size: int = 8
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
# DB poller for RVC training — replaces log-file tailing
# ---------------------------------------------------------------------------

_RVC_POLL_INTERVAL = 2.0  # seconds between epoch_losses polls


async def _poll_rvc_epochs(
    job: "TrainingJob",
    profile_id: str,
    prior_epochs: int,
    total_epoch: int,
    overtrain_threshold: int,
    proc_done: asyncio.Event,
    train_start_ts: float = 0.0,
) -> None:
    """Poll epoch_losses every 2s; emit epoch_done events with DB losses.

    Replaces stdout/logfile parsing for epoch detection.  train.py's
    TrainingDBWriter already writes losses to DB before emitting the
    '====> Epoch:' log line, so polling always gets complete data.

    Stops two seconds after proc_done is set (one final sweep to catch
    the last epoch written before subprocess exit).
    """
    _seen_epoch: int = prior_epochs
    _best_avg_gen: float = float("inf")
    _best_epoch: int = prior_epochs
    _min_delta: float = 0.004
    _ot_consecutive: int = 0
    _ot_triggered: bool = False

    # Time tracking — rolling window of recent epoch timestamps for ETA
    _epoch_ts: list[float] = []          # monotonic timestamps when each epoch completed
    _WINDOW = 5                           # use last 5 epochs for secs/epoch estimate
    if train_start_ts == 0.0:
        train_start_ts = time.monotonic()

    # Seed OT state from DB on resume
    if overtrain_threshold > 0 and prior_epochs > 0:
        try:
            async with get_db() as _db:
                row = await _db.execute_fetchall(
                    "SELECT best_avg_gen_loss, best_epoch FROM profiles WHERE id = ?",
                    (profile_id,),
                )
                if row and row[0][0] is not None:
                    _best_avg_gen = float(row[0][0])
                    _best_epoch = int(row[0][1] or prior_epochs)
        except Exception:
            pass

    while True:
        await asyncio.sleep(_RVC_POLL_INTERVAL)

        try:
            async with get_db() as _db:
                rows = await _db.execute_fetchall(
                    """SELECT epoch, loss_mel, loss_gen, loss_disc,
                              loss_fm, loss_kl, loss_spk
                       FROM epoch_losses
                       WHERE profile_id = ? AND epoch > ?
                       ORDER BY epoch ASC""",
                    (profile_id, _seen_epoch),
                )
        except Exception:
            rows = []

        for row in rows:
            epoch_num = int(row[0])
            losses = {
                "loss_mel":  row[1],
                "loss_gen":  row[2],
                "loss_disc": row[3],
                "loss_fm":   row[4],
                "loss_kl":   row[5],
                "loss_spk":  row[6],
            }

            # Record timestamp for this epoch
            now_ts = time.monotonic()
            _epoch_ts.append(now_ts)
            if len(_epoch_ts) > _WINDOW:
                _epoch_ts.pop(0)

            # Overtraining detection
            ot_label = ""
            if overtrain_threshold > 0 and not _ot_triggered:
                cur_mel = losses.get("loss_mel") or float("inf")
                if cur_mel < _best_avg_gen - _min_delta:
                    _ot_consecutive = 0
                    _best_avg_gen = cur_mel
                    _best_epoch = epoch_num
                else:
                    _ot_consecutive += 1

                if _ot_consecutive >= overtrain_threshold:
                    _ot_triggered = True
                    ot_label = f" ⚠ overtraining ({_ot_consecutive} epochs no improvement)"
                    await job.queue.put({
                        "type": "log",
                        "message": (
                            f"Stopping early at epoch {epoch_num}: mel loss has not "
                            f"improved for {_ot_consecutive} consecutive epochs "
                            f"(threshold={overtrain_threshold}). "
                            f"Best was epoch {_best_epoch} "
                            f"(avg_mel={_best_avg_gen:.4f})."
                        ),
                        "phase": "train",
                    })
                    try:
                        if job._proc is not None:
                            job._proc.terminate()
                    except Exception:
                        pass

            # Check best-epoch flag from DB
            try:
                async with get_db() as _db2:
                    brow = await _db2.execute_fetchall(
                        "SELECT best_epoch FROM profiles WHERE id = ?",
                        (profile_id,),
                    )
                    db_best = int(brow[0][0]) if brow and brow[0][0] else _best_epoch
            except Exception:
                db_best = _best_epoch

            is_best = (db_best == epoch_num)
            if is_best:
                _best_epoch = epoch_num

            msg: dict = {
                "type": "epoch_done",
                "message": f"Epoch {epoch_num}{ot_label}",
                "phase": "train",
                "epoch": epoch_num,
                "is_best": is_best,
                "best_epoch": _best_epoch,
                "best_avg_gen": round(_best_avg_gen, 4) if _best_avg_gen < float("inf") else None,
                "losses": losses,
            }
            if overtrain_threshold > 0:
                msg["overtrain_consecutive"] = _ot_consecutive
            await job.queue.put(msg)
            _seen_epoch = epoch_num

            # Emit time stats log line (mirrors Beatrice 2 pattern)
            elapsed_s = now_ts - train_start_ts
            epochs_done = epoch_num - prior_epochs  # epochs completed this session
            epochs_remaining = total_epoch - epoch_num

            # secs/epoch from rolling window of recent epoch timestamps
            secs_per_epoch: float = 0.0
            if len(_epoch_ts) >= 2:
                window_secs = _epoch_ts[-1] - _epoch_ts[0]
                window_epochs = len(_epoch_ts) - 1
                secs_per_epoch = window_secs / window_epochs
            elif epochs_done > 0:
                secs_per_epoch = elapsed_s / epochs_done

            parts = [f"Epoch {epoch_num}/{total_epoch}"]

            # secs/epoch or mins/epoch
            if secs_per_epoch >= 60:
                parts.append(f"{secs_per_epoch / 60:.1f}min/epoch")
            elif secs_per_epoch >= 1:
                parts.append(f"{secs_per_epoch:.1f}s/epoch")

            # elapsed
            e_h, e_rem = divmod(int(elapsed_s), 3600)
            e_m, e_s   = divmod(e_rem, 60)
            if e_h:
                parts.append(f"elapsed {e_h}h{e_m:02d}m")
            else:
                parts.append(f"elapsed {e_m}m{e_s:02d}s")

            # ETA
            if secs_per_epoch > 0 and epochs_remaining > 0:
                remaining_s = epochs_remaining * secs_per_epoch
                r_h, r_rem = divmod(int(remaining_s), 3600)
                r_m, r_s   = divmod(r_rem, 60)
                if r_h:
                    parts.append(f"ETA {r_h}h{r_m:02d}m")
                else:
                    parts.append(f"ETA {r_m}m{r_s:02d}s")
            elif epochs_remaining == 0:
                parts.append("done")

            await job.queue.put({
                "type": "log",
                "message": "  ".join(parts),
                "phase": "train",
            })

        # Stop one cycle after proc exits (catches last epoch)
        if proc_done.is_set():
            break


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
    project_root: str,
    exp_dir: str,
    batch_size: int = 8,
    c_spk: float = 2.0,
    loss_mode: str = "classic",
    vocoder: str = "HiFi-GAN",
    pretrain_g: str = "",
    adv_loss: str = "lsgan",
    kl_anneal: bool = False,
    kl_anneal_epochs: int = 40,
    optimizer: str = "adamw",
    use_reference_encoder: bool = False,
    reference_encoder_channels: int = 256,
    ref_clip_seconds: float = 6.0,
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

    learning_rate auto-scaling (sqrt rule):
    - The pretrained weights were tuned at batch_size=32, LR=1e-4.
    - AdamW with a smaller batch sees noisier gradients; the sqrt scaling rule
      compensates: LR_eff = LR_base * sqrt(batch_size / reference_batch_size).
    - At batch_size=8:  LR = 1e-4 * sqrt(8/32)  = 5e-5.
    - At batch_size=16: LR = 1e-4 * sqrt(16/32) ≈ 7.07e-5.
    - At batch_size=32: LR = 1e-4 * sqrt(32/32) = 1e-4 (unchanged, matches CUDA baseline).
    - This eliminates the ~2x effective over-learning-rate that caused noisy
      loss curves and a higher loss plateau on MPS vs CUDA.

    fp16_run disabled on MPS (fp32 always):
    - MPS fp16 AMP is experimental and has two compounding instabilities:
      1. 137 weight_norm hooks: ‖weight_v‖ can underflow to 0 in fp16 → NaN
         gradient → GradScaler halves scale; with growth_interval=2000 and
         ~125 steps/epoch (bs=8), scale recovery takes ~16 epochs per event.
      2. CUDA uses higher-precision fused ops for reductions; MPS does not.
    - fp16 is valuable on CUDA to reduce off-chip VRAM bandwidth; MPS unified
      memory has no off-chip bandwidth to save, so fp16 has no throughput benefit.
    - On CUDA torch.cuda.is_available() is True → fp16_run=True (unchanged).
    - On MPS torch.cuda.is_available() is False → fp16_run=False (fp32).

    spk_embed_dim alignment for RefineGAN:
    - The pretrained RefineGAN f0G32k.pth was trained with spk_embed_dim=129
      but 32k.json has spk_embed_dim=109 (the HiFi-GAN pretrain value).
    - Passing spk_embed_dim=109 to the model then loading pretrain weights
      with shape [129, 256] causes a hard RuntimeError in load_state_dict.
    - Fix: when pretrain_g is provided, read emb_g.weight.shape[0] from the
      checkpoint and use that as spk_embed_dim in config.json. This is safe
      for HiFi-GAN too (its pretrain also has 109, matching the config).
    """
    import json as _json
    import math as _math
    import torch as _torch

    config_save_path = os.path.join(exp_dir, "config.json")
    src = os.path.join(
        project_root, "backend", "rvc", "configs", "inuse", "v2", "32k.json"
    )

    with open(src) as _f:
        cfg = _json.load(_f)

    # log_interval=10: log every 10 batches instead of every batch.
    # Eliminates ~90% of the matplotlib MPS→CPU syncs that account for ~15s/epoch.
    cfg["train"]["log_interval"] = 10
    cfg["train"]["c_spk"] = c_spk
    cfg["train"]["loss_mode"] = loss_mode
    cfg["train"]["adv_loss"] = adv_loss
    cfg["train"]["kl_anneal"] = int(kl_anneal)
    cfg["train"]["kl_anneal_epochs"] = max(1, kl_anneal_epochs)
    cfg["train"]["optimizer"] = optimizer

    # Auto-scale learning rate by sqrt(batch_size / reference_batch_size).
    # Reference: pretrained weights were optimised at batch_size=32, LR=1e-4.
    # Smaller batches produce noisier gradients; the sqrt rule keeps the
    # effective per-sample update magnitude constant for AdamW.
    _REFERENCE_BATCH_SIZE = 32
    _BASE_LR = 1e-4
    _scaled_lr = _BASE_LR * _math.sqrt(batch_size / _REFERENCE_BATCH_SIZE)
    cfg["train"]["learning_rate"] = round(_scaled_lr, 7)

    # Disable fp16 on MPS — train in fp32 on Apple Silicon.
    # MPS fp16 has two compounding instabilities:
    #   1. weight_norm (137 hooks in this model) computes weight_g × (weight_v / ‖weight_v‖).
    #      In fp16, ‖weight_v‖ can underflow to zero for small weights early in training
    #      → NaN gradient → GradScaler halves its scale. With growth_interval=2000 and
    #      ~125 steps/epoch on MPS (bs=8), scale recovery takes ~16 epochs per NaN event.
    #   2. MPS fp16 AMP is still experimental; CUDA uses higher-precision fused ops
    #      for reductions that reduce inf/NaN occurrence significantly.
    # fp16 is valuable on CUDA primarily to reduce off-chip VRAM bandwidth — MPS uses
    # unified memory so there is no off-chip bandwidth to save. fp32 on MPS is correct.
    cfg["train"]["fp16_run"] = _torch.cuda.is_available()

    # Align spk_embed_dim with the pretrained weights.
    # The embedding table shape is [spk_embed_dim, gin_channels]. If the
    # checkpoint was trained with a different spk_embed_dim than 32k.json
    # declares (RefineGAN: 129 vs HiFi-GAN: 109), load_state_dict crashes on
    # a fresh run (strict=True path) or silently resets emb_g to random on
    # resume (strict=False path in load_checkpoint). Both are wrong.
    #
    # Strategy: read emb_g.weight.shape[0] from whichever checkpoint is about
    # to be loaded:
    #   - Fresh run  → pretrain_g  (assets/refinegan/f0G32k.pth etc.)
    #   - Resume run → G_latest.pth already seeded into exp_dir at this point
    _ckpt_to_probe = pretrain_g if pretrain_g else os.path.join(exp_dir, "G_latest.pth")
    if os.path.isfile(_ckpt_to_probe):
        try:
            ckpt = _torch.load(_ckpt_to_probe, map_location="cpu", weights_only=True)
            emb_shape = ckpt.get("model", {}).get("emb_g.weight")
            if emb_shape is not None:
                cfg["model"]["spk_embed_dim"] = emb_shape.shape[0]
        except Exception:
            pass  # leave config value as-is; train.py will surface the mismatch

    # Reference encoder flags — stored at top level so train.py and hps can read them
    cfg["use_reference_encoder"] = use_reference_encoder
    cfg["reference_encoder_channels"] = reference_encoder_channels
    cfg["ref_clip_seconds"] = ref_clip_seconds

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
    loss_mode: str = "classic",
    adv_loss: str = "lsgan",
    kl_anneal: bool = False,
    kl_anneal_epochs: int = 40,
    optimizer: str = "adamw",
    use_reference_encoder: bool = False,
    reference_encoder_channels: int = 256,
    ref_clip_seconds: float = 6.0,
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

    # Guard: prior_epochs > 0 but checkpoint missing — treat as fresh run rather
    # than starting from random init with no pretrain weights, which would silently
    # produce garbage output.  Reset prior_epochs so pretrain weights are loaded.
    if prior_epochs > 0 and not seeded_checkpoint:
        await job.queue.put(
            {
                "type": "log",
                "message": (
                    f"WARNING: DB records {prior_epochs} prior epochs but no checkpoint "
                    "found in profile checkpoints/ — treating as fresh run with pretrain weights."
                ),
                "phase": "setup",
            }
        )
        prior_epochs = 0

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
        _KEEP_FILES = ({"G_latest.pth", "D_latest.pth", "G_best.pth"} if seeded_checkpoint else set()) | {"profile_embedding.pt"}
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
            for ckpt_file in ("G_latest.pth", "D_latest.pth", "G_best.pth"):
                src = os.path.join(ckpt_dir, ckpt_file)
                dst = os.path.join(exp_dir, ckpt_file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
    else:
        _KEEP_DIRS = {"0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"}
        _KEEP_FILES = ({"G_latest.pth", "D_latest.pth", "G_best.pth"} if seeded_checkpoint else set()) | {"profile_embedding.pt"}
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
    # Skip if same profile, no audio changes, and 0_gt_wavs already populated.
    # Preprocess is expensive and idempotent — safe to skip when inputs unchanged.
    # Always runs when: different_profile, files_changed, or 0_gt_wavs empty/missing.
    # ------------------------------------------------------------------
    gt_wavs_dir_check = os.path.join(exp_dir, "0_gt_wavs")
    _gt_wavs_exist = (
        (
            os.path.isdir(gt_wavs_dir_check)
            and any(f.endswith(".wav") for f in os.listdir(gt_wavs_dir_check))
        )
        if os.path.isdir(gt_wavs_dir_check)
        else False
    )
    _skip_preprocess = (
        (not different_profile) and (not files_changed) and _gt_wavs_exist
    )

    if _skip_preprocess:
        await job.queue.put(
            {
                "type": "log",
                "message": "Skipping preprocess — same profile, audio unchanged, 0_gt_wavs present",
                "phase": "preprocess",
            }
        )
    else:
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
    # F0 worker count: GPU is GPU-bound — forking N processes wastes VRAM with
    # zero throughput gain.  CPU is genuinely parallel; cap at 8 (I/O bound).
    _f0_n_proc = 1 if _rvc_device in ("cuda", "mps") else min(n_cpu, 8)

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
        str(_f0_n_proc),  # n_p — 1 for GPU (prevent forking 80 RMVPE processes), capped at 8 for CPU
        "rmvpe",  # f0method
        _rvc_device,  # device — resolved at runtime
        _is_half_str,  # is_half
    ]

    ok = await _run_subprocess(job, f0_args, f0_env, rvc_pkg_dir, "extract_f0")
    if not ok:
        await _fail("extract_f0", "F0 extraction phase failed (non-zero exit code)")
        return

    # Compute and save mean F0 of profile audio for auto pitch-shift at inference.
    # Uses geometric mean (log-domain average) over all voiced frames — same method
    # as Beatrice 2 trainer. Saved to profile_dir/speaker_f0.json.
    if profile_dir:
        try:
            import math as _math
            import numpy as _np
            import torch as _torch
            import torchaudio as _torchaudio

            _audio_files = sorted([
                os.path.join(sample_dir, f) for f in os.listdir(sample_dir)
                if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a"))
            ])
            _sum_log_f0 = 0.0
            _n_frames = 0
            for _af in _audio_files:
                try:
                    import pyworld as _pyworld
                    _wav, _sr = _torchaudio.load(_af)
                    _mono = _wav.mean(0).numpy().astype(_np.float64)
                    if _sr != 16000:
                        import torchaudio.functional as _F
                        _mono = _F.resample(
                            _torch.from_numpy(_mono), _sr, 16000
                        ).numpy().astype(_np.float64)
                        _sr = 16000
                    _f0, _ = _pyworld.dio(_mono, _sr)
                    _voiced = _f0[_f0 > 0]
                    if len(_voiced):
                        _sum_log_f0 += float(_np.log(_voiced).sum())
                        _n_frames += len(_voiced)
                except Exception:
                    pass
            if _n_frames > 0:
                _mean_f0 = _math.exp(_sum_log_f0 / _n_frames)
                _f0_path = os.path.join(profile_dir, "speaker_f0.json")
                with open(_f0_path, "w") as _fj:
                    json.dump({"mean_f0": round(_mean_f0, 4), "method": "dio"}, _fj)
                await job.queue.put({
                    "type": "log",
                    "message": f"Profile mean F0: {_mean_f0:.1f} Hz → saved to speaker_f0.json",
                    "phase": "extract_f0",
                })
        except Exception as _e:
            await job.queue.put({
                "type": "log",
                "message": f"[warn] Could not compute mean F0: {_e}",
                "phase": "extract_f0",
            })

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
    # Pre-phase 3: Extract profile-level speaker embedding (if loss_mode requires it)
    # Single source of truth: profile_dir/profile_embedding.pt
    # ------------------------------------------------------------------
    profile_emb_path = (
        os.path.join(profile_dir, "profile_embedding.pt") if profile_dir else None
    )
    exp_emb_path = os.path.join(exp_dir, "profile_embedding.pt")  # copy for training

    _needs_spk_emb = loss_mode == "combined" or c_spk > 0
    if _needs_spk_emb:
        _emit_phase("extract_profile_emb", "Extracting profile speaker embedding")
        gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")

        # Also re-extract if the extraction script itself is newer than the embedding
        # (catches chunk duration changes like the 3s → 400ms migration)
        _extractor_path = os.path.join(
            rvc_pkg_dir, "infer/modules/train/extract_profile_embedding.py"
        )
        _script_newer = (
            profile_emb_path
            and os.path.isfile(profile_emb_path)
            and os.path.isfile(_extractor_path)
            and os.path.getmtime(_extractor_path) > os.path.getmtime(profile_emb_path)
        )

        needs_extraction = (
            not profile_emb_path
            or not os.path.isfile(profile_emb_path)
            or files_changed
            or different_profile
            or _script_newer
        )

        if needs_extraction:
            # Extract fresh embedding from training audio
            emb_device = _rvc_device
            emb_extract_args = [
                python,
                "infer/modules/train/extract_profile_embedding.py",
                gt_wavs_dir,
                profile_emb_path,  # Save directly to profile_dir
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
                    "message": f"Using existing speaker embedding from {profile_emb_path}",
                    "phase": "extract_profile_emb",
                }
            )

        # Copy to exp_dir for training subprocess
        if profile_emb_path and os.path.isfile(profile_emb_path):
            shutil.copy2(profile_emb_path, exp_emb_path)
    else:
        await job.queue.put(
            {
                "type": "log",
                "message": f"Speaker loss disabled (loss_mode={loss_mode}, c_spk={c_spk})",
                "phase": "setup",
            }
        )

    # ------------------------------------------------------------------
    # Pretrain weights — must be resolved before _write_config so that
    # spk_embed_dim can be read from the checkpoint (RefineGAN fix).
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Pre-phase 3 setup: config.json and filelist.txt
    # ------------------------------------------------------------------
    try:
        _write_config(
            project_root,
            exp_dir,
            batch_size=batch_size,
            c_spk=c_spk,
            loss_mode=loss_mode,
            vocoder=vocoder,
            pretrain_g=pretrain_g,
            adv_loss=adv_loss,
            kl_anneal=kl_anneal,
            kl_anneal_epochs=kl_anneal_epochs,
            optimizer=optimizer,
            use_reference_encoder=use_reference_encoder,
            reference_encoder_channels=reference_encoder_channels,
            ref_clip_seconds=ref_clip_seconds,
        )
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
    # Save every 20% of the run, capped at every 10 epochs.
    # Examples: 10→2, 20→4, 50→10, 100→10, 200→10
    effective_save_every = max(1, min(10, total_epoch // 5))

    # Cumulative epoch target: train.py resumes from prior_epochs (loaded
    # from G_latest.pth) and runs until it hits this number.
    cumulative_epochs = prior_epochs + total_epoch

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

    # Pass profile identity and DB path to train.py so it can write
    # best_epoch and epoch_losses directly without stdout/parsing glue.
    profile_id = job.profile_id
    db_path = os.path.join(project_root, "data", "rvc.db")

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
        "-pid",
        profile_id,
        "-db",
        db_path,
    ]

    # Persistent training log — scoped per profile, append mode so continuation
    # runs accumulate into a single file. Open before subprocess so we can pass
    # the fd directly (no asyncio reader needed — epoch events come from DB).
    profile_log_path = os.path.join(profile_dir, "train.log") if profile_dir else None
    _log_fh = None
    if profile_log_path:
        os.makedirs(profile_dir, exist_ok=True)
        try:
            import datetime as _dt
            _log_fh = open(profile_log_path, "a", buffering=1, encoding="utf-8")
            _log_fh.write(
                f"\n{'='*60}\n"
                f"Training session started: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Profile: {job.profile_id}  Prior epochs: {prior_epochs}  Target: +{total_epoch}\n"
                f"{'='*60}\n"
            )
        except OSError:
            _log_fh = None

    train_proc = await asyncio.create_subprocess_exec(
        *train_args,
        stdout=_log_fh if _log_fh else asyncio_subprocess.DEVNULL,
        stderr=asyncio_subprocess.STDOUT,  # merge stderr into same log file
        env=train_env,
        cwd=rvc_pkg_dir,
    )
    job._proc = train_proc

    # proc_done event — set after train_proc.wait() so _poll_rvc_epochs
    # can do one final sweep and then stop cleanly.
    _proc_done = asyncio.Event()

    # DB epoch poller — the source of truth for epoch_done events.
    # Replaces _tail_train_log + epoch parsing in _drain_stdout entirely.
    poll_task = asyncio.create_task(
        _poll_rvc_epochs(
            job,
            job.profile_id,
            prior_epochs,
            cumulative_epochs,
            overtrain_threshold,
            _proc_done,
            train_start_ts=time.monotonic(),
        )
    )

    await train_proc.wait()
    job._proc = None
    if _log_fh:
        _log_fh.close()

    # Signal poller that subprocess has exited; it will do one final sweep
    _proc_done.set()

    await asyncio.sleep(_RVC_POLL_INTERVAL + 0.5)

    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass

    # Determine actual completed epoch count from DB (authoritative source of truth).
    # Previously tracked via stdout; now we read it directly.
    try:
        async with get_db() as _db:
            _cur = await _db.execute(
                "SELECT MAX(epoch) FROM epoch_losses WHERE profile_id = ?",
                (job.profile_id,),
            )
            _row = await _cur.fetchone()
            _last_completed_epoch: int = int(_row[0]) if _row and _row[0] is not None else prior_epochs
    except Exception:
        _last_completed_epoch = prior_epochs
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
        # User-initiated cancel — job.status was set to "cancelled" by
        # cancel_job() before the subprocess was killed.  Proceed to artifact
        # phase so whatever checkpoints exist on disk are preserved.
        is_user_cancel = job.status == "cancelled"

        if not is_overtrain_stop and not is_user_cancel:
            # stderr already drained into job queue by _drain_stderr above
            err_msg = f"Train phase failed (exit code {train_proc.returncode}) — see [stderr] lines above"
            await _fail("train", err_msg)
            return

        if is_user_cancel:
            await job.queue.put(
                {
                    "type": "log",
                    "message": (
                        f"Training cancelled by user at epoch {_last_completed_epoch} — "
                        f"proceeding to save checkpoint and build index."
                    ),
                    "phase": "train",
                }
            )
        else:
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
        dest_best_g = os.path.join(dest_ckpt_dir, "G_best.pth")
        dest_infer_path = os.path.join(profile_dir, "model_infer.pth")
        dest_best_infer_path = os.path.join(profile_dir, "model_best.pth")
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
                # G_best.pth is written directly by train.py at the exact best
                # epoch. Copy it to the profile checkpoints dir if it exists.
                # If train.py never wrote it (cancel before any loss improvement,
                # or degenerate run) leave dest_best_g absent — no fallback to
                # G_latest, which would be misleading.
                g_best_src = os.path.join(exp_dir, "G_best.pth")
                if os.path.exists(g_best_src):
                    shutil.copy2(g_best_src, dest_best_g)
                    await job.queue.put(
                        {
                            "type": "log",
                            "message": f"Saved best checkpoint → {dest_best_g}",
                            "phase": "index",
                        }
                    )
                else:
                    await job.queue.put(
                        {
                            "type": "log",
                            "message": "G_best.pth not written by train.py — no best checkpoint saved",
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

            # Helper: generate a stripped inference model (~54 MB) from a full
            # checkpoint and write it to dest_path. Returns True on success.
            async def _make_infer_model(src_ckpt: str, dest_path: str, label: str) -> bool:
                try:
                    import sys as _sys
                    import json as _json
                    import torch as _torch_ssm

                    # Ensure backend/rvc is on sys.path so `infer.*` imports resolve.
                    # This function runs in the FastAPI process (not the train subprocess),
                    # so rvc_pkg_dir is not on sys.path by default.
                    if rvc_pkg_dir not in _sys.path:
                        _sys.path.insert(0, rvc_pkg_dir)

                    from infer.lib.train.process_ckpt import save_small_model as _ssm
                    from infer.lib.train.utils import HParams as _HParams

                    _cfg_path = os.path.join(exp_dir, "config.json")
                    with open(_cfg_path) as _f:
                        _cfg_d = _json.load(_f)
                    _hps = _HParams(**_cfg_d)
                    _hps.model_dir   = exp_dir
                    _hps.name        = exp_name
                    _hps.sample_rate = "32k"
                    _hps.if_f0       = 1
                    _hps.version     = "v2"
                    _hps.vocoder     = vocoder
                    _hps.embedder    = embedder
                    _hps.author      = ""
                    _hps.save_every_weights = "0"

                    # weights_only=False needed: checkpoint includes optimizer
                    # state dict (Adam moments) which torch refuses to load with
                    # weights_only=True. We only use ['model'] and ['iteration'].
                    _g_ckpt = _torch_ssm.load(src_ckpt, map_location="cpu", weights_only=False)
                    _state  = _g_ckpt.get("model", _g_ckpt)
                    _epoch  = _g_ckpt.get("iteration", _last_completed_epoch)

                    os.environ.setdefault("PROJECT_ROOT", project_root)
                    _result = _ssm(_state, "32k", 1, exp_name, _epoch, "v2", _hps)
                    if os.path.exists(weights_pth):
                        shutil.copy2(weights_pth, dest_path)
                        await job.queue.put({
                            "type": "log",
                            "message": f"Generated {label} → {dest_path} ({_result})",
                            "phase": "index",
                        })
                        return True
                    raise RuntimeError(f"save_small_model returned '{_result}' but {weights_pth} not found")
                except Exception as _e:
                    await job.queue.put({
                        "type": "log",
                        "message": f"Warning: could not generate {label}: {_e}",
                        "phase": "index",
                    })
                    return False

            # ── model_infer.pth (latest epoch, used by default) ──────────────
            if os.path.exists(weights_pth):
                # train.py wrote it at natural end of training
                shutil.copy2(weights_pth, dest_infer_path)
                await job.queue.put({
                    "type": "log",
                    "message": f"Saved inference model → {dest_infer_path}",
                    "phase": "index",
                })
            elif os.path.exists(g_latest):
                # Early stop — generate from G_latest.pth
                ok = await _make_infer_model(g_latest, dest_infer_path, "inference model from G_latest.pth")
                if not ok:
                    dest_infer_path = None
            else:
                await job.queue.put({
                    "type": "log",
                    "message": (
                        f"Warning: no checkpoint or inference model found — "
                        f"overtraining stop fired before first checkpoint save "
                        f"(save_every={effective_save_every}). "
                        f"Try a lower overtrain_threshold or more epochs."
                    ),
                    "phase": "index",
                })
                dest_infer_path = None

            # ── model_best.pth (best epoch, used when "Use best variant" is on) ─
            # G_best.pth is written by train.py at the exact best epoch.
            # Generate a stripped inference model from it only when it exists.
            # If G_best.pth was never written (cancel before any improvement, or
            # degenerate run) we leave model_best.pth absent — the UI checkbox
            # will be disabled, which correctly signals no best model exists.
            g_best_src = os.path.join(exp_dir, "G_best.pth")
            if os.path.exists(g_best_src):
                ok = await _make_infer_model(
                    g_best_src,
                    dest_best_infer_path,
                    "best inference model from G_best.pth",
                )
                if not ok:
                    await job.queue.put({
                        "type": "log",
                        "message": "Warning: could not generate model_best.pth — best variant unavailable",
                        "phase": "index",
                    })
            else:
                await job.queue.put({
                    "type": "log",
                    "message": "G_best.pth not found — model_best.pth not written (no best variant available)",
                    "phase": "index",
                })

            shutil.copy2(added_index_path, dest_index_path)
            await job.queue.put(
                {
                    "type": "log",
                    "message": f"Saved index → {dest_index_path}",
                    "phase": "index",
                }
            )
            # Copy reference encoder artifacts if present
            for _ref_fname in ("reference_encoder.pt", "style_vec_mean.pt"):
                _ref_src = os.path.join(exp_dir, _ref_fname)
                if os.path.isfile(_ref_src):
                    _ref_dst = os.path.join(profile_dir, _ref_fname)
                    shutil.copy2(_ref_src, _ref_dst)
                    await job.queue.put({
                        "type": "log",
                        "message": f"Saved {_ref_fname} → {_ref_dst}",
                        "phase": "index",
                    })
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
        dest_best_infer_path = dest_infer_path  # same file in legacy mode
        dest_index_path = added_index_path

    # ------------------------------------------------------------------
    # Success — update DB
    # ------------------------------------------------------------------
    job.phase = "done"
    job.progress_pct = 100
    # Preserve "cancelled" status so status endpoint reports it correctly
    # after artifact save completes. Only overwrite if it wasn't a cancel.
    if job.status != "cancelled":
        job.status = "done"

    # _last_completed_epoch is the authoritative epoch counter on all exit paths:
    #   - Normal end: tracks all epoch_done events up to total_epoch
    #   - Overtraining stop: tracks epochs before the OT SIGTERM fires
    #   - User cancel: train.py now handles SIGTERM by completing the current
    #     epoch, saving G_latest.pth at that exact epoch, then emitting
    #     "====> Epoch: N [cancel-flush]" before exit. _drain_stdout picks
    #     that up and updates _last_completed_epoch correctly.
    #     The old G_latest.pth-probing epoch downgrade is no longer needed.
    new_total_epochs = _last_completed_epoch

    async with get_db() as db:
        await db.execute(
            """UPDATE profiles
               SET status = 'trained',
                   model_path = ?,
                   checkpoint_path = ?,
                   index_path = ?,
                   total_epochs_trained = ?,
                   needs_retraining = 0,
                   best_model_path = ?
               WHERE id = ?""",
            (
                dest_infer_path,
                dest_model_path,
                dest_index_path,
                new_total_epochs,
                dest_best_infer_path if profile_dir and os.path.exists(dest_best_infer_path) else None,
                job.profile_id,
            ),
        )
        await db.commit()

    elapsed = time.monotonic() - start_ts
    final_msg = "Training cancelled — checkpoint saved." if job.status == "cancelled" else "Training complete"
    await job.queue.put(
        {"type": "done", "message": final_msg, "phase": "done"}
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
        loss_mode: str = "classic",
        adv_loss: str = "lsgan",
        kl_anneal: bool = False,
        kl_anneal_epochs: int = 40,
        optimizer: str = "adamw",
        use_reference_encoder: bool = False,
        reference_encoder_channels: int = 256,
        ref_clip_seconds: float = 6.0,
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
        job = TrainingJob(job_id=job_id, profile_id=profile_id,
                         total_epoch=total_epoch, batch_size=batch_size)
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
                loss_mode,
                adv_loss,
                kl_anneal,
                kl_anneal_epochs,
                optimizer,
                use_reference_encoder,
                reference_encoder_channels,
                ref_clip_seconds,
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
                # Send SIGTERM first — train.py's handler will complete the
                # current epoch, save G_latest.pth at that exact epoch, then
                # call os._exit(0).  We wait up to 120 s for a clean exit
                # (each epoch at bs=8 on MPS is ~60–90 s) before escalating
                # to SIGKILL.  psutil is used to also SIGTERM children so
                # the spawned worker process (mp.Process) receives the signal.
                import psutil

                try:
                    parent = psutil.Process(job._proc.pid)
                    # SIGTERM the whole process tree (parent + spawned child)
                    for child in parent.children(recursive=True):
                        try:
                            child.terminate()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    parent.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            except ImportError:
                # psutil not available — SIGTERM the direct subprocess
                try:
                    job._proc.terminate()
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
