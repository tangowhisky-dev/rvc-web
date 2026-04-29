"""BeatriceTrainingJob — async subprocess wrapper for beatrice_trainer.

Mirrors backend/app/training.py (RVC) for the Beatrice 2 pipeline.

Key differences vs RVC:
  - Subprocess: `python3 -m beatrice_trainer -d {data_dir} -o {out_dir} -c {config}`
  - Progress unit: step (not epoch)
  - Checkpoint: checkpoint_latest.pt.gz
  - CUDA required — checked before launch
  - BeatriceDBWriter writes to beatrice_steps table directly from async task
  - Preprocess: sync (fast) before subprocess launch
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import uuid
from asyncio import subprocess as asyncio_subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional

from backend.app.db import get_db, DB_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BeatriceTrainingJob
# ---------------------------------------------------------------------------


@dataclass
class BeatriceTrainingJob:
    """Holds all runtime state for a single Beatrice 2 training run."""

    job_id: str
    profile_id: str
    phase: str = "idle"
    progress_pct: int = 0
    status: str = "training"
    error: Optional[str] = None
    total_steps: int = 10000
    batch_size: int = 8
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _proc: Optional[asyncio_subprocess.Process] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------


async def _update_db_status(profile_id: str, status: str) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE profiles SET status = ? WHERE id = ?",
            (status, profile_id),
        )
        await db.commit()


_B2_POLL_INTERVAL = 2.0  # seconds between beatrice_steps polls


async def _poll_beatrice_steps(
    job: "BeatriceTrainingJob",
    profile_id: str,
    proc_done: asyncio.Event,
    out_dir: str,
) -> None:
    """Poll beatrice_steps every 2s; emit step_done + progress events from DB.

    Replaces stdout parsing entirely — all timing/ETA is derived from
    elapsed_sec written by the training loop at each _tb_interval.
    Stops one cycle after proc_done is set.
    """
    _seen_step: int = 0
    _last_log_step: int = 0  # emit a Step N/M log line every 100 steps
    _last_phase: str = ""    # track last phase.txt value to emit phase events once
    _session_start_step: int = -1  # first step seen this process session (for resume ETA)

    _PHASE_LABELS = {
        "extract_f0":      ("extract_f0",      "Computing mean F0s of target speakers..."),
        "extract_feature": ("extract_feature", "Building VQ codebooks..."),
        "train":           ("train",            "Training started."),
    }

    while True:
        await asyncio.sleep(_B2_POLL_INTERVAL)

        # --- Phase file: emit phase + log events when loop.py advances phase ---
        try:
            phase_file = os.path.join(out_dir, "phase.txt")
            if os.path.exists(phase_file):
                current_phase = open(phase_file).read().strip()
                if current_phase != _last_phase and current_phase in _PHASE_LABELS:
                    phase_key, label = _PHASE_LABELS[current_phase]
                    job.phase = phase_key
                    job.queue.put_nowait({"type": "phase", "phase": phase_key, "message": label})
                    job.queue.put_nowait({"type": "log",   "phase": phase_key, "message": label})
                    _last_phase = current_phase
        except Exception:
            pass

        try:
            async with get_db() as db:
                cursor = await db.execute(
                    """SELECT step, loss_mel, loss_loud, loss_ap,
                              loss_adv, loss_fm, loss_d, utmos, is_best, elapsed_sec
                       FROM beatrice_steps
                       WHERE profile_id = ? AND step >= ?
                       ORDER BY step ASC""",
                    # Look back 500 steps so a UTMOS write that arrives after
                    # the loss write for the same eval step is always picked up.
                    (profile_id, max(0, _seen_step - 500)),
                )
                rows = await cursor.fetchall()
        except Exception as _poll_exc:
            print(f"[_poll_beatrice_steps] DB query failed: {_poll_exc}", flush=True)
            rows = []

        # Merge rows by step — evaluation steps have two rows (losses + UTMOS).
        # Coalesce so one step_done event has all available fields.
        merged: dict[int, dict] = {}
        for row in rows:
            step = int(row[0])
            if step not in merged:
                merged[step] = {
                    "loss_mel": None, "loss_loud": None, "loss_ap": None,
                    "loss_adv": None, "loss_fm": None, "loss_d": None,
                    "utmos": None, "is_best": None, "elapsed_sec": None,
                }
            m = merged[step]
            m["loss_mel"]    = row[1] if row[1] is not None else m["loss_mel"]
            m["loss_loud"]   = row[2] if row[2] is not None else m["loss_loud"]
            m["loss_ap"]     = row[3] if row[3] is not None else m["loss_ap"]
            m["loss_adv"]    = row[4] if row[4] is not None else m["loss_adv"]
            m["loss_fm"]     = row[5] if row[5] is not None else m["loss_fm"]
            m["loss_d"]      = row[6] if row[6] is not None else m["loss_d"]
            m["utmos"]       = row[7] if row[7] is not None else m["utmos"]
            m["is_best"]     = row[8] if row[8] is not None else m["is_best"]
            m["elapsed_sec"] = row[9] if row[9] is not None else m["elapsed_sec"]

        for step, losses in sorted(merged.items()):
            elapsed_sec = losses.pop("elapsed_sec")

            # First step seen this session — record it so ETA is relative to
            # steps done this run (not total steps, which breaks on resume).
            if _session_start_step < 0:
                _session_start_step = step

            # Progress from DB step — no tqdm needed
            pct = int(100 * step / job.total_steps) if job.total_steps else 0
            job.progress_pct = pct
            job.queue.put_nowait({
                "type": "step_done",
                "step": step,
                "total_steps": job.total_steps,
                "progress_pct": pct,
                "losses": losses,
            })

            # Emit a human-readable Step N/M  Xs/it  ETA log every 100 steps
            if step - _last_log_step >= 100 and elapsed_sec:
                steps_this_session = step - _session_start_step
                secs_per_it = (elapsed_sec / steps_this_session
                               if steps_this_session > 0 else 0)
                remaining   = (job.total_steps - step) * secs_per_it
                parts = [f"Step {step}/{job.total_steps}"]
                if secs_per_it >= 1:
                    parts.append(f"{secs_per_it:.1f}s/it")
                else:
                    parts.append(f"{secs_per_it*1000:.0f}ms/it")
                # Format elapsed
                e_h, e_rem = divmod(int(elapsed_sec), 3600)
                e_m, e_s   = divmod(e_rem, 60)
                if e_h:
                    parts.append(f"elapsed {e_h}h{e_m:02d}m")
                else:
                    parts.append(f"elapsed {e_m}m{e_s:02d}s")
                # Format ETA
                r_h, r_rem = divmod(int(remaining), 3600)
                r_m, r_s   = divmod(r_rem, 60)
                if r_h:
                    parts.append(f"ETA {r_h}h{r_m:02d}m")
                else:
                    parts.append(f"ETA {r_m}m{r_s:02d}s")
                job.queue.put_nowait({"type": "log", "message": "  ".join(parts), "phase": "train"})
                _last_log_step = step

            _seen_step = step

        if proc_done.is_set():
            break


# ---------------------------------------------------------------------------
# Config writer
# ---------------------------------------------------------------------------


def _write_beatrice_config(
    out_dir: str,
    project_root: str,
    n_steps: int,
    batch_size: int,
    record_metrics: bool = False,
    use_reference_encoder: bool = False,
    reference_encoder_channels: int = 256,
    ref_clip_seconds: float = 6.0,
) -> str:
    """Write config.json to out_dir and return its path."""
    assets_dir = os.path.join(project_root, "assets", "beatrice2")

    # Default config merged from beatrice_trainer/assets/default_config.json
    default_cfg_path = os.path.join(
        project_root,
        "backend",
        "beatrice2",
        "beatrice_trainer",
        "assets",
        "default_config.json",
    )
    try:
        with open(default_cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cfg = {}

    # When resuming, n_steps must be initial_iteration + requested_steps so that
    # range(initial_iteration, n_steps) produces the right number of iterations.
    # Read completed steps from the existing config.json if present.
    existing_config_path = os.path.join(out_dir, "config.json")
    completed_steps = 0
    checkpoint_path = os.path.join(out_dir, "checkpoint_latest.pt.gz")
    if os.path.exists(checkpoint_path):
        try:
            import gzip as _gzip
            import torch as _torch
            with _gzip.open(checkpoint_path, "rb") as _f:
                _ckpt = _torch.load(_f, map_location="cpu", weights_only=True)
            completed_steps = int(_ckpt.get("iteration", 0))
        except Exception:
            # Fallback: read previous n_steps from config.json as a rough estimate
            if os.path.exists(existing_config_path):
                try:
                    with open(existing_config_path, encoding="utf-8") as _f:
                        completed_steps = int(json.load(_f).get("n_steps", 0))
                except Exception:
                    completed_steps = 0

    total_steps = completed_steps + n_steps if completed_steps > 0 else n_steps

    cfg.update(
        {
            "phone_extractor_file": os.path.join(
                assets_dir, "pretrained", "122_checkpoint_03000000.pt"
            ),
            "pitch_estimator_file": os.path.join(
                assets_dir, "pretrained", "104_3_checkpoint_00300000.pt"
            ),
            "pretrained_file": os.path.join(
                assets_dir, "pretrained", "151_checkpoint_libritts_r_200_02750000.pt.gz"
            ),
            "in_ir_wav_dir": os.path.join(assets_dir, "ir"),
            "in_noise_wav_dir": os.path.join(assets_dir, "noise"),
            "in_test_wav_dir": os.path.join(assets_dir, "test"),
            "n_steps": total_steps,
            "batch_size": batch_size,
            "use_amp": True,
            # warmup = 50% of total steps, capped at 5000.
            # Must be based on total_steps (not per-run increment) because the
            # scheduler always counts from step 0 and is fast-forwarded on resume.
            # Cap at 5000 matches upstream pretrained model assumption (n_steps≥10k).
            "warmup_steps": min(math.ceil(total_steps * 0.5), 5000),
            "save_interval": max(500, total_steps // 20),
            "evaluation_interval": max(500, total_steps // 20),
            "record_metrics": record_metrics,
            # Reference encoder
            "use_reference_encoder": use_reference_encoder,
            "reference_encoder_channels": reference_encoder_channels,
            "ref_clip_seconds": ref_clip_seconds,
        }
    )

    config_path = os.path.join(out_dir, "config.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return config_path, total_steps


# ---------------------------------------------------------------------------
# Preprocessing helper (synchronous — runs in executor)
# ---------------------------------------------------------------------------


def _build_test_holdout(data_dir: str, test_dir: str, holdout_fraction: float = 0.1, max_files: int = 20) -> int:
    """Copy a stratified holdout of preprocessed chunks to test_dir for UTMOS eval.

    Picks the last ``holdout_fraction`` of chunks from each speaker subdir
    (deterministic — same files every time) up to ``max_files`` total.
    Files are *copied* not moved so training data is unaffected.
    Returns the number of files in test_dir after the call.
    """
    import shutil as _shutil
    from pathlib import Path as _Path

    data_path = _Path(data_dir)
    test_path = _Path(test_dir)
    test_path.mkdir(parents=True, exist_ok=True)

    # Collect all chunk files across speaker subdirs, sorted for determinism
    EXTS = {".wav", ".flac"}
    all_chunks: list[_Path] = []
    for speaker_dir in sorted(data_path.iterdir()):
        if not speaker_dir.is_dir() or speaker_dir.name == "_originals":
            continue
        chunks = sorted(
            f for f in speaker_dir.iterdir()
            if f.suffix.lower() in EXTS and "_originals" not in f.parts
        )
        if not chunks:
            continue
        # Take last holdout_fraction of each speaker's chunks — these are
        # unseen by training only if the dataset is large enough; for small
        # datasets (<10 chunks) we still use the last 1 file so there's
        # always something to eval on.
        n_hold = max(1, int(len(chunks) * holdout_fraction))
        all_chunks.extend(chunks[-n_hold:])

    # Cap total and copy
    selected = all_chunks[:max_files]
    for src in selected:
        dst = test_path / src.name
        if not dst.exists():
            _shutil.copy2(src, dst)

    return len(list(test_path.iterdir()))


def _preprocess_audio(
    audio_dir: str,
    data_dir: str,
    speaker_name: str,
    sample_rate: int = 24000,
    chunk_duration: float = 4.0,
) -> int:
    """Split audio files into ~4s chunks at 24kHz, one subdir per speaker.

    Returns the number of chunks written.
    """
    import warnings
    import torchaudio
    import torch

    warnings.filterwarnings("ignore", category=UserWarning)

    import logging as _logging
    _log = _logging.getLogger(__name__)

    speaker_out = os.path.join(data_dir, speaker_name)
    os.makedirs(speaker_out, exist_ok=True)

    audio_suffixes = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3", ".m4a"}
    chunk_samples = int(chunk_duration * sample_rate)
    total_chunks = 0
    chunk_idx = 0
    load_errors: list[str] = []

    files_found = [
        fname for fname in sorted(os.listdir(audio_dir))
        if os.path.splitext(fname)[1].lower() in audio_suffixes
    ]

    if not files_found:
        _log.warning(
            "_preprocess_audio: no audio files found in %s (suffixes checked: %s)",
            audio_dir, sorted(audio_suffixes),
        )
        return 0

    for fname in files_found:
        src = os.path.join(audio_dir, fname)
        try:
            wav, sr = torchaudio.load(src)
        except Exception as exc:
            msg = f"failed to load {fname}: {type(exc).__name__}: {exc}"
            _log.error("_preprocess_audio: %s", msg)
            load_errors.append(msg)
            continue

        # Resample to 24 kHz
        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)

        # Mix to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        n_samples = wav.shape[-1]
        n_chunks = n_samples // chunk_samples
        if n_chunks == 0:
            n_chunks = 1  # keep even very short files as-is

        for i in range(n_chunks):
            start = i * chunk_samples
            chunk = wav[:, start : start + chunk_samples]
            out_path = os.path.join(speaker_out, f"chunk_{chunk_idx:06d}.wav")
            torchaudio.save(out_path, chunk, sample_rate)
            chunk_idx += 1
            total_chunks += 1

    if load_errors and total_chunks == 0:
        raise RuntimeError(
            f"All {len(load_errors)} audio file(s) failed to load. "
            f"First error: {load_errors[0]}"
        )

    return total_chunks


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main pipeline coroutine
# ---------------------------------------------------------------------------


async def _run_beatrice_pipeline(
    job: BeatriceTrainingJob,
    audio_dir: str,
    project_root: str,
    out_dir: str,
    data_dir: str,
    speaker_name: str,
    n_steps: int,
    batch_size: int,
    profile_id: str,
    use_reference_encoder: bool = False,
    reference_encoder_channels: int = 256,
    ref_clip_seconds: float = 6.0,
) -> None:
    """Full Beatrice 2 training pipeline coroutine.

    Phases:
      1. preprocess — split audio into 4s chunks (skipped on resume if data_dir exists)
      2. train      — launch beatrice_trainer subprocess
    """
    python = sys.executable

    # ------------------------------------------------------------------
    # Phase 1: Preprocessing (sync → executor)
    # Skip on resume if the data_dir already contains chunks — re-chunking
    # audio on every restart is the main source of "starting from step 1"
    # slowness. Re-run if audio_dir has changed or data_dir is empty.
    # ------------------------------------------------------------------
    _checkpoint_exists = os.path.exists(os.path.join(out_dir, "checkpoint_latest.pt.gz"))
    _data_has_chunks = os.path.isdir(os.path.join(data_dir, speaker_name)) and \
        any(True for _ in __import__("os").scandir(os.path.join(data_dir, speaker_name)))
    _skip_preprocess = _checkpoint_exists and _data_has_chunks

    job.phase = "preprocess"
    if _skip_preprocess:
        job.queue.put_nowait(
            {"type": "phase", "message": "Resuming — skipping audio preprocessing (chunks already on disk).", "phase": "preprocess"}
        )
        job.queue.put_nowait(
            {"type": "log", "message": "Preprocessing skipped (resume with existing chunks).", "phase": "preprocess"}
        )
        n_chunks = sum(1 for _ in __import__("os").scandir(os.path.join(data_dir, speaker_name)))
    else:
        job.queue.put_nowait(
            {"type": "phase", "message": "Preprocessing audio for Beatrice 2...", "phase": "preprocess"}
        )

    # ------------------------------------------------------------------
    try:
        loop = asyncio.get_event_loop()
        if not _skip_preprocess:
            n_chunks = await loop.run_in_executor(
                None,
                _preprocess_audio,
                audio_dir,
                data_dir,
                speaker_name,
            )
            job.queue.put_nowait(
                {
                    "type": "log",
                    "message": f"Preprocessing complete: {n_chunks} chunks written.",
                    "phase": "preprocess",
                }
            )
        if n_chunks == 0:
            job.status = "failed"
            job.error = (
                f"Preprocessing produced 0 chunks from {audio_dir}. "
                "Check that the audio/ directory contains valid WAV files and that "
                "libsndfile/torchaudio can load them. See train.log for per-file errors."
            )
            job.queue.put_nowait({"type": "error", "message": job.error, "phase": "preprocess"})
            await _update_db_status(profile_id, "failed")
            return
    except Exception as exc:
        job.status = "failed"
        job.error = f"Preprocessing failed: {exc}"
        job.queue.put_nowait(
            {"type": "error", "message": job.error, "phase": "preprocess"}
        )
        await _update_db_status(profile_id, "failed")
        return

    # ------------------------------------------------------------------
    # Phase 1b: Build in-domain UTMOS holdout from preprocessed chunks
    # ------------------------------------------------------------------
    test_dir = os.path.join(out_dir, "utmos_test")
    try:
        n_test = await loop.run_in_executor(
            None, _build_test_holdout, data_dir, test_dir
        )
        job.queue.put_nowait(
            {
                "type": "log",
                "message": f"UTMOS test holdout: {n_test} files in {test_dir}",
                "phase": "preprocess",
            }
        )
    except Exception as exc:
        # Non-fatal — fallback to assets/beatrice2/test/ happens inside loop.py
        test_dir = None
        job.queue.put_nowait(
            {"type": "log", "message": f"UTMOS holdout failed (will use default test set): {exc}", "phase": "preprocess"}
        )

    # ------------------------------------------------------------------
    # Phase 2: Write config
    # ------------------------------------------------------------------
    config_path, total_steps = _write_beatrice_config(
        out_dir=out_dir,
        project_root=project_root,
        n_steps=n_steps,
        batch_size=batch_size,
        use_reference_encoder=use_reference_encoder,
        reference_encoder_channels=reference_encoder_channels,
        ref_clip_seconds=ref_clip_seconds,
    )
    # Update job with the real total (completed + requested) so progress/ETA
    # are correct when resuming from a prior checkpoint.
    job.total_steps = total_steps

    # ------------------------------------------------------------------
    # Phase 3: Training subprocess
    # ------------------------------------------------------------------
    # Do NOT pre-announce "train" phase here — the subprocess writes
    # phase.txt (extract_f0 → extract_feature → train) and the poller
    # picks those up in order.  Emitting "train" now causes the UI to
    # jump forward then snap back to extract_f0.
    env = dict(os.environ)
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        python,
        "-m",
        "backend.beatrice2.beatrice_trainer",
        "-d",
        data_dir,
        "-o",
        out_dir,
        "-c",
        config_path,
        "--db",
        DB_PATH,
        "--profile-id",
        profile_id,
    ]
    if test_dir and os.path.isdir(test_dir):
        cmd += ["--test-dir", test_dir]

    # Check for existing checkpoint — resume automatically
    checkpoint = os.path.join(out_dir, "checkpoint_latest.pt.gz")
    if os.path.exists(checkpoint):
        cmd += ["--resume"]
        job.queue.put_nowait(
            {"type": "log", "message": "Resuming from existing checkpoint.", "phase": "train"}
        )

    try:
        # Redirect subprocess stdout+stderr to a log file for crash diagnosis.
        # All progress/ETA/step data comes from DB via _poll_beatrice_steps —
        # no stdout parsing needed.
        log_path = os.path.join(out_dir, "train.log")
        _log_file = open(log_path, "a", buffering=1)
        proc = await asyncio_subprocess.create_subprocess_exec(
            *cmd,
            stdout=_log_file.fileno(),
            stderr=_log_file.fileno(),
            cwd=project_root,
            env=env,
        )
        job._proc = proc
    except Exception as exc:
        job.status = "failed"
        job.error = f"Failed to launch trainer: {exc}"
        job.queue.put_nowait(
            {"type": "error", "message": job.error, "phase": "train"}
        )
        await _update_db_status(profile_id, "failed")
        return

    # DB poller — emits step_done + progress events; stops after proc exits.
    _proc_done = asyncio.Event()
    poll_task = asyncio.create_task(
        _poll_beatrice_steps(job, profile_id, _proc_done, out_dir)
    )

    await proc.wait()
    _log_file.close()

    # Signal poller; wait for its final sweep before proceeding
    _proc_done.set()
    await asyncio.sleep(_B2_POLL_INTERVAL + 0.5)
    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass

    rc = proc.returncode
    was_cancelled = job.status == "cancelled"

    if rc == 0 or was_cancelled:
        job.phase = "done"
        job.progress_pct = 100

        # Locate checkpoint — prefer checkpoint_best.pt.gz (best UTMOS) over latest
        checkpoint_best = os.path.join(out_dir, "checkpoint_best.pt.gz")
        model_path = (
            checkpoint_best if os.path.exists(checkpoint_best)
            else checkpoint if os.path.exists(checkpoint)
            else None
        )

        if was_cancelled:
            # Cancelled: mark failed (or partial), don't overwrite status already set
            # by cancel_job(). DB status was already set to 'failed' by cancel_job.
            pass
        else:
            # Successful completion — mark trained
            async with get_db() as db:
                await db.execute(
                    """UPDATE profiles
                       SET status              = 'trained',
                           model_path          = ?,
                           total_epochs_trained = ?
                       WHERE id = ?""",
                    (model_path, n_steps, profile_id),
                )
                await db.commit()
            job.status = "done"
            _used_best = os.path.exists(checkpoint_best)
            job.queue.put_nowait(
                {
                    "type": "done",
                    "message": f"Beatrice 2 training complete. Model: {'best (highest UTMOS)' if _used_best else 'latest checkpoint'}.",
                    "phase": "done",
                    "output_path": model_path or "",
                }
            )
    else:
        job.status = "failed"
        job.error = f"Trainer exited with code {rc}"
        job.queue.put_nowait(
            {"type": "error", "message": job.error, "phase": "train"}
        )
        await _update_db_status(profile_id, "failed")


# ---------------------------------------------------------------------------
# BeatriceTrainingManager
# ---------------------------------------------------------------------------


class BeatriceTrainingManager:
    """Manages at most one Beatrice 2 training job at a time."""

    def __init__(self) -> None:
        self._jobs: Dict[str, BeatriceTrainingJob] = {}

    def start_job(
        self,
        profile_id: str,
        audio_dir: str,
        project_root: str,
        n_steps: int = 10000,
        batch_size: int = 8,
        profile_name: str = "speaker",
        use_reference_encoder: bool = False,
        reference_encoder_channels: int = 256,
        ref_clip_seconds: float = 6.0,
    ) -> BeatriceTrainingJob:
        """Create and launch a Beatrice 2 training job.

        Raises:
            RuntimeError: if CUDA is not available.
            ValueError: if a job is already running.
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Beatrice 2 training requires a CUDA GPU. "
                "No CUDA device was detected on this machine."
            )

        if any(j.status == "training" for j in self._jobs.values()):
            raise ValueError("job already running")

        job_id = uuid.uuid4().hex
        job = BeatriceTrainingJob(
            job_id=job_id,
            profile_id=profile_id,
            total_steps=n_steps,
            batch_size=batch_size,
        )
        self._jobs[profile_id] = job

        # Derived paths
        pdir = os.path.join(project_root, "data", "profiles", profile_id)
        out_dir = os.path.join(pdir, "beatrice2_out")
        data_dir = os.path.join(pdir, "beatrice2_data")

        asyncio.create_task(
            _run_beatrice_pipeline(
                job=job,
                audio_dir=audio_dir,
                project_root=project_root,
                out_dir=out_dir,
                data_dir=data_dir,
                speaker_name=profile_name,
                n_steps=n_steps,
                batch_size=batch_size,
                profile_id=profile_id,
                use_reference_encoder=use_reference_encoder,
                reference_encoder_channels=reference_encoder_channels,
                ref_clip_seconds=ref_clip_seconds,
            )
        )
        return job

    def cancel_job(self, profile_id: str) -> None:
        job = self._jobs.get(profile_id)
        if job is None:
            return
        if job._proc is not None:
            try:
                import psutil

                try:
                    parent = psutil.Process(job._proc.pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.terminate()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    parent.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            except ImportError:
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
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_update_db_status(profile_id, "failed"))
        except RuntimeError:
            pass

    def get_job(self, profile_id: str) -> Optional[BeatriceTrainingJob]:
        return self._jobs.get(profile_id)

    def active_job(self) -> Optional[BeatriceTrainingJob]:
        for job in self._jobs.values():
            if job.status == "training":
                return job
        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

beatrice_manager = BeatriceTrainingManager()
