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
import os
import re
import sys
import uuid
from asyncio import subprocess as asyncio_subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional

from backend.app.db import get_db, DB_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TQDM_RE = re.compile(r"Training:\s+(\d+)%.*?(\d+)/(\d+)")


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
) -> None:
    """Poll beatrice_steps every 2s; emit step_done events with DB losses.

    Mirrors _poll_rvc_epochs for the Beatrice 2 pipeline.
    Stops one cycle after proc_done is set.
    """
    _seen_step: int = 0

    while True:
        await asyncio.sleep(_B2_POLL_INTERVAL)

        try:
            async with get_db() as db:
                cursor = await db.execute(
                    """SELECT step, loss_mel, loss_loud, loss_ap,
                              loss_adv, loss_fm, loss_d
                       FROM beatrice_steps
                       WHERE profile_id = ? AND step > ?
                       ORDER BY step ASC""",
                    (profile_id, _seen_step),
                )
                rows = await cursor.fetchall()
        except Exception:
            rows = []

        for row in rows:
            step = int(row[0])
            losses = {
                "loss_mel":  row[1],
                "loss_loud": row[2],
                "loss_ap":   row[3],
                "loss_adv":  row[4],
                "loss_fm":   row[5],
                "loss_d":    row[6],
            }
            pct = int(100 * step / job.total_steps) if job.total_steps else 0
            job.queue.put_nowait({
                "type": "step_done",
                "step": step,
                "total_steps": job.total_steps,
                "progress_pct": pct,
                "losses": losses,
            })
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
            "n_steps": n_steps,
            "batch_size": batch_size,
            "use_amp": True,
            "num_workers": min(4, os.cpu_count() or 2),
            "save_interval": max(500, n_steps // 20),
            "evaluation_interval": max(500, n_steps // 20),
            "record_metrics": record_metrics,
        }
    )

    config_path = os.path.join(out_dir, "config.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return config_path


# ---------------------------------------------------------------------------
# Preprocessing helper (synchronous — runs in executor)
# ---------------------------------------------------------------------------


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

    speaker_out = os.path.join(data_dir, speaker_name)
    os.makedirs(speaker_out, exist_ok=True)

    audio_suffixes = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3", ".m4a"}
    chunk_samples = int(chunk_duration * sample_rate)
    total_chunks = 0
    chunk_idx = 0

    for fname in sorted(os.listdir(audio_dir)):
        if os.path.splitext(fname)[1].lower() not in audio_suffixes:
            continue
        src = os.path.join(audio_dir, fname)
        try:
            wav, sr = torchaudio.load(src)
        except Exception:
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
) -> None:
    """Full Beatrice 2 training pipeline coroutine.

    Phases:
      1. preprocess — split audio into 4s chunks
      2. train      — launch beatrice_trainer subprocess
    """
    python = sys.executable
    job.phase = "preprocess"
    job.queue.put_nowait(
        {"type": "phase", "message": "Preprocessing audio for Beatrice 2...", "phase": "preprocess"}
    )

    # ------------------------------------------------------------------
    # Phase 1: Preprocessing (sync → executor)
    # ------------------------------------------------------------------
    try:
        loop = asyncio.get_event_loop()
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
    except Exception as exc:
        job.status = "failed"
        job.error = f"Preprocessing failed: {exc}"
        job.queue.put_nowait(
            {"type": "error", "message": job.error, "phase": "preprocess"}
        )
        await _update_db_status(profile_id, "failed")
        return

    # ------------------------------------------------------------------
    # Phase 2: Write config
    # ------------------------------------------------------------------
    config_path = _write_beatrice_config(
        out_dir=out_dir,
        project_root=project_root,
        n_steps=n_steps,
        batch_size=batch_size,
    )

    job.phase = "train"
    job.queue.put_nowait(
        {"type": "phase", "message": "Starting Beatrice 2 trainer...", "phase": "train"}
    )

    # ------------------------------------------------------------------
    # Phase 3: Training subprocess
    # ------------------------------------------------------------------
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

    # Check for existing checkpoint — resume automatically
    checkpoint = os.path.join(out_dir, "checkpoint_latest.pt.gz")
    if os.path.exists(checkpoint):
        cmd += ["--resume"]
        job.queue.put_nowait(
            {"type": "log", "message": "Resuming from existing checkpoint.", "phase": "train"}
        )

    try:
        proc = await asyncio_subprocess.create_subprocess_exec(
            *cmd,
            stdout=asyncio_subprocess.PIPE,
            stderr=asyncio_subprocess.STDOUT,
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

    # DB poller — emits step_done events as BeatriceDBWriter inserts rows.
    # Runs concurrently with stdout drain; stops after proc exits.
    _proc_done = asyncio.Event()
    poll_task = asyncio.create_task(
        _poll_beatrice_steps(job, profile_id, _proc_done)
    )

    # Drain stdout: forward log lines and parse tqdm for progress % display.
    # step_done events are now entirely owned by _poll_beatrice_steps above.
    assert proc.stdout is not None
    async for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue

        # Parse tqdm progress: "Training:  20%|...| 2000/10000 [elapsed<eta]"
        m = _TQDM_RE.search(line)
        if m:
            pct = int(m.group(1))
            step = int(m.group(2))
            total = int(m.group(3))
            job.progress_pct = pct
            job.total_steps = total
            job.queue.put_nowait(
                {
                    "type": "progress",
                    "step": step,
                    "total_steps": total,
                    "progress_pct": pct,
                    "phase": "train",
                }
            )
        else:
            job.queue.put_nowait(
                {"type": "log", "message": line, "phase": "train"}
            )

    await proc.wait()

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

        # Locate checkpoint — use checkpoint_latest.pt.gz
        model_path = checkpoint if os.path.exists(checkpoint) else None

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
            job.queue.put_nowait(
                {
                    "type": "done",
                    "message": "Beatrice 2 training complete.",
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
