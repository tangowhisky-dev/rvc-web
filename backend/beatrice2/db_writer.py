"""Direct synchronous SQLite writes from the Beatrice 2 training subprocess.

Mirrors the pattern of backend/rvc/infer/lib/train/db_writer.py — plain
sqlite3, no asyncio, no aiosqlite, so the subprocess can write step losses
directly to rvc.db without any IPC.

Usage inside loop.py:
    from backend.beatrice2.db_writer import BeatriceDBWriter
    db = BeatriceDBWriter(db_path, profile_id)
    db.insert_step_loss(iteration, losses_dict)   # every _tb_interval steps
"""

import datetime
import logging
import sqlite3

logger = logging.getLogger(__name__)


class BeatriceDBWriter:
    """Synchronous sqlite3 writer for Beatrice 2 training-loop DB updates."""

    def __init__(self, db_path: str, profile_id: str) -> None:
        self._db_path = db_path
        self._profile_id = profile_id
        self._enabled = bool(db_path and profile_id)
        if not self._enabled:
            logger.debug("BeatriceDBWriter: disabled (no db_path or profile_id)")

    def insert_step_loss(self, step: int, losses: dict) -> None:
        """Upsert one row into beatrice_steps for this profile+step.

        Keys expected in losses (all optional — None stored as NULL):
          loss_mel   — mel spectrogram reconstruction (primary quality signal)
          loss_loud  — loudness matching
          loss_ap    — aperiodicity
          loss_adv   — adversarial (generator)
          loss_fm    — feature matching
          loss_d     — discriminator total
          utmos      — UTMOS MOS score (1–5); only present at evaluation steps
          is_best    — 1 if this step has the best UTMOS seen so far
        """
        if not self._enabled:
            return
        row_id = f"{self._profile_id}_{step}"
        trained_at = datetime.datetime.utcnow().isoformat()
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.execute(
                    """INSERT INTO beatrice_steps
                           (id, profile_id, step,
                            loss_mel, loss_loud, loss_ap,
                            loss_adv, loss_fm, loss_d,
                            utmos, is_best,
                            trained_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(profile_id, step) DO UPDATE SET
                           loss_mel   = COALESCE(excluded.loss_mel,  loss_mel),
                           loss_loud  = COALESCE(excluded.loss_loud, loss_loud),
                           loss_ap    = COALESCE(excluded.loss_ap,   loss_ap),
                           loss_adv   = COALESCE(excluded.loss_adv,  loss_adv),
                           loss_fm    = COALESCE(excluded.loss_fm,   loss_fm),
                           loss_d     = COALESCE(excluded.loss_d,    loss_d),
                           utmos      = COALESCE(excluded.utmos,     utmos),
                           is_best    = COALESCE(excluded.is_best,   is_best),
                           trained_at = excluded.trained_at""",
                    (
                        row_id,
                        self._profile_id,
                        step,
                        losses.get("loss_mel"),
                        losses.get("loss_loud"),
                        losses.get("loss_ap"),
                        losses.get("loss_adv"),
                        losses.get("loss_fm"),
                        losses.get("loss_d"),
                        losses.get("utmos"),
                        losses.get("is_best", 0),
                        trained_at,
                    ),
                )
                conn.commit()
        except Exception as exc:
            print(f"[BeatriceDBWriter] insert_step_loss step={step} FAILED: {exc}", flush=True)
            logger.warning(f"[BeatriceDBWriter] insert_step_loss failed: {exc}")

    def update_best_utmos(self, step: int, utmos: float) -> None:
        """Mark `step` as the new best-UTMOS checkpoint and clear previous best flag."""
        if not self._enabled:
            return
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                # Clear previous best
                conn.execute(
                    "UPDATE beatrice_steps SET is_best = 0 WHERE profile_id = ? AND is_best = 1",
                    (self._profile_id,),
                )
                # Set new best (upsert in case the step row doesn't exist yet)
                conn.execute(
                    """UPDATE beatrice_steps SET is_best = 1, utmos = ?
                       WHERE profile_id = ? AND step = ?""",
                    (utmos, self._profile_id, step),
                )
                conn.commit()
        except Exception as exc:
            logger.warning(f"[BeatriceDBWriter] update_best_utmos failed: {exc}")
