"""Direct synchronous SQLite writes from the training subprocess.

train.py runs in a subprocess (mp.Process) that has no event loop and no
access to the FastAPI app's async DB layer.  This module provides plain
sqlite3 calls — no asyncio, no aiosqlite, no HTTP — so the subprocess can
write training state (best_epoch, epoch_losses) directly to rvc.db.

Usage inside train.py:
    from infer.lib.train.db_writer import TrainingDBWriter
    db = TrainingDBWriter(db_path, profile_id)   # once at startup
    db.update_best_epoch(epoch, avg_loss_g)       # every best-epoch save
    db.insert_epoch_loss(epoch, losses_dict)       # every epoch
"""

import sqlite3
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TrainingDBWriter:
    """Synchronous sqlite3 writer for training-loop DB updates."""

    def __init__(self, db_path: str, profile_id: str) -> None:
        self._db_path = db_path
        self._profile_id = profile_id
        self._enabled = bool(db_path and profile_id)
        if not self._enabled:
            logger.debug("TrainingDBWriter: disabled (no db_path or profile_id)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_best_epoch(self, epoch: int, avg_loss_g: float) -> None:
        """Write best_epoch and best_avg_gen_loss to the profiles table."""
        if not self._enabled:
            return
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.execute(
                    """UPDATE profiles
                       SET best_epoch = ?, best_avg_gen_loss = ?
                       WHERE id = ?""",
                    (epoch, round(avg_loss_g, 4), self._profile_id),
                )
                conn.commit()
        except Exception as exc:
            logger.warning(f"[DBWriter] update_best_epoch failed: {exc}")

    def insert_epoch_loss(self, epoch: int, losses: dict) -> None:
        """Upsert a row into epoch_losses for this profile+epoch."""
        if not self._enabled:
            return
        import datetime, uuid
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.execute(
                    """INSERT INTO epoch_losses
                           (id, profile_id, epoch, loss_mel, loss_gen, loss_disc, loss_fm, loss_kl, loss_spk, trained_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(profile_id, epoch) DO UPDATE SET
                           loss_mel   = excluded.loss_mel,
                           loss_gen   = excluded.loss_gen,
                           loss_disc  = excluded.loss_disc,
                           loss_fm    = excluded.loss_fm,
                           loss_kl    = excluded.loss_kl,
                           loss_spk   = excluded.loss_spk,
                           trained_at = excluded.trained_at""",
                    (
                        str(uuid.uuid4()),
                        self._profile_id,
                        epoch,
                        losses.get("loss_mel"),
                        losses.get("loss_gen"),
                        losses.get("loss_disc"),
                        losses.get("loss_fm"),
                        losses.get("loss_kl"),
                        losses.get("loss_spk"),
                        datetime.datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.warning(f"[DBWriter] insert_epoch_loss failed: {exc}")

    def update_total_epochs(self, epoch: int) -> None:
        """Update total_epochs_trained at natural training completion."""
        if not self._enabled:
            return
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.execute(
                    "UPDATE profiles SET total_epochs_trained = ? WHERE id = ?",
                    (epoch, self._profile_id),
                )
                conn.commit()
        except Exception as exc:
            logger.warning(f"[DBWriter] update_total_epochs failed: {exc}")
