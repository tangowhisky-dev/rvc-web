"""SQLite database initialization and connection helpers.

Provides:
  - DB_PATH: canonical path for the SQLite database file
  - init_db(): idempotent schema creation + column migrations
  - get_db(): async context manager yielding an aiosqlite.Connection
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aiosqlite

DB_PATH = "backend/data/rvc.db"

_CREATE_PROFILES = """
CREATE TABLE IF NOT EXISTS profiles (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'untrained',
    sample_path       TEXT,
    model_path        TEXT,
    index_path        TEXT,
    created_at        TEXT NOT NULL,
    batch_size        INTEGER NOT NULL DEFAULT 8,
    audio_duration    REAL,
    preprocessed_path TEXT,
    profile_dir       TEXT
)
"""

# audio_files: one-to-many with profiles.
# file_path is the absolute path to the WAV on disk (always in profile_dir/audio/).
# is_cleaned = 1 after "Clean" has been run; clean overwrites in-place so file_path
# is always the single canonical file for this entry.
_CREATE_AUDIO_FILES = """
CREATE TABLE IF NOT EXISTS audio_files (
    id          TEXT PRIMARY KEY,
    profile_id  TEXT NOT NULL,
    filename    TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    duration    REAL,
    is_cleaned  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
)
"""

# epoch_losses: one row per epoch per profile.
# Epoch numbers are cumulative across all training runs (total_epochs_trained
# is the source of truth for the starting offset).
_CREATE_EPOCH_LOSSES = """
CREATE TABLE IF NOT EXISTS epoch_losses (
    id          TEXT PRIMARY KEY,
    profile_id  TEXT NOT NULL,
    epoch       INTEGER NOT NULL,
    loss_mel    REAL,
    loss_gen    REAL,
    loss_disc   REAL,
    loss_fm     REAL,
    loss_kl     REAL,
    trained_at  TEXT NOT NULL,
    UNIQUE (profile_id, epoch),
    FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
)
"""

# Columns added after the initial schema was shipped.
# Each entry: (column_name, column_definition)
# init_db() runs ALTER TABLE for any that are missing.
_MIGRATIONS: list[tuple[str, str]] = [
    ("batch_size",           "INTEGER NOT NULL DEFAULT 8"),
    ("audio_duration",       "REAL"),
    ("preprocessed_path",    "TEXT"),
    ("profile_dir",          "TEXT"),
    ("total_epochs_trained", "INTEGER NOT NULL DEFAULT 0"),
    ("checkpoint_path",      "TEXT"),   # G_latest.pth — resume checkpoint
    ("needs_retraining",     "INTEGER NOT NULL DEFAULT 0"),
    # Speech-weighted RMS of the first uploaded audio file for this profile.
    # All subsequent uploads are normalized to this value so the training corpus
    # is level-consistent.  Also used at inference time to match microphone input
    # level to the training distribution.  NULL on profiles created before this
    # migration — the worker falls back gracefully (no pre-gain applied).
    ("profile_rms",          "REAL"),
]


async def init_db() -> None:
    """Initialize the database and required data directories.

    Safe to call multiple times — all operations are idempotent.
    Creates:
      - backend/data/      (SQLite DB directory)
      - data/profiles/     (per-profile storage root)
      - profiles table     (CREATE TABLE IF NOT EXISTS)
      - audio_files table  (CREATE TABLE IF NOT EXISTS)
      - applies _MIGRATIONS for any columns missing from profiles
      - migrates legacy profiles (sample_path but no audio_files rows) to
        audio_files so the new UI always has rows to display
    """
    os.makedirs("backend/data", exist_ok=True)
    os.makedirs("data/profiles", exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        await db.execute(_CREATE_PROFILES)
        await db.execute(_CREATE_AUDIO_FILES)
        await db.execute(_CREATE_EPOCH_LOSSES)
        await db.commit()

        # Column-level migrations on profiles (idempotent)
        cursor = await db.execute("PRAGMA table_info(profiles)")
        existing = {row[1] for row in await cursor.fetchall()}

        for col_name, col_def in _MIGRATIONS:
            if col_name not in existing:
                await db.execute(
                    f"ALTER TABLE profiles ADD COLUMN {col_name} {col_def}"
                )

        await db.commit()

        # ------------------------------------------------------------------
        # Migrate legacy profiles: any profile that has a sample_path (or
        # preprocessed_path) but no audio_files rows yet gets a synthetic
        # audio_files entry pointing to the best available file on disk.
        # ------------------------------------------------------------------
        from datetime import datetime, timezone  # noqa: PLC0415
        import uuid as _uuid  # noqa: PLC0415

        cursor = await db.execute(
            "SELECT id, sample_path, preprocessed_path, audio_duration, created_at FROM profiles"
        )
        profiles = await cursor.fetchall()

        for p in profiles:
            pid = p["id"]
            cursor2 = await db.execute(
                "SELECT COUNT(*) FROM audio_files WHERE profile_id = ?", (pid,)
            )
            count = (await cursor2.fetchone())[0]
            if count > 0:
                continue  # already migrated

            # Pick the best available file: prefer cleaned, fall back to raw
            candidate_paths = []
            if p["preprocessed_path"] and os.path.exists(p["preprocessed_path"]):
                candidate_paths.append((p["preprocessed_path"], True))
            if p["sample_path"] and os.path.exists(p["sample_path"]):
                candidate_paths.append((p["sample_path"], False))

            if not candidate_paths:
                continue  # no file on disk — skip

            file_path, is_cleaned = candidate_paths[0]
            filename = os.path.basename(file_path)
            duration = p["audio_duration"]
            created_at = p["created_at"] or datetime.now(timezone.utc).isoformat()

            await db.execute(
                """INSERT INTO audio_files (id, profile_id, filename, file_path, duration, is_cleaned, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (_uuid.uuid4().hex, pid, filename, file_path, duration, 1 if is_cleaned else 0, created_at),
            )

        await db.commit()


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Async context manager that yields a configured aiosqlite connection.

    Sets row_factory to aiosqlite.Row so columns are accessible by name.

    Usage:
        async with get_db() as db:
            cursor = await db.execute("SELECT * FROM profiles")
            rows = await cursor.fetchall()
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def save_epoch_loss(
    profile_id: str,
    epoch: int,
    losses: dict,
) -> None:
    """Persist a single epoch's loss scalars.

    Safe to call from inside the training pipeline coroutine.
    Upserts on (profile_id, epoch) so retrying an epoch doesn't create
    duplicate rows.
    """
    import uuid as _uuid  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO epoch_losses
                   (id, profile_id, epoch, loss_mel, loss_gen, loss_disc, loss_fm, loss_kl, trained_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(profile_id, epoch)
               DO UPDATE SET
                   loss_mel=excluded.loss_mel,
                   loss_gen=excluded.loss_gen,
                   loss_disc=excluded.loss_disc,
                   loss_fm=excluded.loss_fm,
                   loss_kl=excluded.loss_kl,
                   trained_at=excluded.trained_at
            """,
            (
                _uuid.uuid4().hex,
                profile_id,
                epoch,
                losses.get("loss_mel"),
                losses.get("loss_gen"),
                losses.get("loss_disc"),
                losses.get("loss_fm"),
                losses.get("loss_kl"),
                now,
            ),
        )
        await db.commit()
