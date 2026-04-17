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

# All persistent data lives under PROJECT_ROOT/data/.
# Falls back to a path relative to this file so tests work without the env var.
_project_root = os.environ.get(
    "PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
DB_PATH = os.path.join(_project_root, "data", "rvc.db")

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
    loss_spk    REAL,
    trained_at  TEXT NOT NULL,
    UNIQUE (profile_id, epoch),
    FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
)
"""

_CREATE_BEATRICE_STEPS = """
CREATE TABLE IF NOT EXISTS beatrice_steps (
    id          TEXT PRIMARY KEY,
    profile_id  TEXT NOT NULL,
    step        INTEGER NOT NULL,
    loss_mel    REAL,
    loss_loud   REAL,
    loss_ap     REAL,
    loss_adv    REAL,
    loss_fm     REAL,
    loss_d      REAL,
    trained_at  TEXT NOT NULL,
    UNIQUE (profile_id, step)
)
"""

# Columns added after the initial schema was shipped.
# Each entry: (column_name, column_definition)
# init_db() runs ALTER TABLE for any that are missing.
_MIGRATIONS: list[tuple[str, str]] = [
    ("batch_size", "INTEGER NOT NULL DEFAULT 8"),
    ("audio_duration", "REAL"),
    ("preprocessed_path", "TEXT"),
    ("profile_dir", "TEXT"),
    ("total_epochs_trained", "INTEGER NOT NULL DEFAULT 0"),
    ("checkpoint_path", "TEXT"),  # G_latest.pth — resume checkpoint
    ("needs_retraining", "INTEGER NOT NULL DEFAULT 0"),
    # Speech-weighted RMS of the first uploaded audio file for this profile.
    # All subsequent uploads are normalized to this value so the training corpus
    # is level-consistent.  Also used at inference time to match microphone input
    # level to the training distribution.  NULL on profiles created before this
    # migration — the worker falls back gracefully (no pre-gain applied).
    ("profile_rms", "REAL"),
    # Feature embedder model name (e.g. "spin-v2", "contentvec", "hubert").
    # Locked after the first training run: switching embedder mid-training would
    # corrupt the stored feature vectors in 3_feature768/. NULL = "spin-v2"
    # (default for profiles created before this migration).
    ("embedder", "TEXT NOT NULL DEFAULT 'spin-v2'"),
    # Overtraining detection threshold — number of consecutive epochs where
    # generator loss does not improve before training is stopped early.
    # 0 = disabled.  NULL treated as 0.
    ("overtrain_threshold", "INTEGER NOT NULL DEFAULT 0"),
    # Vocoder used to train this profile: "HiFi-GAN" or "RefineGAN".
    # Locked after the first training run — the decoder architecture is baked
    # into the G_latest.pth checkpoint and cannot be swapped mid-training.
    ("vocoder", "TEXT NOT NULL DEFAULT 'HiFi-GAN'"),
    # Inference-ready fp16 model from the best-loss epoch of the most recent
    # training run (profile_dir/model_best.pth).  NULL on older profiles.
    ("best_model_path", "TEXT"),
    # Epoch number and avg-generator-loss of the best epoch, for display.
    ("best_epoch", "INTEGER"),
    ("best_avg_gen_loss", "REAL"),
    # Adversarial loss type: "lsgan" (default, original RVC) or "tprls"
    # (Truncated Paired Relative Least Squares — more stable when discriminator
    # dominates early training).  Not locked — can be changed between runs.
    ("adv_loss", "TEXT NOT NULL DEFAULT 'lsgan'"),
    # Cyclic KL annealing: ramps kl_beta 0→1 over kl_anneal_epochs then repeats,
    # preventing posterior collapse in the first N epochs of each cycle.
    # 0 = disabled (kl_beta = 1.0 always).
    ("kl_anneal", "INTEGER NOT NULL DEFAULT 0"),
    ("kl_anneal_epochs", "INTEGER NOT NULL DEFAULT 40"),
    # Optimizer: "adamw" (default) or "adamspd" (Adam with Selective Projection
    # Decay — applies stiffness penalty toward pretrain anchor weights to limit
    # catastrophic forgetting).  Not locked — can be changed between runs.
    ("optimizer", "TEXT NOT NULL DEFAULT 'adamw'"),
    # Engine / pipeline used by this profile: "rvc" (default) or "beatrice2".
    # Locked at profile creation — cannot be changed after the first training run.
    ("pipeline", "TEXT NOT NULL DEFAULT 'rvc'"),
]

# Migrations for epoch_losses table
_EPOCH_LOSSES_MIGRATIONS: list[tuple[str, str]] = [
    ("loss_spk", "REAL"),
]

# beatrice_steps was originally created with a phantom loss_g column.
# New schema drops loss_g and uses the real component losses.
# SQLite can't drop columns (pre-3.35), so we just ensure the real columns exist.
_BEATRICE_STEPS_MIGRATIONS: list[tuple[str, str]] = [
    ("loss_mel",  "REAL"),
    ("loss_loud", "REAL"),
    ("loss_ap",   "REAL"),
    ("loss_adv",  "REAL"),
    ("loss_fm",   "REAL"),
    ("loss_d",    "REAL"),
    ("utmos",     "REAL"),   # UTMOS MOS score (1–5); written at evaluation_interval steps only
    ("is_best",   "INTEGER NOT NULL DEFAULT 0"),  # 1 if this step had best UTMOS so far
]


async def init_db() -> None:
    """Initialize the database and required data directories.

    Safe to call multiple times — all operations are idempotent.
    Creates:
      - data/           (SQLite DB and profiles root)
      - data/profiles/  (per-profile storage root)
      - profiles table     (CREATE TABLE IF NOT EXISTS)
      - audio_files table  (CREATE TABLE IF NOT EXISTS)
      - applies _MIGRATIONS for any columns missing from profiles
      - migrates legacy profiles (sample_path but no audio_files rows) to
        audio_files so the new UI always has rows to display
    """
    os.makedirs(os.path.join(_project_root, "data"), exist_ok=True)
    os.makedirs(os.path.join(_project_root, "data", "profiles"), exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        await db.execute(_CREATE_PROFILES)
        await db.execute(_CREATE_AUDIO_FILES)
        await db.execute(_CREATE_EPOCH_LOSSES)
        await db.execute(_CREATE_BEATRICE_STEPS)
        await db.commit()

        # Column-level migrations on profiles (idempotent)
        cursor = await db.execute("PRAGMA table_info(profiles)")
        existing = {row[1] for row in await cursor.fetchall()}

        for col_name, col_def in _MIGRATIONS:
            if col_name not in existing:
                await db.execute(
                    f"ALTER TABLE profiles ADD COLUMN {col_name} {col_def}"
                )

        # Column-level migrations on epoch_losses (idempotent)
        cursor = await db.execute("PRAGMA table_info(epoch_losses)")
        existing_el = {row[1] for row in await cursor.fetchall()}

        for col_name, col_def in _EPOCH_LOSSES_MIGRATIONS:
            if col_name not in existing_el:
                await db.execute(
                    f"ALTER TABLE epoch_losses ADD COLUMN {col_name} {col_def}"
                )

        # Column-level migrations on beatrice_steps (idempotent)
        cursor = await db.execute("PRAGMA table_info(beatrice_steps)")
        existing_bs = {row[1] for row in await cursor.fetchall()}

        for col_name, col_def in _BEATRICE_STEPS_MIGRATIONS:
            if col_name not in existing_bs:
                await db.execute(
                    f"ALTER TABLE beatrice_steps ADD COLUMN {col_name} {col_def}"
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
                (
                    _uuid.uuid4().hex,
                    pid,
                    filename,
                    file_path,
                    duration,
                    1 if is_cleaned else 0,
                    created_at,
                ),
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
                   (id, profile_id, epoch, loss_mel, loss_gen, loss_disc, loss_fm, loss_kl, loss_spk, trained_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(profile_id, epoch)
               DO UPDATE SET
                   loss_mel=excluded.loss_mel,
                   loss_gen=excluded.loss_gen,
                   loss_disc=excluded.loss_disc,
                   loss_fm=excluded.loss_fm,
                   loss_kl=excluded.loss_kl,
                   loss_spk=excluded.loss_spk,
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
                losses.get("loss_spk"),
                now,
            ),
        )
        await db.commit()
