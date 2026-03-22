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

# Columns added after the initial schema was shipped.
# Each entry: (column_name, column_definition)
# init_db() runs ALTER TABLE for any that are missing.
_MIGRATIONS: list[tuple[str, str]] = [
    ("batch_size",        "INTEGER NOT NULL DEFAULT 8"),
    ("audio_duration",    "REAL"),
    ("preprocessed_path", "TEXT"),
    ("profile_dir",       "TEXT"),
]


async def init_db() -> None:
    """Initialize the database and required data directories.

    Safe to call multiple times — all operations are idempotent.
    Creates:
      - backend/data/      (SQLite DB directory)
      - data/profiles/     (per-profile storage root)
      - profiles table     (CREATE TABLE IF NOT EXISTS)
      - applies _MIGRATIONS for any columns missing from an existing table
    """
    os.makedirs("backend/data", exist_ok=True)
    os.makedirs("data/profiles", exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_PROFILES)
        await db.commit()

        # Column-level migrations (idempotent — skips columns that already exist)
        cursor = await db.execute("PRAGMA table_info(profiles)")
        existing = {row[1] for row in await cursor.fetchall()}

        for col_name, col_def in _MIGRATIONS:
            if col_name not in existing:
                await db.execute(
                    f"ALTER TABLE profiles ADD COLUMN {col_name} {col_def}"
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
