"""SQLite database initialization and connection helpers.

Provides:
  - DB_PATH: canonical path for the SQLite database file
  - init_db(): idempotent schema creation + required directory setup
  - get_db(): async context manager yielding an aiosqlite.Connection
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aiosqlite

DB_PATH = "backend/data/rvc.db"

_CREATE_PROFILES = """
CREATE TABLE IF NOT EXISTS profiles (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'untrained',
    sample_path TEXT,
    model_path  TEXT,
    index_path  TEXT,
    created_at  TEXT NOT NULL
)
"""


async def init_db() -> None:
    """Initialize the database and required data directories.

    Safe to call multiple times — all operations are idempotent.
    Creates:
      - backend/data/      (SQLite DB directory)
      - data/samples/      (uploaded audio storage root)
      - profiles table     (CREATE TABLE IF NOT EXISTS)
    """
    os.makedirs("backend/data", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_PROFILES)
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
