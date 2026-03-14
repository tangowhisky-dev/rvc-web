"""Shared pytest fixtures for the backend test suite.

Provides:
  - ``client`` async fixture: in-memory ASGI test client backed by an
    isolated per-test SQLite DB, so tests never touch ``backend/data/rvc.db``.
"""

import asyncio
import os
import pytest
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# Event-loop policy
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Isolated DB + client fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
async def client(tmp_path, monkeypatch):
    """Async ASGI test client with a clean per-test SQLite DB.

    Isolation strategy:
      - ``monkeypatch.setenv`` sets ``TEST_DB_PATH`` to a temp file.
      - ``backend.app.db`` is patched so ``DB_PATH`` and ``get_db()`` use
        the temp DB, not the production ``backend/data/rvc.db``.
      - ``DATA_DIR`` is set to ``tmp_path`` so uploaded files go to a temp
        directory, not the repo ``data/samples/`` directory.
    """
    import backend.app.db as db_module  # noqa: PLC0415

    temp_db = str(tmp_path / "test.db")
    temp_data_dir = str(tmp_path)

    # Redirect DB writes
    monkeypatch.setattr(db_module, "DB_PATH", temp_db)
    monkeypatch.setenv("DATA_DIR", temp_data_dir)

    # Initialize the schema in the temp DB
    await db_module.init_db()

    from backend.app.main import app  # noqa: PLC0415

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
