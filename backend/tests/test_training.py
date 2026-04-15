"""Contract tests for training API endpoints and WebSocket.

These tests INTENTIONALLY FAIL until T02–T03 implement the router and
TrainingManager. That failure proves the tests are real contract tests
that constrain the implementation.

Contracts:
  POST   /api/training/start    {profile_id} → 200 {job_id}
                                              → 404 unknown profile
                                              → 400 no sample_path
                                              → 409 job already running
  POST   /api/training/cancel   {profile_id} → 200 {ok: true}
  GET    /api/training/status/{profile_id}   → 200 {phase, progress_pct, status}
                                             → 404 no active job
  WS     /ws/training/{profile_id}           → JSON messages {type, message, phase}
                                               until type=="done"

Mock strategy: patch ``backend.app.routers.training.manager`` at module
scope so no real subprocess is ever spawned. The WebSocket tests use
``starlette.testclient.TestClient.websocket_connect()`` (sync API) since
``httpx`` does not support WebSocket.
"""

import asyncio
import io
import uuid
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wav_bytes() -> bytes:
    """Minimal valid-ish WAV payload for upload tests."""
    return (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00"
        b"\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00"
        b"data\x00\x00\x00\x00"
    )


def _make_fake_job(job_id: str = "test-job-001") -> MagicMock:
    """Build a fake TrainingJob with a pre-populated done queue."""
    job = MagicMock()
    job.job_id = job_id
    job.phase = "train"
    job.progress_pct = 100
    job.status = "done"
    job.error = None

    # Pre-populate queue with a single done message so WS consumer terminates
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait({"type": "done", "message": "ok", "phase": "train"})
    job.queue = q

    return job


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
async def _isolated_db(tmp_path, monkeypatch):
    """Set up an isolated temp DB (mirrors conftest.client pattern).

    Used by fixtures that need a real DB but build their own client.
    """
    import backend.app.db as db_module  # noqa: PLC0415

    temp_db = str(tmp_path / "test.db")
    monkeypatch.setattr(db_module, "DB_PATH", temp_db)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    await db_module.init_db()
    return tmp_path


@pytest.fixture()
async def client(tmp_path, monkeypatch):
    """Async ASGI client with isolated DB — reuses conftest pattern."""
    import backend.app.db as db_module  # noqa: PLC0415

    temp_db = str(tmp_path / "test.db")
    monkeypatch.setattr(db_module, "DB_PATH", temp_db)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    await db_module.init_db()

    from backend.app.main import app  # noqa: PLC0415

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture()
def sync_client(tmp_path, monkeypatch):
    """Synchronous Starlette TestClient — required for WebSocket tests."""
    import backend.app.db as db_module  # noqa: PLC0415

    temp_db = str(tmp_path / "test.db")
    monkeypatch.setattr(db_module, "DB_PATH", temp_db)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    asyncio.run(db_module.init_db())

    from backend.app.main import app  # noqa: PLC0415

    with TestClient(app, raise_server_exceptions=True) as tc:
        yield tc


@pytest.fixture()
def mock_manager(monkeypatch):
    """Patch backend.app.routers.training.manager with a MagicMock.

    Also sets RVC_ROOT to a dummy value so the router's env-check passes.
    The mock's methods return controlled values without spawning processes:
      - start_job  → fake TrainingJob (job_id="test-job-001", done queue)
      - cancel_job → None
      - get_job    → fake TrainingJob
    """
    monkeypatch.setenv("RVC_ROOT", "/fake/rvc_root")

    fake_job = _make_fake_job("test-job-001")

    m = MagicMock()
    m.start_job.return_value = fake_job
    m.cancel_job.return_value = None
    m.get_job.return_value = fake_job

    with patch("backend.app.routers.training.manager", m):
        yield m


# ---------------------------------------------------------------------------
# Helper: create a real profile in the DB via the REST API
# ---------------------------------------------------------------------------

async def _create_profile(client: AsyncClient, name: str = "test_voice") -> str:
    """POST /api/profiles and return the profile_id."""
    resp = await client.post(
        "/api/profiles",
        files={"file": ("voice.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"name": name},
    )
    assert resp.status_code == 200, f"Profile creation failed: {resp.text}"
    return resp.json()["id"]


def _create_profile_sync(sync_client: TestClient, tmp_path, name: str = "test_voice") -> str:
    """Create a profile synchronously for use in sync_client tests."""
    resp = sync_client.post(
        "/api/profiles",
        files={"file": ("voice.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"name": name},
    )
    assert resp.status_code == 200, f"Profile creation failed: {resp.text}"
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# POST /api/training/start
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_start_training_success(client, mock_manager):
    """POST /api/training/start with valid profile → 200 {job_id}."""
    profile_id = await _create_profile(client)

    resp = await client.post(
        "/api/training/start",
        json={"profile_id": profile_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["job_id"] == "test-job-001"


@pytest.mark.anyio
async def test_start_training_unknown_profile(client, mock_manager):
    """POST /api/training/start with unknown profile_id → 404."""
    resp = await client.post(
        "/api/training/start",
        json={"profile_id": "does-not-exist-" + uuid.uuid4().hex},
    )
    assert resp.status_code == 404, resp.text


@pytest.mark.anyio
async def test_start_training_no_sample(client, mock_manager):
    """POST /api/training/start with profile that has sample_path=None → 400."""
    import backend.app.db as db_module  # noqa: PLC0415

    # Insert a profile row directly with NULL sample_path
    bare_id = uuid.uuid4().hex
    async with db_module.get_db() as db:
        await db.execute(
            "INSERT INTO profiles (id, name, status, sample_path, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (bare_id, "no_sample_profile", "untrained", None),
        )
        await db.commit()

    resp = await client.post(
        "/api/training/start",
        json={"profile_id": bare_id},
    )
    assert resp.status_code == 400, resp.text


@pytest.mark.anyio
async def test_start_training_conflict(client, mock_manager):
    """POST /api/training/start twice → 409 on second call (job already running)."""
    profile_id = await _create_profile(client)

    # First call succeeds
    resp1 = await client.post(
        "/api/training/start",
        json={"profile_id": profile_id},
    )
    assert resp1.status_code == 200, resp1.text

    # Second call: mock raises ValueError to simulate active job
    mock_manager.start_job.side_effect = ValueError("job already running")

    resp2 = await client.post(
        "/api/training/start",
        json={"profile_id": profile_id},
    )
    assert resp2.status_code == 409, resp2.text


# ---------------------------------------------------------------------------
# POST /api/training/cancel
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_cancel_training(client, mock_manager):
    """POST /api/training/cancel with profile_id → 200 {ok: true}."""
    profile_id = await _create_profile(client)

    resp = await client.post(
        "/api/training/cancel",
        json={"profile_id": profile_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("ok") is True


# ---------------------------------------------------------------------------
# GET /api/training/status/{profile_id}
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_get_status_active_job(client, mock_manager):
    """GET /api/training/status/<profile_id> → 200 with {phase, progress_pct, status}."""
    profile_id = await _create_profile(client)

    resp = await client.get(f"/api/training/status/{profile_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "phase" in body, f"'phase' missing from {body}"
    assert "progress_pct" in body, f"'progress_pct' missing from {body}"
    assert "status" in body, f"'status' missing from {body}"


@pytest.mark.anyio
async def test_get_status_no_job(client, mock_manager):
    """GET /api/training/status/unknown → 200 idle when no active job (not 404)."""
    mock_manager.get_job.return_value = None

    resp = await client.get("/api/training/status/unknown-profile-id")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["phase"] == "idle"
    assert body["status"] == "idle"


# ---------------------------------------------------------------------------
# WS /ws/training/{profile_id}
# ---------------------------------------------------------------------------

def test_websocket_receives_messages(sync_client, mock_manager, tmp_path):
    """WebSocket /ws/training/<profile_id> delivers {type, message, phase} messages.

    Uses starlette.testclient.TestClient.websocket_connect() — the sync
    WebSocket API — since httpx does not support WebSocket.

    The mock_manager's get_job returns a fake job whose queue is pre-loaded
    with a single {type:"done"} message.  The test reads until type=="done"
    and asserts at least one message was received with the required keys.
    """
    profile_id = _create_profile_sync(sync_client, tmp_path)

    received = []

    with sync_client.websocket_connect(f"/ws/training/{profile_id}") as ws:
        while True:
            msg = ws.receive_json()
            received.append(msg)
            if msg.get("type") in ("done", "error"):
                break

    assert len(received) >= 1, "Expected at least one WebSocket message"

    # Every message must carry the required keys
    for msg in received:
        assert "type" in msg, f"'type' key missing from message: {msg}"
        assert "message" in msg, f"'message' key missing from message: {msg}"
        assert "phase" in msg, f"'phase' key missing from message: {msg}"
