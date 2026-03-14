"""Contract tests for realtime API endpoints and WebSocket.

These tests INTENTIONALLY FAIL until T02–T03 implement the router and
RealtimeManager. That failure proves the tests are real contract tests
that constrain the implementation.

Contracts:
  POST   /api/realtime/start   {profile_id, input_device_id, output_device_id,
                                 pitch, index_rate, protect}
                               → 200 {session_id}
                               → 404 unknown profile
                               → 400 profile not trained
                               → 400 no index file found
                               → 409 session already active
  POST   /api/realtime/stop    {session_id}
                               → 200 {ok: true}
                               → 404 unknown session
  POST   /api/realtime/params  {session_id, pitch?, index_rate?, protect?}
                               → 200 {ok: true}
  GET    /api/realtime/status  → 200 {active, session_id, profile_id,
                                      input_device, output_device}
  WS     /ws/realtime/{id}     → {type:"waveform_in"|"waveform_out", samples, sr}
                                  messages until {type:"done"}

Mock strategy: patch ``backend.app.routers.realtime.manager`` at router-module
scope so no real audio hardware is touched. WebSocket tests use
``starlette.testclient.TestClient.websocket_connect()`` (sync API) since
``httpx`` does not support WebSocket.
"""

import asyncio
import collections
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_session(session_id: str = "test-session-001") -> MagicMock:
    """Build a fake RealtimeSession with a pre-populated waveform deque.

    The deque holds 2 waveform messages followed by a sentinel ``None``.
    The WS handler should send ``{type:"done"}`` when it dequeues ``None``
    or when ``_stop_event`` is set.
    """
    session = MagicMock()
    session.session_id = session_id
    session.profile_id = "test-profile-001"
    session.status = "active"
    session.input_device = 2
    session.output_device = 1
    session.error = None

    # _stop_event: threading.Event, not set (session is still running)
    stop_event = threading.Event()
    session._stop_event = stop_event

    # Pre-populate deque: 2 waveform frames + sentinel None
    deque = collections.deque(maxlen=40)
    deque.append({"type": "waveform_in", "samples": "AAAA", "sr": 48000})
    deque.append({"type": "waveform_out", "samples": "BBBB", "sr": 48000})
    deque.append(None)  # sentinel — tells WS handler to send {type:"done"}
    session._waveform_deque = deque

    return session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
async def client(tmp_path, monkeypatch):
    """Async ASGI client with isolated DB — mirrors conftest pattern."""
    import backend.app.db as db_module  # noqa: PLC0415

    temp_db = str(tmp_path / "test.db")
    monkeypatch.setattr(db_module, "DB_PATH", temp_db)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    await db_module.init_db()

    from backend.app.main import app  # noqa: PLC0415

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture()
def sync_client(tmp_path, monkeypatch):
    """Synchronous Starlette TestClient — required for WebSocket tests.

    httpx does not support WebSocket; starlette.testclient does.
    Uses the same DB isolation strategy as the async ``client`` fixture.
    """
    import backend.app.db as db_module  # noqa: PLC0415

    temp_db = str(tmp_path / "test.db")
    monkeypatch.setattr(db_module, "DB_PATH", temp_db)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    # Init DB synchronously (Python 3.11 compatible)
    asyncio.run(db_module.init_db())

    from backend.app.main import app  # noqa: PLC0415

    with TestClient(app, raise_server_exceptions=True) as tc:
        yield tc


@pytest.fixture()
def mock_manager(monkeypatch):
    """Patch ``backend.app.routers.realtime.manager`` with a MagicMock.

    Provides controlled return values without touching audio hardware:
      - ``active_session()``  → None  (no active session by default)
      - ``start_session()``   → fake RealtimeSession
      - ``stop_session()``    → None
      - ``get_session()``     → fake RealtimeSession

    Also inserts a trained profile into the DB so the router's profile-lookup
    and 400 vs 404 distinction tests work correctly.
    """
    fake_session = _make_fake_session("test-session-001")

    m = MagicMock()
    m.active_session.return_value = None
    m.start_session.return_value = fake_session
    m.stop_session.return_value = None
    m.get_session.return_value = fake_session

    with patch("backend.app.routers.realtime.manager", m):
        yield m


# ---------------------------------------------------------------------------
# Helper: insert a trained profile row directly into the isolated DB
# ---------------------------------------------------------------------------

async def _insert_trained_profile(
    tmp_path,
    profile_id: str,
    status: str = "trained",
) -> None:
    """Insert a minimal profile row with given status into the temp DB."""
    import backend.app.db as db_module  # noqa: PLC0415

    async with db_module.get_db() as db:
        await db.execute(
            "INSERT INTO profiles (id, name, status, sample_path, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (profile_id, "test_voice", status, str(tmp_path / "voice.wav")),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# POST /api/realtime/start
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_start_returns_session_id(client, mock_manager, tmp_path):
    """POST /api/realtime/start with valid trained profile → 200 {session_id}."""
    profile_id = uuid.uuid4().hex
    await _insert_trained_profile(tmp_path, profile_id, status="trained")

    resp = await client.post(
        "/api/realtime/start",
        json={
            "profile_id": profile_id,
            "input_device_id": 2,
            "output_device_id": 1,
            "pitch": 0,
            "index_rate": 0.75,
            "protect": 0.33,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "session_id" in body, f"'session_id' missing from response: {body}"


@pytest.mark.anyio
async def test_start_404_unknown_profile(client, mock_manager):
    """POST /api/realtime/start with nonexistent profile_id → 404."""
    resp = await client.post(
        "/api/realtime/start",
        json={
            "profile_id": "does-not-exist-" + uuid.uuid4().hex,
            "input_device_id": 2,
            "output_device_id": 1,
            "pitch": 0,
            "index_rate": 0.75,
            "protect": 0.33,
        },
    )
    assert resp.status_code == 404, resp.text


@pytest.mark.anyio
async def test_start_400_profile_not_trained(client, mock_manager, tmp_path):
    """POST /api/realtime/start with untrained profile → 400 mentioning 'not trained'."""
    profile_id = uuid.uuid4().hex
    await _insert_trained_profile(tmp_path, profile_id, status="untrained")

    resp = await client.post(
        "/api/realtime/start",
        json={
            "profile_id": profile_id,
            "input_device_id": 2,
            "output_device_id": 1,
            "pitch": 0,
            "index_rate": 0.75,
            "protect": 0.33,
        },
    )
    assert resp.status_code == 400, resp.text
    assert "not trained" in resp.text.lower(), (
        f"Expected 'not trained' in response body, got: {resp.text}"
    )


@pytest.mark.anyio
async def test_start_400_no_index_file(client, mock_manager, tmp_path):
    """POST /api/realtime/start when index file missing → 400."""
    profile_id = uuid.uuid4().hex
    await _insert_trained_profile(tmp_path, profile_id, status="trained")

    # Manager raises 400 HTTPException when FAISS index is not found
    mock_manager.start_session.side_effect = HTTPException(
        status_code=400, detail="no index file found"
    )

    resp = await client.post(
        "/api/realtime/start",
        json={
            "profile_id": profile_id,
            "input_device_id": 2,
            "output_device_id": 1,
            "pitch": 0,
            "index_rate": 0.75,
            "protect": 0.33,
        },
    )
    assert resp.status_code == 400, resp.text


@pytest.mark.anyio
async def test_start_409_already_active(client, mock_manager, tmp_path):
    """POST /api/realtime/start when a session is already active → 409."""
    profile_id = uuid.uuid4().hex
    await _insert_trained_profile(tmp_path, profile_id, status="trained")

    # Simulate an existing active session
    mock_manager.active_session.return_value = _make_fake_session("existing-session")

    resp = await client.post(
        "/api/realtime/start",
        json={
            "profile_id": profile_id,
            "input_device_id": 2,
            "output_device_id": 1,
            "pitch": 0,
            "index_rate": 0.75,
            "protect": 0.33,
        },
    )
    assert resp.status_code == 409, resp.text


# ---------------------------------------------------------------------------
# POST /api/realtime/stop
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_stop_200_ok(client, mock_manager):
    """POST /api/realtime/stop with valid session_id → 200 {ok: true}."""
    session_id = "test-session-001"
    mock_manager.get_session.return_value = _make_fake_session(session_id)

    resp = await client.post(
        "/api/realtime/stop",
        json={"session_id": session_id},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("ok") is True, f"Expected {{ok: true}}, got: {body}"


@pytest.mark.anyio
async def test_stop_404_unknown(client, mock_manager):
    """POST /api/realtime/stop with unknown session_id → 404."""
    mock_manager.get_session.return_value = None

    resp = await client.post(
        "/api/realtime/stop",
        json={"session_id": "does-not-exist-" + uuid.uuid4().hex},
    )
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# POST /api/realtime/params
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_params_200_ok(client, mock_manager):
    """POST /api/realtime/params with valid session → 200 {ok: true}."""
    session_id = "test-session-001"
    fake_session = _make_fake_session(session_id)
    # Add a mock rvc attribute so the router can call set_key / set_index_rate
    fake_session.rvc = MagicMock()
    mock_manager.get_session.return_value = fake_session

    resp = await client.post(
        "/api/realtime/params",
        json={
            "session_id": session_id,
            "pitch": 2,
            "index_rate": 0.5,
            "protect": 0.33,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("ok") is True, f"Expected {{ok: true}}, got: {body}"


# ---------------------------------------------------------------------------
# GET /api/realtime/status
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_status_inactive(client, mock_manager):
    """GET /api/realtime/status when no active session → 200 {active: false}."""
    mock_manager.active_session.return_value = None

    resp = await client.get("/api/realtime/status")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("active") is False, (
        f"Expected 'active' == false, got: {body}"
    )


# ---------------------------------------------------------------------------
# WS /ws/realtime/{session_id}
# ---------------------------------------------------------------------------

def test_ws_streams_waveform_and_done(sync_client, mock_manager, tmp_path):
    """WS /ws/realtime/{session_id} delivers waveform messages then {type:'done'}.

    Uses ``starlette.testclient.TestClient.websocket_connect()`` — the sync
    WebSocket API — since httpx does not support WebSocket.

    The mock session's ``_waveform_deque`` is pre-loaded with 2 waveform
    messages and a sentinel ``None`` that signals done. The test verifies:
      - The first 2 messages have ``type`` in (``waveform_in``, ``waveform_out``)
      - The final message has ``type == "done"``
    """
    session_id = "test-session-ws-001"
    fake_session = _make_fake_session(session_id)
    mock_manager.get_session.return_value = fake_session

    # Also wire active_session and the lookup used by WS handler
    mock_manager.active_session.return_value = fake_session

    received = []

    with sync_client.websocket_connect(f"/ws/realtime/{session_id}") as ws:
        while True:
            msg = ws.receive_json()
            received.append(msg)
            if msg.get("type") in ("done", "error"):
                break

    assert len(received) >= 3, (
        f"Expected at least 3 messages (2 waveform + done), got {len(received)}: {received}"
    )

    # First two messages must be waveform_in or waveform_out
    for msg in received[:-1]:
        assert msg.get("type") in ("waveform_in", "waveform_out"), (
            f"Expected waveform message type, got: {msg}"
        )

    # Final message must be done
    assert received[-1].get("type") == "done", (
        f"Expected final message to be {{type:'done'}}, got: {received[-1]}"
    )
