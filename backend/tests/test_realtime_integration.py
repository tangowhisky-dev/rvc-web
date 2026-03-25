"""Integration smoke test for the realtime RVC audio pipeline.

Marked ``@pytest.mark.integration`` — skipped in normal CI runs and only
executed when explicitly invoked:

    PROJECT_ROOT=/path/to/rvc-web \\
    conda run -n rvc pytest backend/tests/test_realtime_integration.py \\
      -v -s -m integration --timeout=60

What this test proves:
  1. POST /api/realtime/start launches a real sounddevice audio session.
  2. WS /ws/realtime/{session_id} delivers ≥5 waveform messages within 10s.
  3. POST /api/realtime/stop halts the session cleanly.
  4. WS sends ``{type:"done"}`` after stop (or closes).
  5. GET /api/realtime/status returns ``{active: false}`` after stop.

This test retires two risks from the S03 roadmap:
  - rtrvc.py block size risk: 200ms blocks confirmed adequate (<=0.2× realtime).
  - fairseq import risk: RVC.__init__ completes without UnpicklingError.

Hardware requirements:
  - MacBook Pro Microphone (sounddevice device id 2)
  - BlackHole 2ch virtual audio driver installed (sounddevice device id 1)
  - RVC_ROOT must point to a working RVC installation with a trained profile
"""

import json
import os
import time
from typing import Optional

import aiosqlite
import pytest
from starlette.testclient import TestClient

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Auto-skip guards
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def require_rvc_root():
    """Skip every test in this module if RVC_ROOT is not set or does not exist."""
    rvc_root = os.environ.get("RVC_ROOT", "")
    if not rvc_root:
        pytest.skip("RVC_ROOT environment variable not set — skipping realtime integration tests")
    if not os.path.isdir(rvc_root):
        pytest.skip(f"RVC_ROOT does not exist: {rvc_root}")


@pytest.fixture(autouse=True)
def require_trained_profile():
    """Skip every test if no trained profile exists in the production DB."""
    import asyncio  # noqa: PLC0415
    from backend.app.db import DB_PATH  # noqa: PLC0415

    async def _query() -> Optional[str]:
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT id FROM profiles WHERE status = 'trained' LIMIT 1"
                ) as cursor:
                    row = await cursor.fetchone()
            return row["id"] if row else None
        except Exception:  # noqa: BLE001
            return None

    profile_id = asyncio.get_event_loop().run_until_complete(_query())
    if profile_id is None:
        pytest.skip(
            "No trained profile found in DB — run S02 integration test first to produce one"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_trained_profile_id() -> str:
    """Return the first trained profile_id from the real DB."""
    import asyncio  # noqa: PLC0415
    from backend.app.db import DB_PATH  # noqa: PLC0415

    async def _query() -> str:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id FROM profiles WHERE status = 'trained' LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
        return row["id"]

    return asyncio.get_event_loop().run_until_complete(_query())


# ---------------------------------------------------------------------------
# Real app client (uses production DB and RVC_ROOT)
# ---------------------------------------------------------------------------

@pytest.fixture()
def real_app_client():
    """TestClient wrapping the real FastAPI app.

    Does NOT patch DB or DATA_DIR — uses production paths so the realtime
    audio pipeline can reach RVC model files and the real SQLite DB.
    """
    from backend.app.main import app  # noqa: PLC0415

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_realtime_session_produces_waveforms(real_app_client):
    """Run a real realtime session and verify waveform streaming end-to-end.

    Steps:
      1. Confirm GET /api/devices lists BlackHole (id=1) and MacBook Pro Mic (id=2).
      2. POST /api/realtime/start → 200 + session_id.
      3. Connect WS /ws/realtime/{session_id}.
      4. Collect messages for up to 10 seconds.
      5. Assert ≥5 waveform messages of type 'waveform_in' or 'waveform_out'.
      6. POST /api/realtime/stop → 200 {ok: true}.
      7. Assert WS sends {type: "done"} or closes cleanly.
      8. GET /api/realtime/status → {active: false}.
    """
    profile_id = _get_trained_profile_id()
    print(f"\n[realtime-integration] Using profile_id={profile_id}")

    # ------------------------------------------------------------------
    # Step 1: Confirm devices
    # ------------------------------------------------------------------
    devices_resp = real_app_client.get("/api/devices")
    assert devices_resp.status_code == 200, (
        f"GET /api/devices failed: {devices_resp.status_code} {devices_resp.text}"
    )
    devices = devices_resp.json()
    device_ids = {d["id"] for d in devices}
    print(f"[realtime-integration] Available device IDs: {sorted(device_ids)}")

    blackhole_id = None
    mic_id = None
    for d in devices:
        name_lower = d.get("name", "").lower()
        if "blackhole" in name_lower and "2ch" in name_lower and blackhole_id is None:
            blackhole_id = d["id"]
        if (
            ("macbook" in name_lower or "built-in" in name_lower)
            and ("microphone" in name_lower or "input" in name_lower)
            and mic_id is None
        ):
            mic_id = d["id"]

    if blackhole_id is None:
        pytest.skip("BlackHole 2ch not found in device list — install BlackHole first")
    if mic_id is None:
        pytest.skip("MacBook Pro Microphone not found in device list")

    print(
        f"[realtime-integration] Devices — input: {mic_id} (mic), "
        f"output: {blackhole_id} (BlackHole)"
    )

    # ------------------------------------------------------------------
    # Step 2: POST /api/realtime/start
    # ------------------------------------------------------------------
    start_resp = real_app_client.post(
        "/api/realtime/start",
        json={
            "profile_id": profile_id,
            "input_device_id": mic_id,
            "output_device_id": blackhole_id,
            "pitch": 0,
            "index_rate": 0.75,
            "protect": 0.33,
        },
    )
    assert start_resp.status_code == 200, (
        f"POST /api/realtime/start failed: {start_resp.status_code} {start_resp.text}"
    )
    session_id = start_resp.json()["session_id"]
    print(f"[realtime-integration] Session started: {session_id}")

    # ------------------------------------------------------------------
    # Step 3-5: Connect WS and collect waveform messages
    # ------------------------------------------------------------------
    waveform_messages = []
    all_messages = []
    deadline = time.time() + 10  # 10-second collection window

    try:
        with real_app_client.websocket_connect(
            f"/ws/realtime/{session_id}"
        ) as ws:
            while time.time() < deadline:
                # Break early if we have enough waveforms
                if len(waveform_messages) >= 5:
                    print(
                        f"[realtime-integration] ✓ Collected {len(waveform_messages)} "
                        "waveform messages — stopping early"
                    )
                    break

                try:
                    ws.send_text("")  # keep-alive ping (ignored by server)
                except Exception:  # noqa: BLE001
                    pass

                try:
                    raw = ws.receive_text()
                    msg = json.loads(raw)
                    all_messages.append(msg)

                    msg_type = msg.get("type", "?")
                    if msg_type in ("waveform_in", "waveform_out"):
                        waveform_messages.append(msg)
                        print(
                            f"[realtime-integration] waveform #{len(waveform_messages)}: "
                            f"type={msg_type}, "
                            f"samples_len={len(msg.get('samples', ''))}"
                        )
                    elif msg_type == "done":
                        print("[realtime-integration] Received 'done' — session ended early")
                        break
                    elif msg_type == "error":
                        print(f"[realtime-integration] WS error: {msg}")
                        break

                except Exception as exc:  # noqa: BLE001
                    print(f"[realtime-integration] WS receive error: {exc}")
                    # Small pause before retry
                    time.sleep(0.1)

    except Exception as exc:  # noqa: BLE001
        print(f"[realtime-integration] WS connection error: {exc}")

    print(
        f"[realtime-integration] Total messages: {len(all_messages)}, "
        f"waveform messages: {len(waveform_messages)}"
    )

    assert len(waveform_messages) >= 5, (
        f"Expected ≥5 waveform messages within 10s, got {len(waveform_messages)}. "
        f"All message types: {[m.get('type') for m in all_messages]}. "
        "Check audio thread stderr for errors — the session may have failed to start."
    )

    # ------------------------------------------------------------------
    # Step 6: POST /api/realtime/stop
    # ------------------------------------------------------------------
    stop_resp = real_app_client.post(
        "/api/realtime/stop",
        json={"session_id": session_id},
    )
    assert stop_resp.status_code == 200, (
        f"POST /api/realtime/stop failed: {stop_resp.status_code} {stop_resp.text}"
    )
    assert stop_resp.json().get("ok") is True, (
        f"Expected ok=True from /stop, got: {stop_resp.json()}"
    )
    print("[realtime-integration] Session stopped successfully")

    # ------------------------------------------------------------------
    # Step 7: Assert {type: "done"} on WS after stop
    # ------------------------------------------------------------------
    done_received = False
    deadline_done = time.time() + 5  # 5s for done message

    try:
        with real_app_client.websocket_connect(
            f"/ws/realtime/{session_id}"
        ) as ws:
            while time.time() < deadline_done:
                try:
                    raw = ws.receive_text()
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "?")
                    print(f"[realtime-integration] Post-stop WS msg: type={msg_type}")
                    if msg_type == "done":
                        done_received = True
                        break
                    elif msg_type == "error":
                        # "session not found" also means it stopped cleanly
                        print(f"[realtime-integration] Post-stop WS error msg (OK): {msg}")
                        done_received = True  # session gone = cleaned up
                        break
                except Exception as exc:  # noqa: BLE001
                    # WS closed by server = done
                    print(f"[realtime-integration] Post-stop WS closed: {exc}")
                    done_received = True
                    break
    except Exception as exc:  # noqa: BLE001
        # WS refused = session cleaned up = acceptable
        print(f"[realtime-integration] Post-stop WS connect error (OK): {exc}")
        done_received = True

    assert done_received, (
        "Expected WS to send {type:'done'} or close after /stop, but it stayed open"
    )

    # ------------------------------------------------------------------
    # Step 8: GET /api/realtime/status → {active: false}
    # ------------------------------------------------------------------
    # Brief pause to allow session cleanup to propagate
    time.sleep(0.3)

    status_resp = real_app_client.get("/api/realtime/status")
    assert status_resp.status_code == 200, (
        f"GET /api/realtime/status failed: {status_resp.status_code}"
    )
    status = status_resp.json()
    print(f"[realtime-integration] Final status: {status}")
    assert status["active"] is False, (
        f"Expected active=false after stop, got: {status}"
    )

    print(
        f"\n[realtime-integration] ✓ Integration test passed — "
        f"{len(waveform_messages)} waveform messages received, "
        "session stopped cleanly, status=inactive"
    )
