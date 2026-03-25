"""Integration smoke test for the full RVC training pipeline.

Marked ``@pytest.mark.integration`` — skipped in normal CI runs and only
executed when explicitly invoked:

    PROJECT_ROOT=/path/to/rvc-web \\
    conda run -n rvc pytest backend/tests/test_training_integration.py \\
      -v -s -m integration --timeout=1800

What this test proves:
  1. Full subprocess chain executes on macOS Apple Silicon without crashing.
  2. Log streaming architecture works end-to-end (messages flow via WebSocket).
  3. ``assets/weights/rvc_finetune_active.pth`` is produced on disk.
  4. Profile status is updated to ``"trained"`` in SQLite.

This test retires the "MPS training stability" risk from the M001 roadmap.
"""

import io
import json
import os
import time
from pathlib import Path

import aiosqlite
import numpy as np
import pytest
from scipy.io import wavfile
from starlette.testclient import TestClient

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Auto-skip if RVC_ROOT is not set in environment
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def require_rvc_root():
    """Skip every test in this module if RVC_ROOT is not set."""
    rvc_root = os.environ.get("RVC_ROOT")
    if not rvc_root:
        pytest.skip("RVC_ROOT environment variable not set — skipping integration tests")
    if not os.path.isdir(rvc_root):
        pytest.skip(f"RVC_ROOT does not exist: {rvc_root}")


# ---------------------------------------------------------------------------
# WAV fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def real_wav_path(tmp_path):
    """Generate a 5-second mono 48kHz 440Hz sine-wave WAV, return its path.

    Uses numpy + scipy.io.wavfile so the file is a real PCM WAV that
    RVC's preprocess.py can load without complaint.
    """
    sample_rate = 48000
    duration_s = 5
    freq_hz = 440.0

    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # 16-bit PCM sine wave at modest amplitude to avoid clipping
    amplitude = 16000
    samples = (np.sin(2 * np.pi * freq_hz * t) * amplitude).astype(np.int16)

    wav_path = tmp_path / "test_voice.wav"
    wavfile.write(str(wav_path), sample_rate, samples)
    return str(wav_path)


# ---------------------------------------------------------------------------
# Integration-specific DB path (use the real DB, not a test-isolated one)
# ---------------------------------------------------------------------------

def _get_real_db_path() -> str:
    """Return the real production DB path for integration assertions.

    Integration tests write to the real DB (data/rvc.db) because they use
    the production app with real subprocesses — isolation would prevent the
    training pipeline from persisting the "trained" status.
    """
    from backend.app.db import DB_PATH  # noqa: PLC0415
    return DB_PATH


# ---------------------------------------------------------------------------
# Fixtures: real app client with RVC_ROOT set
# ---------------------------------------------------------------------------

@pytest.fixture()
def real_app_client():
    """TestClient wrapping the real FastAPI app with RVC_ROOT set.

    Does NOT patch the DB or DATA_DIR — uses the production backend/data/
    paths so the training pipeline can write to the real SQLite DB and the
    profile's sample file is stored on a persistent path that the subprocess
    can read.
    """
    from backend.app.main import app  # noqa: PLC0415

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.fixture()
def trained_profile(real_app_client, real_wav_path):
    """Create a real profile with a 5-second WAV and return its profile_id.

    Cleanup: profile is deleted from DB and disk after the test completes,
    leaving the DB in a clean state. The .pth and FAISS index files in
    RVC_ROOT/assets/weights/ and RVC_ROOT/logs/ are NOT deleted so they
    can be inspected post-run.
    """
    # Upload the real WAV and create the profile
    with open(real_wav_path, "rb") as f:
        resp = real_app_client.post(
            "/api/profiles",
            data={"name": "Integration Test Voice"},
            files={"file": ("test_voice.wav", f, "audio/wav")},
        )
    assert resp.status_code == 200, f"Profile creation failed: {resp.text}"
    profile = resp.json()
    profile_id = profile["id"]

    yield profile_id

    # Cleanup: remove the profile from DB and disk
    real_app_client.delete(f"/api/profiles/{profile_id}")


# ---------------------------------------------------------------------------
# Main integration test
# ---------------------------------------------------------------------------

def test_full_training_pipeline(real_app_client, trained_profile):
    """Run one epoch of RVC training and verify all observable outputs.

    Steps:
      1. POST /api/training/start with epochs=1, save_every=1
      2. Assert 200 + job_id returned
      3. Connect WebSocket /ws/training/<profile_id>
      4. Collect all messages until type=="done" or type=="error"
      5. Assert: at least one log message, final message is "done"
      6. Assert: assets/weights/rvc_finetune_active.pth exists on disk
      7. Assert: profile status is "trained" in SQLite

    Timeout: 30 minutes (training 1 epoch on CPU may take 10-25 minutes).
    """
    profile_id = trained_profile
    rvc_root = os.environ["RVC_ROOT"]
    timeout_s = 30 * 60  # 30-minute overall timeout

    # ------------------------------------------------------------------
    # Step 1: POST /api/training/start
    # ------------------------------------------------------------------
    start_resp = real_app_client.post(
        "/api/training/start",
        json={"profile_id": profile_id, "epochs": 1, "save_every": 1},
    )
    assert start_resp.status_code == 200, (
        f"Expected 200 from /api/training/start, got {start_resp.status_code}: "
        f"{start_resp.text}"
    )
    start_data = start_resp.json()
    assert "job_id" in start_data, f"Expected job_id in response: {start_data}"
    job_id = start_data["job_id"]
    print(f"\n[integration] Training started: job_id={job_id}, profile_id={profile_id}")

    # ------------------------------------------------------------------
    # Step 2: Connect WebSocket and collect messages
    # ------------------------------------------------------------------
    messages = []
    deadline = time.time() + timeout_s

    with real_app_client.websocket_connect(
        f"/ws/training/{profile_id}"
    ) as ws:
        while True:
            if time.time() > deadline:
                print(
                    f"[integration] TIMEOUT after {timeout_s}s "
                    f"({len(messages)} messages collected so far)"
                )
                # Print last few messages for diagnostics
                for m in messages[-5:]:
                    print(f"  last msg: {m}")
                break

            try:
                raw = ws.receive_text()
            except Exception as exc:
                print(f"[integration] WebSocket receive error: {exc}")
                break

            msg = json.loads(raw)
            messages.append(msg)

            # Print every message for -s (no-capture) visibility
            print(
                f"[integration] [{msg.get('phase','?')}] "
                f"[{msg.get('type','?')}] {msg.get('message','')[:120]}"
            )

            if msg.get("type") in ("done", "error"):
                break

    print(f"\n[integration] Collected {len(messages)} WebSocket messages total")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    # At least one log message received
    assert any(m.get("type") == "log" for m in messages), (
        "Expected at least one 'log' message in WebSocket stream. "
        f"Message types seen: {[m.get('type') for m in messages]}"
    )

    # Final message type
    assert messages, "No messages received from WebSocket"
    final_msg = messages[-1]
    assert final_msg.get("type") == "done", (
        f"Expected final message type='done' but got: {final_msg}. "
        "Check train.log for the training error:\n"
        f"  tail -50 {rvc_root}/logs/rvc_finetune_active/train.log"
    )

    # .pth file on disk
    pth_path = Path(rvc_root) / "assets" / "weights" / "rvc_finetune_active.pth"
    assert pth_path.exists(), (
        f"Expected model file at {pth_path} but it does not exist. "
        "Training may have completed but save_small_model() failed."
    )
    pth_size = pth_path.stat().st_size
    print(f"[integration] .pth file: {pth_path} ({pth_size:,} bytes)")
    assert pth_size > 0, f".pth file exists but is empty: {pth_path}"

    # DB status = "trained"
    import asyncio  # noqa: PLC0415

    async def _check_db_status() -> str:
        db_path = _get_real_db_path()
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT status FROM profiles WHERE id = ?", (profile_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return "profile_not_found"
        return row[0]

    db_status = asyncio.get_event_loop().run_until_complete(_check_db_status())
    assert db_status == "trained", (
        f"Expected profile status='trained' in DB but got '{db_status}'. "
        "The training pipeline completed but DB update may have failed."
    )

    print(
        f"\n[integration] ✓ Training complete: "
        f".pth={pth_path.name} ({pth_size:,}b), DB status={db_status}"
    )
