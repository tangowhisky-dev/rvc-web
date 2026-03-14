"""Contract tests for voice profile CRUD endpoints.

These tests will FAIL until T03 implements the routes — that is intentional.
They encode the exact API shape the slice must satisfy.

Schema contract (from S01-PLAN.md boundary):
  POST   /api/profiles  multipart: name (str) + file (UploadFile)
         → 200  {id, name, status:"untrained", created_at, sample_path}
  GET    /api/profiles
         → 200  [{id, name, status, created_at, sample_path}, ...]
  GET    /api/profiles/{id}
         → 200  {id, name, status, created_at, sample_path}
         → 404  if unknown id
  DELETE /api/profiles/{id}
         → 204  if found (removes DB row + sample dir)
         → 404  if unknown id
  POST   /api/profiles without file
         → 422  validation error
"""

import io
import pytest
from httpx import AsyncClient, ASGITransport


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


# ---------------------------------------------------------------------------
# POST /api/profiles — create
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_create_profile_returns_200_with_required_fields(client):
    """POST creates a profile and returns required JSON fields."""
    resp = await client.post(
        "/api/profiles",
        files={"file": ("voice.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"name": "test_voice"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "test_voice"
    assert body["status"] == "untrained"
    assert "id" in body
    assert "created_at" in body
    assert "sample_path" in body


@pytest.mark.anyio
async def test_create_profile_file_exists_on_disk(client):
    """POST saves the uploaded file to data/samples/<id>/."""
    import os  # noqa: PLC0415

    resp = await client.post(
        "/api/profiles",
        files={"file": ("voice.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"name": "disk_test"},
    )
    assert resp.status_code == 200, resp.text
    profile_id = resp.json()["id"]
    sample_path = resp.json()["sample_path"]
    # File must physically exist (returned path is absolute from conftest tmp_path)
    assert os.path.exists(sample_path), f"Expected file on disk at {sample_path}"
    _ = profile_id  # referenced in error message above


@pytest.mark.anyio
async def test_create_profile_without_file_returns_422(client):
    """POST without a file upload returns 422 Unprocessable Entity."""
    resp = await client.post("/api/profiles", data={"name": "no_file"})
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# GET /api/profiles — list
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_list_profiles_returns_array(client):
    """GET /api/profiles returns a JSON array."""
    resp = await client.get("/api/profiles")
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json(), list)


@pytest.mark.anyio
async def test_list_profiles_includes_created_profile(client):
    """A freshly created profile appears in the list response."""
    # Create
    create = await client.post(
        "/api/profiles",
        files={"file": ("v.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"name": "list_test"},
    )
    assert create.status_code == 200, create.text
    created_id = create.json()["id"]

    # List
    resp = await client.get("/api/profiles")
    assert resp.status_code == 200, resp.text
    ids = [p["id"] for p in resp.json()]
    assert created_id in ids, f"{created_id} not found in {ids}"


# ---------------------------------------------------------------------------
# GET /api/profiles/{id} — get by id
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_get_profile_by_id_returns_profile(client):
    """GET /api/profiles/{id} returns the profile with correct fields."""
    create = await client.post(
        "/api/profiles",
        files={"file": ("v.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"name": "get_by_id"},
    )
    assert create.status_code == 200, create.text
    profile_id = create.json()["id"]

    resp = await client.get(f"/api/profiles/{profile_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == profile_id
    assert body["name"] == "get_by_id"
    assert "status" in body
    assert "created_at" in body
    assert "sample_path" in body


@pytest.mark.anyio
async def test_get_profile_unknown_id_returns_404(client):
    """GET /api/profiles/{id} with an unknown id returns 404."""
    resp = await client.get("/api/profiles/doesnotexist000")
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# DELETE /api/profiles/{id}
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_delete_profile_returns_204(client):
    """DELETE /api/profiles/{id} of an existing profile returns 204."""
    create = await client.post(
        "/api/profiles",
        files={"file": ("v.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"name": "delete_me"},
    )
    assert create.status_code == 200, create.text
    profile_id = create.json()["id"]

    resp = await client.delete(f"/api/profiles/{profile_id}")
    assert resp.status_code == 204, resp.text


@pytest.mark.anyio
async def test_delete_profile_removes_from_list(client):
    """After DELETE, the profile no longer appears in the list."""
    create = await client.post(
        "/api/profiles",
        files={"file": ("v.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"name": "delete_list_check"},
    )
    assert create.status_code == 200, create.text
    profile_id = create.json()["id"]

    await client.delete(f"/api/profiles/{profile_id}")

    resp = await client.get("/api/profiles")
    ids = [p["id"] for p in resp.json()]
    assert profile_id not in ids, f"Deleted profile {profile_id} still in list"


@pytest.mark.anyio
async def test_delete_unknown_profile_returns_404(client):
    """DELETE /api/profiles/{id} with an unknown id returns 404."""
    resp = await client.delete("/api/profiles/doesnotexist000")
    assert resp.status_code == 404, resp.text
