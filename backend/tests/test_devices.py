"""Contract tests for the audio devices endpoint.

These tests will FAIL until T04 implements the route — that is intentional.
They encode the exact API shape the slice must satisfy.

Schema contract (from S01-PLAN.md boundary):
  GET /api/devices
      → 200  [{id: int, name: str, is_input: bool, is_output: bool}, ...]
      → list is non-empty on any real machine
      → at least one entry has name containing "BlackHole" (dev machine)
"""

import pytest
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# GET /api/devices
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_devices_returns_200(client):
    """GET /api/devices returns HTTP 200."""
    resp = await client.get("/api/devices")
    assert resp.status_code == 200, resp.text


@pytest.mark.anyio
async def test_devices_returns_non_empty_list(client):
    """GET /api/devices returns a non-empty list (every machine has at least one device)."""
    resp = await client.get("/api/devices")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    assert len(data) > 0, "Expected at least one audio device"


@pytest.mark.anyio
async def test_devices_entries_have_required_keys(client):
    """Every device entry has id, name, is_input, is_output keys."""
    resp = await client.get("/api/devices")
    assert resp.status_code == 200, resp.text
    devices = resp.json()
    required_keys = {"id", "name", "is_input", "is_output"}
    for device in devices:
        missing = required_keys - set(device.keys())
        assert not missing, f"Device entry missing keys {missing}: {device}"


@pytest.mark.anyio
async def test_devices_id_is_int(client):
    """Every device entry's id field is an integer."""
    resp = await client.get("/api/devices")
    assert resp.status_code == 200, resp.text
    for device in resp.json():
        assert isinstance(device["id"], int), f"id not int: {device}"


@pytest.mark.anyio
async def test_devices_name_is_str(client):
    """Every device entry's name field is a non-empty string."""
    resp = await client.get("/api/devices")
    assert resp.status_code == 200, resp.text
    for device in resp.json():
        assert isinstance(device["name"], str), f"name not str: {device}"
        assert device["name"], f"name is empty: {device}"


@pytest.mark.anyio
async def test_devices_is_input_is_output_are_bool(client):
    """is_input and is_output are boolean values."""
    resp = await client.get("/api/devices")
    assert resp.status_code == 200, resp.text
    for device in resp.json():
        assert isinstance(device["is_input"], bool), f"is_input not bool: {device}"
        assert isinstance(device["is_output"], bool), f"is_output not bool: {device}"


@pytest.mark.anyio
async def test_devices_includes_blackhole(client):
    """Response includes a BlackHole device (required on dev machine)."""
    resp = await client.get("/api/devices")
    assert resp.status_code == 200, resp.text
    names = [d["name"] for d in resp.json()]
    blackhole_found = any("BlackHole" in name for name in names)
    assert blackhole_found, f"BlackHole not found in device list: {names}"
