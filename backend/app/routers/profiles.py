"""Voice profile CRUD router.

Endpoints:
  POST   /api/profiles          — upload audio + create profile row
  GET    /api/profiles          — list all profiles
  GET    /api/profiles/{id}     — get profile by id
  DELETE /api/profiles/{id}     — delete profile row + sample directory
"""

import logging
import os
import shutil
import uuid
from datetime import datetime, timezone

import aiofiles
from fastapi import APIRouter, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

from backend.app.db import get_db

logger = logging.getLogger("rvc_web.profiles")

router = APIRouter(prefix="/api/profiles", tags=["profiles"])

# 200 MiB upload limit
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class ProfileOut(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    sample_path: str


def _samples_root() -> str:
    """Return the base directory for uploaded audio files.

    Reads DATA_DIR env var so tests can redirect to a tmp_path.
    Falls back to ``data/samples`` (repo-relative).
    """
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, "samples")
    return "data/samples"


@router.post("", status_code=200, response_model=ProfileOut)
async def create_profile(
    name: str = Form(...),
    file: UploadFile = ...,
) -> ProfileOut:
    """Upload an audio file and create a voice profile record.

    Args:
        name: Human-readable display name for the voice profile.
        file: Audio file upload (WAV, MP3, etc.).

    Returns:
        The newly created ProfileOut with status ``untrained``.

    Raises:
        HTTPException(413): If the uploaded file exceeds 200 MiB.
        HTTPException(500): If the file cannot be written to disk.
    """
    profile_id = uuid.uuid4().hex

    # Stream file to disk with size enforcement
    samples_root = _samples_root()
    dest_dir = os.path.join(samples_root, profile_id)
    os.makedirs(dest_dir, exist_ok=True)

    original_filename = file.filename or "upload"
    dest_path = os.path.join(dest_dir, original_filename)

    bytes_written = 0
    chunk_size = 64 * 1024  # 64 KiB chunks

    try:
        async with aiofiles.open(dest_path, "wb") as out_file:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    # Clean up partial write before raising
                    await out_file.flush()
                    shutil.rmtree(dest_dir, ignore_errors=True)
                    logger.warning(
                        "Upload rejected: size limit exceeded",
                        extra={"profile_id": profile_id, "limit": MAX_UPLOAD_BYTES},
                    )
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_BYTES} bytes",
                    )
                await out_file.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        logger.error(
            "File write failed",
            extra={"profile_id": profile_id, "error": str(exc)},
        )
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"File write error: {exc}") from exc

    created_at = datetime.now(timezone.utc).isoformat()

    async with get_db() as db:
        await db.execute(
            "INSERT INTO profiles (id, name, status, sample_path, created_at) VALUES (?, ?, ?, ?, ?)",
            (profile_id, name, "untrained", dest_path, created_at),
        )
        await db.commit()

    logger.info(
        "Profile created",
        extra={"profile_id": profile_id, "name": name, "sample_path": dest_path},
    )

    return ProfileOut(
        id=profile_id,
        name=name,
        status="untrained",
        created_at=created_at,
        sample_path=dest_path,
    )


@router.get("", response_model=list[ProfileOut])
async def list_profiles() -> list[ProfileOut]:
    """Return all voice profiles."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, name, status, created_at, sample_path FROM profiles ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()

    return [
        ProfileOut(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            created_at=row["created_at"],
            sample_path=row["sample_path"] or "",
        )
        for row in rows
    ]


@router.get("/{profile_id}", response_model=ProfileOut)
async def get_profile(profile_id: str) -> ProfileOut:
    """Return a single voice profile by id.

    Raises:
        HTTPException(404): If no profile with this id exists.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, name, status, created_at, sample_path FROM profiles WHERE id = ?",
            (profile_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        logger.info("Profile not found", extra={"profile_id": profile_id, "op": "get"})
        raise HTTPException(
            status_code=404, detail=f"Profile not found: {profile_id}"
        )

    return ProfileOut(
        id=row["id"],
        name=row["name"],
        status=row["status"],
        created_at=row["created_at"],
        sample_path=row["sample_path"] or "",
    )


@router.delete("/{profile_id}", status_code=204)
async def delete_profile(profile_id: str) -> Response:
    """Delete a voice profile and remove its sample directory.

    Raises:
        HTTPException(404): If no profile with this id exists.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, sample_path FROM profiles WHERE id = ?", (profile_id,)
        )
        row = await cursor.fetchone()

        if row is None:
            logger.info(
                "Profile not found", extra={"profile_id": profile_id, "op": "delete"}
            )
            raise HTTPException(
                status_code=404, detail=f"Profile not found: {profile_id}"
            )

        await db.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        await db.commit()

    # Remove sample directory after DB commit to keep storage consistent
    samples_root = _samples_root()
    sample_dir = os.path.join(samples_root, profile_id)
    shutil.rmtree(sample_dir, ignore_errors=True)

    logger.info("Profile deleted", extra={"profile_id": profile_id})
    return Response(status_code=204)
