"""Voice profile CRUD + audio preprocessing router.

Endpoints
---------
POST   /api/profiles                   — upload audio + create profile
GET    /api/profiles                   — list all profiles
GET    /api/profiles/{id}              — get single profile
DELETE /api/profiles/{id}              — delete profile + files
POST   /api/profiles/{id}/preprocess   — run noise removal on selected segment
GET    /api/profiles/{id}/audio        — stream original audio file
GET    /api/profiles/{id}/audio/clean  — stream preprocessed (cleaned) audio
"""

import asyncio
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Optional

import aiofiles
from fastapi import APIRouter, Form, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi import UploadFile
from pydantic import BaseModel

from backend.app.audio_preprocessing import get_duration
from backend.app.db import get_db

logger = logging.getLogger("rvc_web.profiles")

router = APIRouter(prefix="/api/profiles", tags=["profiles"])

# Maximum upload size: ~200 MB is generous for 30 min audio.
# Duration enforcement (30 min) is done after write via get_duration().
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Audio duration constraints (seconds)
MAX_DURATION_SEC = 30 * 60   # 30 minutes


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ProfileOut(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    sample_path: str
    batch_size: int = 8
    audio_duration: Optional[float] = None
    preprocessed_path: Optional[str] = None


class PreprocessRequest(BaseModel):
    start_sec: float = 0.0
    end_sec: Optional[float] = None


class PreprocessResponse(BaseModel):
    ok: bool
    message: str
    preprocessed_path: Optional[str] = None
    duration_sec: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _samples_root() -> str:
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, "samples")
    return "data/samples"


def _row_to_out(row) -> ProfileOut:
    return ProfileOut(
        id=row["id"],
        name=row["name"],
        status=row["status"],
        created_at=row["created_at"],
        sample_path=row["sample_path"] or "",
        batch_size=row["batch_size"] if row["batch_size"] is not None else 8,
        audio_duration=row["audio_duration"],
        preprocessed_path=row["preprocessed_path"],
    )


# ---------------------------------------------------------------------------
# POST /api/profiles  — upload
# ---------------------------------------------------------------------------

@router.post("", status_code=200, response_model=ProfileOut)
async def create_profile(
    name: str = Form(...),
    file: UploadFile = ...,
) -> ProfileOut:
    """Upload an audio file and create a voice profile record.

    Enforces:
      - 200 MiB upload cap (byte-level, before duration check)
      - 30-minute duration cap (after write — checked via torchaudio header)

    On success the profile is stored with status='untrained'.
    """
    profile_id = uuid.uuid4().hex
    samples_root = _samples_root()
    dest_dir = os.path.join(samples_root, profile_id)
    os.makedirs(dest_dir, exist_ok=True)

    original_filename = file.filename or "upload"
    dest_path = os.path.join(dest_dir, original_filename)

    bytes_written = 0
    try:
        async with aiofiles.open(dest_path, "wb") as out_f:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    await out_f.flush()
                    shutil.rmtree(dest_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds 200 MiB limit",
                    )
                await out_f.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"File write error: {exc}") from exc

    # Duration check — run in executor so we don't block the event loop
    loop = asyncio.get_event_loop()
    duration = await loop.run_in_executor(None, get_duration, dest_path)

    if duration is not None and duration > MAX_DURATION_SEC:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail=f"Audio is {duration/60:.1f} min — max allowed is 30 min",
        )

    created_at = datetime.now(timezone.utc).isoformat()

    async with get_db() as db:
        await db.execute(
            """INSERT INTO profiles
               (id, name, status, sample_path, batch_size, audio_duration, preprocessed_path, created_at)
               VALUES (?, ?, 'untrained', ?, 8, ?, NULL, ?)""",
            (profile_id, name, dest_path, duration, created_at),
        )
        await db.commit()

    logger.info("profile created id=%s name=%s duration=%.1fs", profile_id, name, duration or 0)

    return ProfileOut(
        id=profile_id,
        name=name,
        status="untrained",
        created_at=created_at,
        sample_path=dest_path,
        batch_size=8,
        audio_duration=duration,
        preprocessed_path=None,
    )


# ---------------------------------------------------------------------------
# GET /api/profiles
# ---------------------------------------------------------------------------

@router.get("", response_model=list[ProfileOut])
async def list_profiles() -> list[ProfileOut]:
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT id, name, status, created_at, sample_path,
                      batch_size, audio_duration, preprocessed_path
               FROM profiles ORDER BY created_at DESC"""
        )
        rows = await cursor.fetchall()
    return [_row_to_out(r) for r in rows]


# ---------------------------------------------------------------------------
# GET /api/profiles/{id}
# ---------------------------------------------------------------------------

@router.get("/{profile_id}", response_model=ProfileOut)
async def get_profile(profile_id: str) -> ProfileOut:
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT id, name, status, created_at, sample_path,
                      batch_size, audio_duration, preprocessed_path
               FROM profiles WHERE id = ?""",
            (profile_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
    return _row_to_out(row)


# ---------------------------------------------------------------------------
# DELETE /api/profiles/{id}
# ---------------------------------------------------------------------------

@router.delete("/{profile_id}", status_code=204)
async def delete_profile(profile_id: str) -> Response:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM profiles WHERE id = ?", (profile_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
        await db.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        await db.commit()

    sample_dir = os.path.join(_samples_root(), profile_id)
    shutil.rmtree(sample_dir, ignore_errors=True)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# POST /api/profiles/{id}/preprocess
# ---------------------------------------------------------------------------

@router.post("/{profile_id}/preprocess", response_model=PreprocessResponse)
async def preprocess_audio(
    profile_id: str,
    req: PreprocessRequest,
) -> PreprocessResponse:
    """Run noise removal on the selected segment of the profile's audio file.

    The cleaned audio is saved as ``cleaned.wav`` in the profile's sample dir.
    The profile's ``preprocessed_path`` DB column is updated on success.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT sample_path, audio_duration FROM profiles WHERE id = ?",
            (profile_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")

    input_path = row["sample_path"]
    if not input_path or not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail="No audio file on disk for this profile")

    # Validate selected segment length
    total_dur = row["audio_duration"] or 0.0
    start = req.start_sec
    end   = req.end_sec if req.end_sec is not None else total_dur
    seg_len = end - start

    MIN_SEG = 10 * 60   # 10 minutes
    MAX_SEG = 15 * 60   # 15 minutes

    if seg_len < MIN_SEG:
        return PreprocessResponse(
            ok=False,
            message=f"Selected segment is {seg_len/60:.1f} min — minimum is 10 min",
        )
    if seg_len > MAX_SEG:
        return PreprocessResponse(
            ok=False,
            message=f"Selected segment is {seg_len/60:.1f} min — maximum is 15 min",
        )

    # Output path
    out_dir  = os.path.dirname(input_path)
    out_path = os.path.join(out_dir, "cleaned.wav")

    # Run noise removal in thread pool so we don't block the event loop
    from backend.app.audio_preprocessing import remove_noise  # noqa: PLC0415

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: remove_noise(input_path, out_path, start, end),
        )
    except RuntimeError as exc:
        return PreprocessResponse(ok=False, message=str(exc))

    cleaned_dur = await loop.run_in_executor(None, get_duration, out_path)

    async with get_db() as db:
        await db.execute(
            "UPDATE profiles SET preprocessed_path = ? WHERE id = ?",
            (out_path, profile_id),
        )
        await db.commit()

    return PreprocessResponse(
        ok=True,
        message="Noise removal complete",
        preprocessed_path=out_path,
        duration_sec=cleaned_dur,
    )


# ---------------------------------------------------------------------------
# GET /api/profiles/{id}/audio  — stream original audio
# ---------------------------------------------------------------------------

@router.get("/{profile_id}/audio")
async def stream_audio(profile_id: str):
    """Stream the original uploaded audio file."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT sample_path FROM profiles WHERE id = ?", (profile_id,)
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    path = row["sample_path"]
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    return FileResponse(path, media_type="audio/mpeg", filename=os.path.basename(path))


# ---------------------------------------------------------------------------
# GET /api/profiles/{id}/audio/clean  — stream cleaned audio
# ---------------------------------------------------------------------------

@router.get("/{profile_id}/audio/clean")
async def stream_clean_audio(profile_id: str):
    """Stream the noise-removed audio file."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT preprocessed_path FROM profiles WHERE id = ?", (profile_id,)
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    path = row["preprocessed_path"]
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Cleaned audio not found — run preprocessing first")
    return FileResponse(path, media_type="audio/wav", filename="cleaned.wav")
