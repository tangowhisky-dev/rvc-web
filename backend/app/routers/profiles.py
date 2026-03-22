"""Voice profile CRUD + audio preprocessing router.

Directory layout (per profile):
    data/profiles/{id}/
        audio/
            sample.wav         — clipped upload (always present after create)
            cleaned.wav        — noise-removed (optional, after preprocess)
        model.pth              — generator weights (present after training)
        model.index            — FAISS index (present after training)

Older profiles created before this layout used data/samples/{id}/ — the
_profile_dir() helper falls back to that legacy path transparently.

Endpoints
---------
POST   /api/profiles                   — upload audio + create profile
GET    /api/profiles                   — list all profiles
GET    /api/profiles/{id}              — get single profile
DELETE /api/profiles/{id}              — delete profile + files
POST   /api/profiles/{id}/preprocess   — run noise removal on selected segment
GET    /api/profiles/{id}/audio        — stream original audio file
GET    /api/profiles/{id}/audio/clean  — stream preprocessed (cleaned) audio
GET    /api/profiles/{id}/health       — async file-existence check
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
    profile_dir: Optional[str] = None
    model_path: Optional[str] = None        # inference-ready (~54 MB)
    checkpoint_path: Optional[str] = None  # resume checkpoint (~431 MB)
    index_path: Optional[str] = None
    total_epochs_trained: int = 0


class PreprocessResponse(BaseModel):
    ok: bool
    message: str
    preprocessed_path: Optional[str] = None
    duration_sec: Optional[float] = None


class HealthStatus(BaseModel):
    """Per-profile file-existence health check result."""
    profile_id: str
    audio_ok: bool            # sample.wav exists
    preprocessed_ok: bool     # cleaned.wav exists (only relevant if preprocessed_path set)
    model_ok: bool            # model.pth exists
    index_ok: bool            # model.index exists
    can_train: bool           # audio_ok (no model required to start training)
    can_infer: bool           # model_ok AND index_ok
    errors: list[str]         # human-readable descriptions of what's missing


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _profiles_root() -> str:
    """Return the root directory for per-profile storage."""
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, "profiles")
    return "data/profiles"


def _samples_root() -> str:
    """Legacy samples root — kept for backward-compat path resolution."""
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, "samples")
    return "data/samples"


def _profile_dir(profile_id: str, stored_dir: Optional[str] = None) -> str:
    """Return the canonical profile directory for a profile.

    Prefers the stored profile_dir column if set (new layout).
    Falls back to the legacy data/samples/{id}/ path for old profiles.
    """
    if stored_dir:
        return stored_dir
    # Legacy fallback
    return os.path.join(_samples_root(), profile_id)


def _audio_dir(pdir: str) -> str:
    return os.path.join(pdir, "audio")


def _resolve_sample_path(row) -> str:
    """Return the best available sample_path for a DB row.

    New profiles: profile_dir/audio/sample.wav
    Legacy profiles: stored sample_path column
    """
    stored = row["sample_path"] or ""
    pdir = row["profile_dir"]
    if pdir:
        candidate = os.path.join(pdir, "audio", "sample.wav")
        if os.path.exists(candidate):
            return candidate
    return stored


def _resolve_model_path(row) -> Optional[str]:
    """Return the inference-ready model path for this profile.

    New layout: profile_dir/model_infer.pth  (fp16 weights, ~54 MB)
    Legacy layout: profile_dir/model.pth
    Fallback: stored model_path from DB
    """
    pdir = row["profile_dir"]
    stored = row["model_path"]
    if pdir:
        # New layout: model_infer.pth (inference-ready fp16 weights)
        candidate = os.path.join(pdir, "model_infer.pth")
        if os.path.exists(candidate):
            return candidate
        # Legacy layout: model.pth
        candidate_legacy = os.path.join(pdir, "model.pth")
        if os.path.exists(candidate_legacy):
            return candidate_legacy
    return stored or None


def _resolve_checkpoint_path(row) -> Optional[str]:
    """Return the resume checkpoint path (G_latest.pth) for this profile."""
    pdir = row["profile_dir"]
    keys = row.keys() if hasattr(row, "keys") else []
    stored = row["checkpoint_path"] if "checkpoint_path" in keys else None
    if pdir:
        candidate = os.path.join(pdir, "checkpoints", "G_latest.pth")
        if os.path.exists(candidate):
            return candidate
    return stored or None


def _resolve_index_path(row) -> Optional[str]:
    pdir = row["profile_dir"]
    stored = row["index_path"]
    if pdir:
        candidate = os.path.join(pdir, "model.index")
        if os.path.exists(candidate):
            return candidate
    return stored or None


def _row_to_out(row) -> ProfileOut:
    pdir = row["profile_dir"]
    keys = row.keys() if hasattr(row, "keys") else []
    return ProfileOut(
        id=row["id"],
        name=row["name"],
        status=row["status"],
        created_at=row["created_at"],
        sample_path=_resolve_sample_path(row),
        batch_size=row["batch_size"] if row["batch_size"] is not None else 8,
        audio_duration=row["audio_duration"],
        preprocessed_path=row["preprocessed_path"],
        profile_dir=pdir,
        model_path=_resolve_model_path(row),
        checkpoint_path=_resolve_checkpoint_path(row),
        index_path=_resolve_index_path(row),
        total_epochs_trained=int(row["total_epochs_trained"] or 0) if "total_epochs_trained" in keys else 0,
    )


# ---------------------------------------------------------------------------
# POST /api/profiles  — upload
# ---------------------------------------------------------------------------

@router.post("", status_code=200, response_model=ProfileOut)
async def create_profile(
    name: str = Form(...),
    file: UploadFile = ...,
    seg_start: float = Form(0.0),
    seg_end: Optional[float] = Form(None),
) -> ProfileOut:
    """Upload an audio file, clip to the selected segment, and create a profile.

    Files land in data/profiles/{id}/audio/sample.wav.
    profile_dir is stored in the DB so future code always knows where to find
    model.pth and model.index for this profile.
    """
    profile_id = uuid.uuid4().hex
    pdir = os.path.join(_profiles_root(), profile_id)
    audio_dir = _audio_dir(pdir)
    os.makedirs(audio_dir, exist_ok=True)

    original_filename = file.filename or "upload"
    # Write upload to a temp file in the audio dir before clipping
    raw_path = os.path.join(audio_dir, f"_upload_{original_filename}")

    bytes_written = 0
    try:
        async with aiofiles.open(raw_path, "wb") as out_f:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    await out_f.flush()
                    shutil.rmtree(pdir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds 200 MiB limit",
                    )
                await out_f.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        shutil.rmtree(pdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"File write error: {exc}") from exc

    # Duration check
    loop = asyncio.get_event_loop()
    duration = await loop.run_in_executor(None, get_duration, raw_path)

    if duration is not None and duration > MAX_DURATION_SEC:
        shutil.rmtree(pdir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail=f"Audio is {duration/60:.1f} min — max allowed is 30 min",
        )

    # Clip to selected segment and save as sample.wav
    import soundfile as sf  # noqa: PLC0415

    clip_path = os.path.join(audio_dir, "sample.wav")
    end_for_clip = seg_end if seg_end is not None else duration
    clip_dur: Optional[float] = None

    try:
        raw, sr = sf.read(raw_path, always_2d=True, dtype="float32")
        s_frame = int(round((seg_start or 0.0) * sr))
        e_frame = int(round(end_for_clip * sr)) if end_for_clip is not None else raw.shape[0]
        s_frame = max(0, min(s_frame, raw.shape[0]))
        e_frame = max(s_frame, min(e_frame, raw.shape[0]))
        clipped = raw[s_frame:e_frame, :]
        sf.write(clip_path, clipped, sr, subtype="PCM_16")
        clip_dur = round(clipped.shape[0] / sr, 3)
        sample_path_final = clip_path
    except Exception as exc:
        logger.warning("clip failed (%s) — keeping original as sample.wav: %s", raw_path, exc)
        shutil.copy2(raw_path, clip_path)
        sample_path_final = clip_path
        clip_dur = duration

    # Remove the raw upload temp file
    try:
        os.remove(raw_path)
    except OSError:
        pass

    created_at = datetime.now(timezone.utc).isoformat()

    async with get_db() as db:
        await db.execute(
            """INSERT INTO profiles
               (id, name, status, sample_path, batch_size, audio_duration,
                preprocessed_path, profile_dir, created_at)
               VALUES (?, ?, 'untrained', ?, 8, ?, NULL, ?, ?)""",
            (profile_id, name, sample_path_final, clip_dur, pdir, created_at),
        )
        await db.commit()

    logger.info("profile created id=%s name=%s dir=%s dur=%.1fs",
                profile_id, name, pdir, clip_dur or 0)

    return ProfileOut(
        id=profile_id,
        name=name,
        status="untrained",
        created_at=created_at,
        sample_path=sample_path_final,
        batch_size=8,
        audio_duration=clip_dur,
        preprocessed_path=None,
        profile_dir=pdir,
        model_path=None,
        index_path=None,
    )


# ---------------------------------------------------------------------------
# GET /api/profiles
# ---------------------------------------------------------------------------

@router.get("", response_model=list[ProfileOut])
async def list_profiles() -> list[ProfileOut]:
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT id, name, status, created_at, sample_path,
                      batch_size, audio_duration, preprocessed_path,
                      profile_dir, model_path, index_path
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
                      batch_size, audio_duration, preprocessed_path,
                      profile_dir, model_path, index_path
               FROM profiles WHERE id = ?""",
            (profile_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
    return _row_to_out(row)


# ---------------------------------------------------------------------------
# GET /api/profiles/{id}/health
# ---------------------------------------------------------------------------

@router.get("/{profile_id}/health", response_model=HealthStatus)
async def profile_health(profile_id: str) -> HealthStatus:
    """Async file-existence check for a profile.

    Returns can_train / can_infer flags and a list of human-readable errors
    for anything that's missing. Does not raise 4xx for missing files —
    returns can_train=False / can_infer=False with errors instead so the
    frontend can show per-card health status without crashing the list.
    """
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT id, status, sample_path, preprocessed_path,
                      profile_dir, model_path, checkpoint_path, index_path
               FROM profiles WHERE id = ?""",
            (profile_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")

    errors: list[str] = []

    # Audio check
    sample = _resolve_sample_path(row)
    audio_ok = bool(sample and os.path.exists(sample))
    if not audio_ok:
        errors.append("audio file missing (sample.wav not found on disk)")

    # Preprocessed check — only relevant if the DB says it was done
    preprocessed_ok = True
    if row["preprocessed_path"]:
        preprocessed_ok = os.path.exists(row["preprocessed_path"])
        if not preprocessed_ok:
            errors.append("cleaned audio referenced in DB but not found on disk")

    # Model check (inference-ready small model)
    model = _resolve_model_path(row)
    model_ok = bool(model and os.path.exists(model))
    if row["status"] == "trained" and not model_ok:
        errors.append("model_infer.pth missing — profile is marked trained but inference model not found")

    # Index check
    index = _resolve_index_path(row)
    index_ok = bool(index and os.path.exists(index))
    if row["status"] == "trained" and not index_ok:
        errors.append("model.index missing — profile is marked trained but index file not found")

    can_train = audio_ok
    can_infer = model_ok and index_ok

    return HealthStatus(
        profile_id=profile_id,
        audio_ok=audio_ok,
        preprocessed_ok=preprocessed_ok,
        model_ok=model_ok,
        index_ok=index_ok,
        can_train=can_train,
        can_infer=can_infer,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# DELETE /api/profiles/{id}
# ---------------------------------------------------------------------------

@router.delete("/{profile_id}", status_code=204)
async def delete_profile(profile_id: str) -> Response:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, profile_dir, sample_path FROM profiles WHERE id = ?", (profile_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
        await db.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        await db.commit()

    # Remove the profile directory (new layout)
    pdir = row["profile_dir"]
    if pdir and os.path.isdir(pdir):
        shutil.rmtree(pdir, ignore_errors=True)

    # Also clean up legacy samples dir if it exists
    legacy_dir = os.path.join(_samples_root(), profile_id)
    if os.path.isdir(legacy_dir):
        shutil.rmtree(legacy_dir, ignore_errors=True)

    return Response(status_code=204)


# ---------------------------------------------------------------------------
# POST /api/profiles/{id}/preprocess
# ---------------------------------------------------------------------------

@router.post("/{profile_id}/preprocess", response_model=PreprocessResponse)
async def preprocess_audio(profile_id: str) -> PreprocessResponse:
    """Run noise removal on the profile's audio sample.

    Writes cleaned.wav next to sample.wav in the profile's audio directory.
    """
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT sample_path, profile_dir FROM profiles WHERE id = ?""",
            (profile_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")

    input_path = _resolve_sample_path(row)
    if not input_path or not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail="No audio file on disk for this profile")

    # Output always goes to audio/cleaned.wav in the profile dir
    pdir = row["profile_dir"]
    if pdir:
        out_dir  = _audio_dir(pdir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.dirname(input_path)
    out_path = os.path.join(out_dir, "cleaned.wav")

    from backend.app.audio_preprocessing import remove_noise  # noqa: PLC0415

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: remove_noise(input_path, out_path),
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
            "SELECT sample_path, profile_dir FROM profiles WHERE id = ?", (profile_id,)
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    path = _resolve_sample_path(row)
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
