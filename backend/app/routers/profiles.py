"""Voice profile CRUD + audio file management router.

Directory layout (per profile):
    data/profiles/{id}/
        audio/
            voice1.wav          — clipped upload (cleaned in-place after noise removal)
            voice2.wav          — additional audio files
            ...
        checkpoints/
            G_latest.pth        — resume checkpoint
            D_latest.pth
        model_infer.pth         — inference-ready fp16 weights (~54 MB)
        model.index             — FAISS index

Audio files are tracked in the audio_files table (1 profile → N audio files).
Clean overwrites in-place — there is never a separate "raw" and "cleaned" copy.

Endpoints
---------
POST   /api/profiles                              — create profile + first audio file
GET    /api/profiles                              — list all profiles (with audio_files)
GET    /api/profiles/{id}                         — get single profile
DELETE /api/profiles/{id}                         — delete profile + entire profile_dir tree

POST   /api/profiles/{id}/audio                   — add audio file to existing profile
GET    /api/profiles/{id}/audio/{file_id}         — stream audio file
DELETE /api/profiles/{id}/audio/{file_id}         — delete audio file
POST   /api/profiles/{id}/audio/{file_id}/clean   — run noise removal in-place

GET    /api/profiles/{id}/health                  — file-existence health check
GET    /api/profiles/{id}/export                  — export profile as rvc-profile-v1.zip
POST   /api/profiles/import                       — import profile from rvc-profile-v1.zip
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Optional

import aiofiles
from fastapi import APIRouter, Form, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi import UploadFile
from pydantic import BaseModel

from backend.app.audio_preprocessing import (
    get_duration,
    measure_speech_rms,
    normalize_rms_inplace,
    preprocess_audio_inplace,
)
from backend.app.db import get_db

logger = logging.getLogger("rvc_web.profiles")

router = APIRouter(prefix="/api/profiles", tags=["profiles"])

# Maximum upload size: ~200 MB is generous for 30 min audio.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Audio duration constraints (seconds)
MAX_DURATION_SEC = 60 * 60  # 60 minutes


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AudioFileOut(BaseModel):
    id: str
    profile_id: str
    filename: str
    file_path: str
    duration: Optional[float] = None
    is_cleaned: bool = False
    created_at: str


class ProfileOut(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    # Legacy field — kept for backward compat (training router still reads it)
    sample_path: str
    batch_size: int = 8
    profile_dir: Optional[str] = None
    model_path: Optional[str] = None  # inference-ready (~54 MB)
    checkpoint_path: Optional[str] = None  # resume checkpoint (~431 MB)
    index_path: Optional[str] = None
    total_epochs_trained: int = 0
    needs_retraining: bool = False
    profile_rms: Optional[float] = None  # speech-weighted RMS of training data
    # Embedder model used for feature extraction.  Locked after first training.
    embedder: str = "spin-v2"
    # Vocoder decoder architecture.  Locked after first training.
    vocoder: str = "HiFi-GAN"
    # Pipeline engine: "rvc" (default) or "beatrice2". Locked at creation.
    pipeline: str = "rvc"
    # Best-epoch checkpoint info (None on old profiles that predate the feature).
    best_model_path: Optional[str] = None   # model_best.pth on disk, or None
    best_epoch: Optional[int] = None
    best_avg_gen_loss: Optional[float] = None
    audio_files: list[AudioFileOut] = []


class PreprocessResponse(BaseModel):
    ok: bool
    message: str
    duration_sec: Optional[float] = None


class HealthStatus(BaseModel):
    """Per-profile file-existence health check result."""

    profile_id: str
    audio_ok: bool  # at least one audio file exists
    model_ok: bool  # model_infer.pth exists
    index_ok: bool  # model.index exists
    can_train: bool  # audio_ok
    can_infer: bool  # model_ok AND index_ok
    errors: list[str]


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def _project_root() -> str:
    return os.environ.get(
        "PROJECT_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    )


def _profiles_root() -> str:
    return os.path.join(_project_root(), "data", "profiles")


def _samples_root() -> str:
    """Legacy samples root — kept for backward-compat path resolution."""
    return os.path.join(_project_root(), "data", "samples")


def _profile_dir(profile_id: str, stored_dir: Optional[str] = None) -> str:
    if stored_dir:
        return stored_dir
    return os.path.join(_samples_root(), profile_id)


def _audio_dir(pdir: str) -> str:
    return os.path.join(pdir, "audio")


def _dedup_filename(audio_dir: str, stem: str, ext: str = ".wav") -> str:
    """Return a filename that doesn't conflict with existing files in audio_dir.

    e.g. if "voice.wav" exists, returns "voice_1.wav", then "voice_2.wav", etc.
    """
    candidate = f"{stem}{ext}"
    if not os.path.exists(os.path.join(audio_dir, candidate)):
        return candidate
    n = 1
    while True:
        candidate = f"{stem}_{n}{ext}"
        if not os.path.exists(os.path.join(audio_dir, candidate)):
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _audio_file_row_to_out(row) -> AudioFileOut:
    return AudioFileOut(
        id=row["id"],
        profile_id=row["profile_id"],
        filename=row["filename"],
        file_path=row["file_path"],
        duration=row["duration"],
        is_cleaned=bool(row["is_cleaned"]),
        created_at=row["created_at"],
    )


async def _fetch_audio_files(db, profile_id: str) -> list[AudioFileOut]:
    cursor = await db.execute(
        "SELECT id, profile_id, filename, file_path, duration, is_cleaned, created_at "
        "FROM audio_files WHERE profile_id = ? ORDER BY created_at ASC",
        (profile_id,),
    )
    rows = await cursor.fetchall()
    return [_audio_file_row_to_out(r) for r in rows]


def _resolve_model_path(row) -> Optional[str]:
    pdir = row["profile_dir"]
    stored = row["model_path"]
    pipeline = (row["pipeline"] if "pipeline" in row.keys() else None) or "rvc"
    if pdir:
        if pipeline == "beatrice2":
            # Beatrice 2 checkpoint
            b2_ckpt = os.path.join(pdir, "beatrice2_out", "checkpoint_latest.pt.gz")
            if os.path.exists(b2_ckpt):
                return b2_ckpt
        else:
            candidate = os.path.join(pdir, "model_infer.pth")
            if os.path.exists(candidate):
                return candidate
            candidate_legacy = os.path.join(pdir, "model.pth")
            if os.path.exists(candidate_legacy):
                return candidate_legacy
    return stored or None


def _resolve_checkpoint_path(row) -> Optional[str]:
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


def _resolve_best_model_path(row) -> Optional[str]:
    """Return path to model_best.pth if it exists on disk, else None."""
    pdir = row["profile_dir"]
    if pdir:
        candidate = os.path.join(pdir, "model_best.pth")
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_sample_path_legacy(row) -> str:
    """Return best-available sample path for backward compat (training router)."""
    stored = row["sample_path"] or ""
    pdir = row["profile_dir"]
    if pdir:
        candidate = os.path.join(pdir, "audio", "sample.wav")
        if os.path.exists(candidate):
            return candidate
    return stored


async def _row_to_out(row, db) -> ProfileOut:
    pdir = row["profile_dir"]
    keys = row.keys() if hasattr(row, "keys") else []
    audio_files = await _fetch_audio_files(db, row["id"])
    return ProfileOut(
        id=row["id"],
        name=row["name"],
        status=row["status"],
        created_at=row["created_at"],
        sample_path=_resolve_sample_path_legacy(row),
        batch_size=row["batch_size"] if row["batch_size"] is not None else 8,
        profile_dir=pdir,
        model_path=_resolve_model_path(row),
        checkpoint_path=_resolve_checkpoint_path(row),
        index_path=_resolve_index_path(row),
        total_epochs_trained=int(row["total_epochs_trained"] or 0)
        if "total_epochs_trained" in keys
        else 0,
        needs_retraining=bool(row["needs_retraining"])
        if "needs_retraining" in keys
        else False,
        profile_rms=float(row["profile_rms"])
        if "profile_rms" in keys and row["profile_rms"] is not None
        else None,
        embedder=str(row["embedder"])
        if "embedder" in keys and row["embedder"]
        else "spin-v2",
        vocoder=str(row["vocoder"])
        if "vocoder" in keys and row["vocoder"]
        else "HiFi-GAN",
        pipeline=str(row["pipeline"])
        if "pipeline" in keys and row["pipeline"]
        else "rvc",
        best_model_path=_resolve_best_model_path(row),
        best_epoch=int(row["best_epoch"])
        if "best_epoch" in keys and row["best_epoch"] is not None
        else None,
        best_avg_gen_loss=float(row["best_avg_gen_loss"])
        if "best_avg_gen_loss" in keys and row["best_avg_gen_loss"] is not None
        else None,
        audio_files=audio_files,
    )


# ---------------------------------------------------------------------------
# Shared upload + clip helper
# ---------------------------------------------------------------------------


async def _upload_and_clip(
    file: UploadFile,
    audio_dir: str,
    seg_start: float,
    seg_end: Optional[float],
    original_filename: str,
) -> tuple[str, str, float]:
    """Write the upload, clip to [seg_start, seg_end], save as deduped .wav.

    Returns (file_path, filename, clip_duration_sec).
    Raises HTTPException on size/format errors.
    Caller is responsible for ensuring audio_dir exists.
    """
    import soundfile as sf  # noqa: PLC0415

    raw_path = os.path.join(
        audio_dir, f"_upload_{uuid.uuid4().hex}_{original_filename}"
    )

    bytes_written = 0
    try:
        async with aiofiles.open(raw_path, "wb") as out_f:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail="File exceeds 200 MiB limit"
                    )
                await out_f.write(chunk)
    except HTTPException:
        try:
            os.remove(raw_path)
        except OSError:
            pass
        raise
    except OSError as exc:
        try:
            os.remove(raw_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"File write error: {exc}") from exc

    loop = asyncio.get_event_loop()
    duration = await loop.run_in_executor(None, get_duration, raw_path)

    if duration is not None and duration > MAX_DURATION_SEC:
        try:
            os.remove(raw_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=422,
            detail=f"Audio is {duration / 60:.1f} min — max allowed is 60 min",
        )

    # Derive a clean stem from the original filename
    stem = os.path.splitext(original_filename)[0] or "audio"
    # Strip path separators that could be injected
    stem = stem.replace("/", "_").replace("\\", "_").strip(".")[:80]
    if not stem:
        stem = "audio"

    filename = _dedup_filename(audio_dir, stem, ".wav")
    clip_path = os.path.join(audio_dir, filename)

    end_for_clip = seg_end if seg_end is not None else duration
    clip_dur: Optional[float] = None

    try:
        raw, sr = sf.read(raw_path, always_2d=True, dtype="float32")
        s_frame = int(round((seg_start or 0.0) * sr))
        e_frame = (
            int(round(end_for_clip * sr)) if end_for_clip is not None else raw.shape[0]
        )
        s_frame = max(0, min(s_frame, raw.shape[0]))
        e_frame = max(s_frame, min(e_frame, raw.shape[0]))
        clipped = raw[s_frame:e_frame, :]
        sf.write(clip_path, clipped, sr, subtype="PCM_16")
        clip_dur = round(clipped.shape[0] / sr, 3)
    except Exception as exc:
        logger.warning("clip failed (%s) — keeping original: %s", raw_path, exc)
        shutil.copy2(raw_path, clip_path)
        clip_dur = duration

    try:
        os.remove(raw_path)
    except OSError:
        pass

    return clip_path, filename, clip_dur or 0.0


# ---------------------------------------------------------------------------
# POST /api/profiles  — create profile + first audio file
# ---------------------------------------------------------------------------


@router.post("", status_code=200, response_model=ProfileOut)
async def create_profile(
    name: str = Form(...),
    file: UploadFile = ...,
    seg_start: float = Form(0.0),
    seg_end: Optional[float] = Form(None),
    embedder: str = Form("spin-v2"),
    vocoder: str = Form("HiFi-GAN"),
    pipeline: str = Form("rvc"),
) -> ProfileOut:
    """Upload an audio file, clip to the selected segment, and create a profile.

    embedder: feature embedder to use during training (locked after first run).
    vocoder: decoder architecture — "HiFi-GAN" or "RefineGAN" (locked after first run).
    pipeline: voice conversion engine — "rvc" (default) or "beatrice2" (CUDA only).
    """
    import torch

    # Validate pipeline
    _VALID_PIPELINES = {"rvc", "beatrice2"}
    if pipeline not in _VALID_PIPELINES:
        raise HTTPException(status_code=400, detail=f"Invalid pipeline: {pipeline!r}")
    if pipeline == "beatrice2" and not torch.cuda.is_available():
        raise HTTPException(
            status_code=400,
            detail="Beatrice 2 requires a CUDA GPU. No CUDA device is available on this machine.",
        )

    # Validate embedder / vocoder only for RVC profiles
    _VALID_EMBEDDERS = {"spin-v2", "spin", "contentvec", "hubert"}
    if pipeline == "rvc" and embedder not in _VALID_EMBEDDERS:
        raise HTTPException(status_code=400, detail=f"Invalid embedder: {embedder!r}")
    _VALID_VOCODERS = {"HiFi-GAN", "RefineGAN"}
    if pipeline == "rvc" and vocoder not in _VALID_VOCODERS:
        raise HTTPException(status_code=400, detail=f"Invalid vocoder: {vocoder!r}")

    profile_id = uuid.uuid4().hex
    pdir = os.path.join(_profiles_root(), profile_id)
    audio_dir = _audio_dir(pdir)
    os.makedirs(audio_dir, exist_ok=True)
    loop = asyncio.get_event_loop()

    original_filename = file.filename or "upload.wav"

    try:
        file_path, filename, clip_dur = await _upload_and_clip(
            file, audio_dir, seg_start, seg_end, original_filename
        )
    except HTTPException:
        shutil.rmtree(pdir, ignore_errors=True)
        raise

    # DC removal → highpass 80Hz → silence trim
    try:
        pp = await loop.run_in_executor(None, preprocess_audio_inplace, file_path)
        clip_dur = pp["duration_after_s"] or clip_dur
        logger.info(
            "preprocessed first file: trimmed=%.0fms  dur=%.1fs",
            pp["trimmed_ms"],
            clip_dur,
        )
    except Exception as exc:
        logger.warning(
            "audio preprocessing failed for %s: %s — continuing without", file_path, exc
        )

    # Measure the speech-weighted RMS of this first (preprocessed) file.
    # All subsequent files added to the profile will be normalized to match it.
    profile_rms: Optional[float] = None
    try:
        profile_rms = await loop.run_in_executor(None, measure_speech_rms, file_path)
    except Exception as exc:
        logger.warning("profile RMS measurement failed for %s: %s", file_path, exc)

    created_at = datetime.now(timezone.utc).isoformat()
    file_id = uuid.uuid4().hex

    async with get_db() as db:
        await db.execute(
            """INSERT INTO profiles
               (id, name, status, sample_path, batch_size, audio_duration,
                preprocessed_path, profile_dir, profile_rms, embedder, vocoder, pipeline, created_at)
               VALUES (?, ?, 'untrained', ?, 8, ?, NULL, ?, ?, ?, ?, ?, ?)""",
            (
                profile_id,
                name,
                file_path,
                clip_dur,
                pdir,
                profile_rms,
                embedder,
                vocoder,
                pipeline,
                created_at,
            ),
        )
        await db.execute(
            """INSERT INTO audio_files (id, profile_id, filename, file_path, duration, is_cleaned, created_at)
               VALUES (?, ?, ?, ?, ?, 0, ?)""",
            (file_id, profile_id, filename, file_path, clip_dur, created_at),
        )
        await db.commit()

        cursor = await db.execute(
            """SELECT id, name, status, created_at, sample_path,
                      batch_size, audio_duration, preprocessed_path,
                      profile_dir, model_path, checkpoint_path, index_path,
                      total_epochs_trained, needs_retraining, profile_rms, embedder, vocoder, pipeline,
                      best_epoch, best_avg_gen_loss
               FROM profiles WHERE id = ?""",
            (profile_id,),
        )
        row = await cursor.fetchone()
        result = await _row_to_out(row, db)

    logger.info(
        "profile created id=%s name=%s dir=%s dur=%.1fs",
        profile_id,
        name,
        pdir,
        clip_dur,
    )
    return result


# ---------------------------------------------------------------------------
# POST /api/profiles/{id}/audio  — add audio file to existing profile
# ---------------------------------------------------------------------------


@router.post("/{profile_id}/audio", status_code=200, response_model=AudioFileOut)
async def add_audio_file(
    profile_id: str,
    file: UploadFile = ...,
    seg_start: float = Form(0.0),
    seg_end: Optional[float] = Form(None),
) -> AudioFileOut:
    """Clip and add an audio file to an existing profile."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, profile_dir, total_epochs_trained, profile_rms FROM profiles WHERE id = ?",
            (profile_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")

    pdir = row["profile_dir"]
    if not pdir:
        raise HTTPException(
            status_code=400, detail="Profile has no directory — recreate it"
        )

    audio_dir = _audio_dir(pdir)
    os.makedirs(audio_dir, exist_ok=True)
    loop = asyncio.get_event_loop()

    original_filename = file.filename or "upload.wav"
    file_path, filename, clip_dur = await _upload_and_clip(
        file, audio_dir, seg_start, seg_end, original_filename
    )

    # DC removal → highpass 80Hz → silence trim
    try:
        pp = await loop.run_in_executor(None, preprocess_audio_inplace, file_path)
        clip_dur = pp["duration_after_s"] or clip_dur
    except Exception as exc:
        logger.warning(
            "audio preprocessing failed for %s: %s — continuing without", file_path, exc
        )

    # Normalize to the profile's RMS reference so all training files are
    # level-consistent.  If the profile has no stored RMS yet (legacy profile),
    # skip normalization — it will be measured next time training starts.
    stored_rms = row["profile_rms"]
    if stored_rms is not None:
        try:
            await loop.run_in_executor(
                None, normalize_rms_inplace, file_path, float(stored_rms)
            )
            logger.info(
                "normalized new file %s to profile_rms=%.5f", filename, stored_rms
            )
        except Exception as exc:
            logger.warning(
                "RMS normalization failed for %s: %s — keeping original level",
                filename,
                exc,
            )

    file_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    prior_epochs = int(row["total_epochs_trained"] or 0)

    async with get_db() as db:
        await db.execute(
            """INSERT INTO audio_files (id, profile_id, filename, file_path, duration, is_cleaned, created_at)
               VALUES (?, ?, ?, ?, ?, 0, ?)""",
            (file_id, profile_id, filename, file_path, clip_dur, created_at),
        )
        if prior_epochs > 0:
            await db.execute(
                "UPDATE profiles SET needs_retraining = 1 WHERE id = ?",
                (profile_id,),
            )
        await db.commit()

    logger.info(
        "audio file added profile=%s file=%s dur=%.1fs", profile_id, filename, clip_dur
    )
    return AudioFileOut(
        id=file_id,
        profile_id=profile_id,
        filename=filename,
        file_path=file_path,
        duration=clip_dur,
        is_cleaned=False,
        created_at=created_at,
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
                      profile_dir, model_path, checkpoint_path, index_path,
                      total_epochs_trained, needs_retraining, profile_rms, embedder, vocoder, pipeline,
                      best_epoch, best_avg_gen_loss
               FROM profiles ORDER BY created_at DESC"""
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            results.append(await _row_to_out(row, db))
    return results


# ---------------------------------------------------------------------------
# GET /api/profiles/{id}
# ---------------------------------------------------------------------------


@router.get("/{profile_id}", response_model=ProfileOut)
async def get_profile(profile_id: str) -> ProfileOut:
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT id, name, status, created_at, sample_path,
                      batch_size, audio_duration, preprocessed_path,
                      profile_dir, model_path, checkpoint_path, index_path,
                      total_epochs_trained, needs_retraining, profile_rms, embedder, vocoder, pipeline,
                      best_epoch, best_avg_gen_loss
               FROM profiles WHERE id = ?""",
            (profile_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(
                status_code=404, detail=f"Profile not found: {profile_id}"
            )
        return await _row_to_out(row, db)


# ---------------------------------------------------------------------------
# GET /api/profiles/{id}/audio/{file_id}  — stream individual audio file
# ---------------------------------------------------------------------------


@router.get("/{profile_id}/audio/{file_id}")
async def stream_audio_file(profile_id: str, file_id: str):
    """Stream an individual audio file."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT file_path, filename FROM audio_files WHERE id = ? AND profile_id = ?",
            (file_id, profile_id),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Audio file not found")
    path = row["file_path"]
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    # Serve as WAV; filename for download prompt
    return FileResponse(path, media_type="audio/wav", filename=row["filename"])


# ---------------------------------------------------------------------------
# DELETE /api/profiles/{id}/audio/{file_id}  — delete one audio file
# ---------------------------------------------------------------------------


@router.delete("/{profile_id}/audio/{file_id}", status_code=204)
async def delete_audio_file(profile_id: str, file_id: str) -> Response:
    """Delete an audio file from the profile. Sets needs_retraining if trained."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT file_path FROM audio_files WHERE id = ? AND profile_id = ?",
            (file_id, profile_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Audio file not found")

        cursor2 = await db.execute(
            "SELECT total_epochs_trained FROM profiles WHERE id = ?",
            (profile_id,),
        )
        profile_row = await cursor2.fetchone()
        prior_epochs = (
            int(profile_row["total_epochs_trained"] or 0) if profile_row else 0
        )

        await db.execute(
            "DELETE FROM audio_files WHERE id = ?",
            (file_id,),
        )
        if prior_epochs > 0:
            await db.execute(
                "UPDATE profiles SET needs_retraining = 1 WHERE id = ?",
                (profile_id,),
            )
        await db.commit()

    # Remove file from disk (non-fatal if missing)
    path = row["file_path"]
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning("could not remove audio file %s: %s", path, exc)

    return Response(status_code=204)


# ---------------------------------------------------------------------------
# POST /api/profiles/{id}/audio/{file_id}/clean  — noise removal in-place
# ---------------------------------------------------------------------------


@router.post("/{profile_id}/audio/{file_id}/clean", response_model=PreprocessResponse)
async def clean_audio_file(profile_id: str, file_id: str) -> PreprocessResponse:
    """Run noise removal on one audio file, overwriting it in-place.

    After noise removal the file is RMS-normalized to the profile's stored
    reference level.  The noise removal step peak-normalizes to −1 dBFS which
    changes the RMS; the subsequent RMS step brings it back to the profile level.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT file_path, filename FROM audio_files WHERE id = ? AND profile_id = ?",
            (file_id, profile_id),
        )
        row = await cursor.fetchone()
        cursor2 = await db.execute(
            "SELECT profile_rms FROM profiles WHERE id = ?",
            (profile_id,),
        )
        profile_row = await cursor2.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Audio file not found")

    file_path = row["file_path"]
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Audio file not found on disk")

    from backend.app.audio_preprocessing import remove_noise  # noqa: PLC0415

    loop = asyncio.get_event_loop()
    try:
        # Write to a temp file first, then atomically replace on success
        tmp_path = file_path + ".cleaning"
        await loop.run_in_executor(
            None,
            lambda: remove_noise(file_path, tmp_path),
        )
        os.replace(tmp_path, file_path)
    except RuntimeError as exc:
        # Clean up temp if it exists
        tmp_path = file_path + ".cleaning"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return PreprocessResponse(ok=False, message=str(exc))
    except Exception as exc:
        tmp_path = file_path + ".cleaning"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return PreprocessResponse(ok=False, message=f"Unexpected error: {exc}")

    # Re-apply RMS normalization — noise removal peak-normalizes to −1 dBFS
    # which shifts the RMS away from the profile reference.
    stored_rms = profile_row["profile_rms"] if profile_row else None
    if stored_rms is not None:
        try:
            await loop.run_in_executor(
                None, normalize_rms_inplace, file_path, float(stored_rms)
            )
        except Exception as exc:
            logger.warning(
                "post-clean RMS normalization failed for %s: %s", file_path, exc
            )

    cleaned_dur = await loop.run_in_executor(None, get_duration, file_path)

    async with get_db() as db:
        await db.execute(
            "UPDATE audio_files SET is_cleaned = 1, duration = ? WHERE id = ?",
            (cleaned_dur, file_id),
        )
        await db.commit()

    return PreprocessResponse(
        ok=True,
        message="Noise removal complete",
        duration_sec=cleaned_dur,
    )


# ---------------------------------------------------------------------------
# GET /api/profiles/{id}/health
# ---------------------------------------------------------------------------


@router.get("/{profile_id}/health", response_model=HealthStatus)
async def profile_health(profile_id: str) -> HealthStatus:
    """File-existence health check for a profile."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT id, status, profile_dir, model_path, checkpoint_path, index_path,
                      pipeline
               FROM profiles WHERE id = ?""",
            (profile_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(
                status_code=404, detail=f"Profile not found: {profile_id}"
            )

        audio_files = await _fetch_audio_files(db, profile_id)

    errors: list[str] = []
    pipeline = (row["pipeline"] if "pipeline" in row.keys() else None) or "rvc"
    is_beatrice2 = pipeline == "beatrice2"

    # Audio check — at least one file must exist on disk
    audio_ok = (
        any(os.path.exists(af.file_path) for af in audio_files)
        if audio_files
        else False
    )
    if not audio_ok:
        errors.append("no audio files found — add at least one audio file to train")

    # Model check
    model = _resolve_model_path(row)
    model_ok = bool(model and os.path.exists(model))
    if row["status"] == "trained" and not model_ok:
        errors.append(
            "model_infer.pth missing — profile is marked trained but inference model not found"
        )

    # Index check — Beatrice 2 has no index file; skip the check entirely.
    if is_beatrice2:
        index_ok = True
    else:
        index = _resolve_index_path(row)
        index_ok = bool(index and os.path.exists(index))
        if row["status"] == "trained" and not index_ok:
            errors.append(
                "model.index missing — profile is marked trained but index file not found"
            )

    can_train = audio_ok
    can_infer = model_ok and index_ok

    return HealthStatus(
        profile_id=profile_id,
        audio_ok=audio_ok,
        model_ok=model_ok,
        index_ok=index_ok,
        can_train=can_train,
        can_infer=can_infer,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# PATCH /api/profiles/{id} — update embedder/vocoder (only if not locked)
# ---------------------------------------------------------------------------


class PatchProfileRequest(BaseModel):
    embedder: Optional[str] = None
    vocoder: Optional[str] = None


@router.patch("/{profile_id}")
async def patch_profile(profile_id: str, body: PatchProfileRequest):
    """Update embedder or vocoder for a profile. Only allowed before first training."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, embedder, vocoder, total_epochs_trained FROM profiles WHERE id = ?",
            (profile_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(
                status_code=404, detail=f"Profile not found: {profile_id}"
            )

        if row["total_epochs_trained"] > 0:
            raise HTTPException(
                status_code=400,
                detail="Cannot change embedder/vocoder after first training run",
            )

        updates = {}
        if body.embedder is not None:
            if body.embedder not in ("spin-v2", "spin", "contentvec", "hubert"):
                raise HTTPException(
                    status_code=400, detail=f"Invalid embedder: {body.embedder}"
                )
            updates["embedder"] = body.embedder

        if body.vocoder is not None:
            if body.vocoder not in ("HiFi-GAN", "RefineGAN"):
                raise HTTPException(
                    status_code=400, detail=f"Invalid vocoder: {body.vocoder}"
                )
            updates["vocoder"] = body.vocoder

        if not updates:
            raise HTTPException(status_code=400, detail="No valid fields to update")

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        await db.execute(
            f"UPDATE profiles SET {set_clause} WHERE id = ?",
            (*updates.values(), profile_id),
        )
        await db.commit()

    return {"ok": True, **updates}


# ---------------------------------------------------------------------------
# DELETE /api/profiles/{id}
# ---------------------------------------------------------------------------


@router.delete("/{profile_id}", status_code=204)
async def delete_profile(profile_id: str) -> Response:
    """Delete a profile, all its audio_files, epoch_losses, beatrice_steps rows, and the entire profile_dir tree."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, profile_dir FROM profiles WHERE id = ?", (profile_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(
                status_code=404, detail=f"Profile not found: {profile_id}"
            )

        # audio_files and epoch_losses have ON DELETE CASCADE but aiosqlite
        # doesn't enforce FK constraints by default — delete explicitly.
        await db.execute("DELETE FROM audio_files WHERE profile_id = ?", (profile_id,))
        await db.execute("DELETE FROM epoch_losses WHERE profile_id = ?", (profile_id,))
        await db.execute("DELETE FROM beatrice_steps WHERE profile_id = ?", (profile_id,))
        await db.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        await db.commit()

    # Remove the entire profile directory (covers audio, checkpoints, models)
    pdir = row["profile_dir"]
    if pdir and os.path.isdir(pdir):
        shutil.rmtree(pdir, ignore_errors=True)

    # Also clean up legacy samples dir if it exists
    legacy_dir = os.path.join(_samples_root(), profile_id)
    if os.path.isdir(legacy_dir):
        shutil.rmtree(legacy_dir, ignore_errors=True)

    # Wipe the shared RVC experiment workspace so a subsequent fresh profile
    # doesn't inherit stale checkpoints from this one.
    project_root = os.environ.get("PROJECT_ROOT", "")
    if project_root:
        exp_dir = os.path.join(
            os.path.abspath(project_root), "logs", "rvc_finetune_active"
        )
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir, ignore_errors=True)

    return Response(status_code=204)


# ---------------------------------------------------------------------------
# GET /api/profiles/{id}/export  — export profile as rvc-profile-v1.zip
# ---------------------------------------------------------------------------

# Files included in the export bundle (relative to profile_dir).
# D_latest.pth is excluded — it's the discriminator checkpoint (~431 MB) which
# is only useful mid-training and would inflate the zip for little benefit.
_EXPORT_FILES = [
    "model_infer.pth",
    "model_best.pth",
    "model.index",
    "profile_embedding.pt",
    "train.log",
    "checkpoints/G_latest.pth",
    "checkpoints/G_best.pth",
]


@router.get("/{profile_id}/export")
async def export_profile(profile_id: str):
    """Build a portable rvc-profile-v1.zip and stream it to the client.

    Zip layout:
        manifest.json             — DB metadata, no absolute paths
        audio/voice.wav           — all audio files
        model_infer.pth           — inference model (if present)
        model_best.pth            — best-epoch model (if present)
        model.index               — FAISS index (if present)
        profile_embedding.pt      — speaker embedding (if present)
        checkpoints/G_latest.pth  — resume checkpoint (if present)
        checkpoints/G_best.pth    — best-epoch generator (if present)
        train.log                 — training log (if present)

    D_latest.pth (discriminator) is excluded: ~431 MB, not needed for inference
    or resume training (train.py reconstructs it from pretrain weights on resume).
    """
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT id, name, status, created_at, batch_size,
                      profile_dir, total_epochs_trained, needs_retraining,
                      profile_rms, embedder, vocoder, pipeline,
                      best_epoch, best_avg_gen_loss
               FROM profiles WHERE id = ?""",
            (profile_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")

        audio_cursor = await db.execute(
            """SELECT filename, file_path, duration, is_cleaned, created_at
               FROM audio_files WHERE profile_id = ? ORDER BY created_at ASC""",
            (profile_id,),
        )
        audio_rows = await audio_cursor.fetchall()

        loss_cursor = await db.execute(
            """SELECT epoch, loss_mel, loss_gen, loss_disc, loss_fm, loss_kl, loss_spk, trained_at
               FROM epoch_losses WHERE profile_id = ? ORDER BY epoch ASC""",
            (profile_id,),
        )
        loss_rows = await loss_cursor.fetchall()

    pdir = row["profile_dir"]
    if not pdir or not os.path.isdir(pdir):
        raise HTTPException(
            status_code=400, detail="Profile directory not found — cannot export"
        )

    files_included: list[str] = []
    manifest: dict = {
        "format_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "profile": {
            "name": row["name"],
            "status": row["status"],
            "created_at": row["created_at"],
            "batch_size": row["batch_size"] or 8,
            "embedder": row["embedder"] or "spin-v2",
            "vocoder": row["vocoder"] or "HiFi-GAN",
            "pipeline": row["pipeline"] or "rvc",
            "total_epochs_trained": int(row["total_epochs_trained"] or 0),
            "needs_retraining": bool(row["needs_retraining"]),
            "profile_rms": float(row["profile_rms"]) if row["profile_rms"] is not None else None,
            "best_epoch": int(row["best_epoch"]) if row["best_epoch"] is not None else None,
            "best_avg_gen_loss": float(row["best_avg_gen_loss"]) if row["best_avg_gen_loss"] is not None else None,
        },
        "audio_files": [
            {
                "filename": r["filename"],
                "duration": r["duration"],
                "is_cleaned": bool(r["is_cleaned"]),
                "created_at": r["created_at"],
            }
            for r in audio_rows
        ],
        "epoch_losses": [
            {
                "epoch": r["epoch"],
                "loss_mel": r["loss_mel"],
                "loss_gen": r["loss_gen"],
                "loss_disc": r["loss_disc"],
                "loss_fm": r["loss_fm"],
                "loss_kl": r["loss_kl"],
                "loss_spk": r["loss_spk"],
                "trained_at": r["trained_at"],
            }
            for r in loss_rows
        ],
        "files_included": [],  # filled after we know what actually exists
    }

    # Write the zip to a temp file on disk — never holds file data in RAM.
    # zipfile.ZipFile.write() copies directly from source path to the zip file
    # on disk without buffering through Python. ZIP_STORED is used because
    # .pth/.wav files are already binary-dense and don't compress meaningfully.
    # The temp file is streamed back in 64 KB chunks and deleted in the
    # generator's finally block, so it is cleaned up even if the client
    # disconnects mid-download.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="rvc-export-")
    os.close(tmp_fd)  # zipfile opens by path; release the fd

    try:
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_STORED) as zf:
            # Audio files
            for ar in audio_rows:
                src = ar["file_path"]
                if src and os.path.isfile(src):
                    arc_name = f"audio/{ar['filename']}"
                    zf.write(src, arc_name)
                    files_included.append(arc_name)
                else:
                    logger.warning("export: audio file missing on disk: %s", src)

            # Static known files (RVC)
            for rel in _EXPORT_FILES:
                src = os.path.join(pdir, rel)
                if os.path.isfile(src):
                    zf.write(src, rel)
                    files_included.append(rel)

            # Beatrice 2 files (if applicable)
            if (row.get("pipeline") or "rvc") == "beatrice2":
                b2_out_dir = os.path.join(pdir, "beatrice2_out")
                # checkpoint_latest.pt.gz
                b2_ckpt = os.path.join(b2_out_dir, "checkpoint_latest.pt.gz")
                if os.path.isfile(b2_ckpt):
                    zf.write(b2_ckpt, "beatrice2_out/checkpoint_latest.pt.gz")
                    files_included.append("beatrice2_out/checkpoint_latest.pt.gz")
                # Most recent paraphernalia dir (inference binaries)
                if os.path.isdir(b2_out_dir):
                    para_dirs = sorted(
                        [d for d in os.listdir(b2_out_dir)
                         if d.startswith("paraphernalia_") and
                         os.path.isdir(os.path.join(b2_out_dir, d))],
                        reverse=True,
                    )
                    if para_dirs:
                        latest_para = os.path.join(b2_out_dir, para_dirs[0])
                        for fname in os.listdir(latest_para):
                            fpath = os.path.join(latest_para, fname)
                            if os.path.isfile(fpath):
                                arc = f"beatrice2_out/{para_dirs[0]}/{fname}"
                                zf.write(fpath, arc)
                                files_included.append(arc)

            # Finalize manifest with the actual file list
            manifest["files_included"] = files_included
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    safe_name = row["name"].replace(" ", "_").replace("/", "_")[:40]
    filename = f"{safe_name}.rvc-profile.zip"
    size_mb = os.path.getsize(tmp_path) / 1024 / 1024

    logger.info(
        "export: profile=%s name=%s files=%d size=%.1f MB",
        profile_id, row["name"], len(files_included), size_mb,
    )

    def _stream_and_delete(path: str, chunk_size: int = 64 * 1024):
        """Yield file in chunks, delete temp file when done or on error."""
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    return StreamingResponse(
        _stream_and_delete(tmp_path),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# POST /api/profiles/import  — import profile from rvc-profile-v1.zip
#
# IMPORTANT: this route must be registered BEFORE /{profile_id} routes so
# FastAPI does not match the literal string "import" as a profile_id.
# The import endpoint is wired in main.py via a separate sub-router registered
# with higher priority, or simply declared first in this file. Since FastAPI
# resolves routes in declaration order for the same HTTP method, declaring
# POST /import before POST /{profile_id}/audio is sufficient.
# ---------------------------------------------------------------------------


class ImportProfileResponse(BaseModel):
    profile: ProfileOut
    warnings: list[str] = []


@router.post("/import", status_code=200, response_model=ImportProfileResponse)
async def import_profile(
    file: UploadFile = ...,
    name_override: Optional[str] = Form(None),
) -> ImportProfileResponse:
    """Import a profile from a rvc-profile-v1.zip bundle.

    Always creates a new profile with a fresh UUID — the source profile_id is
    never reused.  If the profile name already exists in the DB and no
    name_override is supplied, returns HTTP 409 with a JSON detail describing
    the collision and a suggested alternative name.

    Request: multipart/form-data
        file:          the .zip file  (required)
        name_override: rename profile on import  (optional)

    Responses:
        200 — success; body: ImportProfileResponse
        409 — name collision; detail: {code, name, suggested}
        422 — invalid zip or manifest format
    """
    warnings: list[str] = []

    # FastAPI UploadFile.file is a SpooledTemporaryFile backed by disk once it
    # exceeds the spool threshold (~1 MB by default).  We seek to 0 and pass it
    # directly to zipfile — no read() into memory.  Size check via seek/tell.
    upload_fh = file.file
    upload_fh.seek(0, 2)  # seek to end
    upload_size = upload_fh.tell()
    upload_fh.seek(0)

    if upload_size > 4 * 1024 * 1024 * 1024:  # 4 GB hard cap
        raise HTTPException(status_code=413, detail="Import file exceeds 4 GB limit")

    try:
        zf = zipfile.ZipFile(upload_fh)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=422, detail=f"Not a valid zip file: {exc}") from exc

    if "manifest.json" not in zf.namelist():
        raise HTTPException(status_code=422, detail="manifest.json not found in zip")

    try:
        manifest = json.loads(zf.read("manifest.json"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"manifest.json parse error: {exc}") from exc

    if manifest.get("format_version") != 1:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported format_version: {manifest.get('format_version')} (expected 1)",
        )

    prof_meta = manifest.get("profile", {})
    if not prof_meta.get("name"):
        raise HTTPException(status_code=422, detail="manifest.json: profile.name is missing")

    source_name: str = prof_meta["name"]
    import_name: str = name_override.strip() if (name_override and name_override.strip()) else source_name

    # --- Name collision check ---
    async with get_db() as db:
        cursor = await db.execute("SELECT id FROM profiles WHERE name = ?", (import_name,))
        collision = await cursor.fetchone()

    if collision:
        # Suggest a free name: "Original Name (1)", "(2)", etc.
        n = 1
        async with get_db() as db:
            while True:
                candidate = f"{source_name} ({n})"
                cursor = await db.execute("SELECT id FROM profiles WHERE name = ?", (candidate,))
                if await cursor.fetchone() is None:
                    break
                n += 1
        raise HTTPException(
            status_code=409,
            detail={
                "code": "name_taken",
                "name": import_name,
                "suggested": candidate,
            },
        )

    # --- Allocate new profile directory ---
    new_profile_id = uuid.uuid4().hex
    pdir = os.path.join(_profiles_root(), new_profile_id)
    audio_dir = os.path.join(pdir, "audio")
    ckpt_dir = os.path.join(pdir, "checkpoints")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    zip_names = set(zf.namelist())
    files_included = set(manifest.get("files_included", []))

    try:
        # Extract audio files
        audio_meta = manifest.get("audio_files", [])
        extracted_audio: list[dict] = []
        for af in audio_meta:
            arc_name = f"audio/{af['filename']}"
            if arc_name in zip_names:
                dest = os.path.join(audio_dir, af["filename"])
                with zf.open(arc_name) as src_f, open(dest, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                extracted_audio.append({**af, "file_path": dest})
            else:
                warnings.append(f"Audio file missing from zip: {af['filename']}")
                logger.warning("import: audio file %s not in zip", arc_name)

        # Extract static files
        _STATIC_IMPORTS = [
            "model_infer.pth",
            "model_best.pth",
            "model.index",
            "profile_embedding.pt",
            "train.log",
            "checkpoints/G_latest.pth",
            "checkpoints/G_best.pth",
        ]
        for rel in _STATIC_IMPORTS:
            if rel in zip_names:
                dest = os.path.join(pdir, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(rel) as src_f, open(dest, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
            elif rel in files_included:
                warnings.append(f"Expected file missing from zip: {rel}")

        # Extract Beatrice 2 files (checkpoint + paraphernalia)
        if prof_meta.get("pipeline") == "beatrice2":
            b2_files = [n for n in zip_names if n.startswith("beatrice2_out/")]
            for arc_name in b2_files:
                dest = os.path.join(pdir, arc_name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(arc_name) as src_f, open(dest, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)

    except Exception as exc:
        shutil.rmtree(pdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract zip: {exc}") from exc

    # --- Write DB rows ---
    created_at = datetime.now(timezone.utc).isoformat()
    epoch_losses = manifest.get("epoch_losses", [])

    async with get_db() as db:
        await db.execute(
            """INSERT INTO profiles
               (id, name, status, sample_path, batch_size, audio_duration,
                preprocessed_path, profile_dir, profile_rms, embedder, vocoder, pipeline,
                total_epochs_trained, needs_retraining, best_epoch, best_avg_gen_loss,
                created_at)
               VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                new_profile_id,
                import_name,
                prof_meta.get("status", "untrained"),
                extracted_audio[0]["file_path"] if extracted_audio else "",
                int(prof_meta.get("batch_size", 8)),
                sum(af.get("duration") or 0 for af in extracted_audio) or None,
                pdir,
                float(prof_meta["profile_rms"]) if prof_meta.get("profile_rms") is not None else None,
                prof_meta.get("embedder", "spin-v2"),
                prof_meta.get("vocoder", "HiFi-GAN"),
                prof_meta.get("pipeline", "rvc"),
                int(prof_meta.get("total_epochs_trained", 0)),
                int(bool(prof_meta.get("needs_retraining", False))),
                int(prof_meta["best_epoch"]) if prof_meta.get("best_epoch") is not None else None,
                float(prof_meta["best_avg_gen_loss"]) if prof_meta.get("best_avg_gen_loss") is not None else None,
                created_at,
            ),
        )

        for af in extracted_audio:
            await db.execute(
                """INSERT INTO audio_files
                   (id, profile_id, filename, file_path, duration, is_cleaned, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex,
                    new_profile_id,
                    af["filename"],
                    af["file_path"],
                    af.get("duration"),
                    int(bool(af.get("is_cleaned", False))),
                    af.get("created_at", created_at),
                ),
            )

        for el in epoch_losses:
            if el.get("epoch") is None:
                continue
            await db.execute(
                """INSERT OR IGNORE INTO epoch_losses
                   (id, profile_id, epoch, loss_mel, loss_gen, loss_disc,
                    loss_fm, loss_kl, loss_spk, trained_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex,
                    new_profile_id,
                    el["epoch"],
                    el.get("loss_mel"),
                    el.get("loss_gen"),
                    el.get("loss_disc"),
                    el.get("loss_fm"),
                    el.get("loss_kl"),
                    el.get("loss_spk"),
                    el.get("trained_at", created_at),
                ),
            )

        await db.commit()

        cursor = await db.execute(
            """SELECT id, name, status, created_at, sample_path,
                      batch_size, audio_duration, preprocessed_path,
                      profile_dir, model_path, checkpoint_path, index_path,
                      total_epochs_trained, needs_retraining, profile_rms, embedder, vocoder, pipeline,
                      best_epoch, best_avg_gen_loss
               FROM profiles WHERE id = ?""",
            (new_profile_id,),
        )
        new_row = await cursor.fetchone()
        profile_out = await _row_to_out(new_row, db)

    logger.info(
        "import: new_id=%s name=%s epochs=%d audio=%d warnings=%d",
        new_profile_id, import_name,
        prof_meta.get("total_epochs_trained", 0),
        len(extracted_audio), len(warnings),
    )

    return ImportProfileResponse(profile=profile_out, warnings=warnings)


# ---------------------------------------------------------------------------
# Legacy compat endpoints — kept so old frontend code doesn't 404
# These are superseded by the per-file endpoints above.
# ---------------------------------------------------------------------------


@router.get("/{profile_id}/audio")
async def stream_audio_legacy(profile_id: str):
    """Stream the first audio file for backward compat."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT file_path, filename FROM audio_files WHERE profile_id = ? ORDER BY created_at ASC LIMIT 1",
            (profile_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No audio files for this profile")
    path = row["file_path"]
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    return FileResponse(path, media_type="audio/wav", filename=row["filename"])


@router.get("/{profile_id}/audio/clean")
async def stream_clean_audio_legacy(profile_id: str):
    """Stream the first cleaned audio file for backward compat."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT file_path, filename FROM audio_files
               WHERE profile_id = ? AND is_cleaned = 1
               ORDER BY created_at ASC LIMIT 1""",
            (profile_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail="No cleaned audio files for this profile"
        )
    path = row["file_path"]
    if not path or not os.path.exists(path):
        raise HTTPException(
            status_code=404, detail="Cleaned audio file not found on disk"
        )
    return FileResponse(path, media_type="audio/wav", filename=row["filename"])
