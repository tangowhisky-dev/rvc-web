"""Analysis router — server-side temp file approach.

Upload flow (direct):
  POST /api/analysis/upload           → { ref_id, slot }  (saves to temp/)
  POST /api/analysis/compare          → CompareResponse   (reads from temp/ by ref_id)
  GET  /api/analysis/audio/{ref_id}   → streams temp file (for waveform rendering)

Offline flow:
  Offline convert already saves input+output to temp/ under the job_id.
  POST /api/offline/analyze/{job_id}  → CompareResponse   (in offline router)
  GET  /api/offline/input/{job_id}    → streams input temp file
  GET  /api/offline/result/{job_id}   → streams output temp file  (unchanged)
"""

import logging
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger("rvc_web.analysis")

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temp_dir() -> str:
    root = os.environ.get("PROJECT_ROOT", "")
    if not root:
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
    d = os.path.join(root, "temp")
    os.makedirs(d, exist_ok=True)
    return d


def _temp_path(ref_id: str, slot: str, ext: str) -> str:
    return os.path.join(_temp_dir(), f"{ref_id}_{slot}{ext}")


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    ref_id: str
    slot: str   # "a" or "b"

class CompareResponse(BaseModel):
    ab_similarity: float
    profile_a_similarity: Optional[float] = None
    profile_b_similarity: Optional[float] = None
    improvement: Optional[float] = None
    improvement_pct: Optional[float] = None
    quality_a: str
    quality_b: str
    quality_profile_a: Optional[str] = None
    quality_profile_b: Optional[str] = None
    summary: str
    emb_a: list[float]
    emb_b: list[float]
    emb_profile: Optional[list[float]] = None
    duration_a: float
    duration_b: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    slot: str = Form("a"),   # "a" or "b"
):
    """Save an uploaded audio file to temp/ and return a ref_id.

    The client holds onto the ref_id and passes it to /compare.
    """
    if slot not in ("a", "b"):
        raise HTTPException(status_code=400, detail="slot must be 'a' or 'b'")

    ref_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    dest = _temp_path(ref_id, slot, ext)

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    logger.info("Saved analysis upload: slot=%s ref=%s size=%d", slot, ref_id, len(content))
    return UploadResponse(ref_id=ref_id, slot=slot)


@router.post("/compare", response_model=CompareResponse)
async def compare_refs(
    ref_a: str = Form(...),
    ref_b: str = Form(...),
):
    """Compare two previously uploaded files by their ref_ids."""
    from backend.app.voice_analysis import analyze_pair

    tmp = _temp_dir()
    device = os.environ.get("RVC_DEVICE", "cpu")

    # Locate files — they may have any extension
    def _find(ref_id: str, slot: str) -> str:
        import glob
        matches = glob.glob(os.path.join(tmp, f"{ref_id}_{slot}.*"))
        if not matches:
            raise HTTPException(
                status_code=404,
                detail=f"Audio ref '{ref_id}' (slot {slot}) not found — may have expired",
            )
        return matches[0]

    path_a = _find(ref_a, "a")
    path_b = _find(ref_b, "b")

    try:
        result = analyze_pair(path_a=path_a, path_b=path_b, device=device)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return CompareResponse(**result)


@router.get("/audio/{ref_id}")
async def stream_audio(ref_id: str, slot: str = "a"):
    """Stream a previously uploaded temp audio file for waveform rendering."""
    import glob

    tmp = _temp_dir()
    matches = glob.glob(os.path.join(tmp, f"{ref_id}_{slot}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Audio ref not found or expired")

    path = matches[0]
    ext = os.path.splitext(path)[1].lower()
    media_types = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg",
        ".flac": "audio/flac", ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
    }
    return FileResponse(
        path,
        media_type=media_types.get(ext, "audio/wav"),
        headers={"Cache-Control": "no-cache"},
    )
