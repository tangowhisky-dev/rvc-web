"""FastAPI application entry point."""

import json
import logging
import os
from contextlib import asynccontextmanager

# Fix libomp duplicate-init crash when PyTorch and system OpenMP both load
# (common on macOS / conda). This environment variable must be set BEFORE
# any module imports numpy, torch, or faiss.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.db import init_db
from backend.app.routers import profiles, realtime, training
from backend.app.routers.realtime import ws_router as realtime_ws_router
from backend.app.routers.training import ws_router as training_ws_router

# Configure logging for uvicorn
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rvc_web.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    db_path = await init_db()
    project_root = os.environ.get(
        "PROJECT_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )

    # Ensure samples dir exists
    samples_dir = os.path.join(project_root, "data", "samples")
    os.makedirs(samples_dir, exist_ok=True)

    print(json.dumps({
        "event": "startup",
        "db_path": str(db_path),
        "project_root": project_root,
        "data_dir": samples_dir,
    }), flush=True)
    yield

    # Shutdown
    from backend.app.realtime import manager
    for session_id in list(manager._sessions.keys()):
        try:
            manager.stop_session(session_id)
        except Exception:
            pass

app = FastAPI(title="RVC Studio API", lifespan=lifespan)

# Allow WebSocket and HTTP cross-origin requests from the Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root-level endpoints
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/devices")
async def list_devices():
    import sounddevice as sd
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        result.append({
            "id": i,
            "name": dev["name"],
            "is_input": dev["max_input_channels"] > 0,
            "is_output": dev["max_output_channels"] > 0,
        })
    return result

# Sub-routers
app.include_router(profiles.router)
app.include_router(training.router)
app.include_router(training_ws_router)
app.include_router(realtime.router)
app.include_router(realtime_ws_router)
