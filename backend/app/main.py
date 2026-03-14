"""FastAPI application entry point."""

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.db import DB_PATH, init_db
from backend.app.routers.devices import router as devices_router
from backend.app.routers.profiles import router as profiles_router

logger = logging.getLogger("rvc_web")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Startup:
      - Initializes SQLite schema (idempotent)
      - Creates required data directories
      - Emits a structured JSON startup log line to stdout
    """
    await init_db()

    startup_event = {
        "event": "startup",
        "db_path": DB_PATH,
        "rvc_root": os.environ.get("RVC_ROOT", "not set"),
        "data_dir": "data/samples",
    }
    print(json.dumps(startup_event), flush=True)

    yield
    # No explicit teardown needed — aiosqlite connections are per-request


app = FastAPI(title="RVC Web API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Liveness check. Returns DB path and RVC_ROOT for configuration confirmation."""
    return {
        "status": "ok",
        "db_path": DB_PATH,
        "rvc_root": os.environ.get("RVC_ROOT", "not set"),
    }


# Routers
app.include_router(profiles_router)
app.include_router(devices_router)
