"""Audio devices router.

GET /api/devices — returns the system audio device list as enumerated by
sounddevice (PortAudio). Called synchronously; query_devices() is confirmed
safe and completes in <1ms.

Schema contract (S01-PLAN.md boundary):
  [{id: int, name: str, is_input: bool, is_output: bool}, ...]
"""

from typing import List

import sounddevice as sd
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api")


class DeviceOut(BaseModel):
    id: int
    name: str
    is_input: bool
    is_output: bool


@router.get("/devices", response_model=List[DeviceOut])
def get_devices() -> List[DeviceOut]:
    """Return all system audio devices with their capabilities.

    Device IDs are stable integer indices assigned by PortAudio.  Names can
    contain non-ASCII characters so integer indices are used as IDs rather
    than names.

    Observability: if PortAudio is unavailable this will raise a
    sounddevice.PortAudioError which FastAPI propagates as HTTP 500 with the
    PortAudio error string — explicit and actionable.
    """
    devices = sd.query_devices()
    return [
        DeviceOut(
            id=i,
            name=d["name"],
            is_input=d["max_input_channels"] > 0,
            is_output=d["max_output_channels"] > 0,
        )
        for i, d in enumerate(devices)
    ]
