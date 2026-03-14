# Project

## What This Is

A realtime voice cloning web application built on top of the RVC (Retrieval-based Voice Conversion) v2 model. The system lets a user speak into a microphone and hear their voice converted to a trained target voice with minimal latency — routed through BlackHole 2ch for system-wide audio integration on macOS.

The app has three integrated capabilities:
1. **Voice Sample Library** — upload, record, name, and manage audio samples stored in SQLite.
2. **Training Pipeline** — fine-tune a copy of the base pretrained v2 model on a selected voice sample (preprocess → feature extract → train), streaming logs live via WebSocket. The base model is never overwritten; one fine-tuned copy is maintained per active profile.
3. **Realtime Voice Conversion** — a Python sounddevice audio loop reads from the selected input device, runs RVC inference (using `rtrvc.py`), and writes to the selected output device (BlackHole 2ch). The browser controls start/stop, displays input/output waveforms rendered in a Web Worker, and lets the user tune pitch, index rate, and protect parameters in real time.

## Core Value

User speaks into their mic → RVC converts their voice to the cloned target voice → output lands on BlackHole → any app that listens to BlackHole hears the converted voice.

## Current State

Bootstrapping. The upstream RVC repo (`Retrieval-based-Voice-Conversion-WebUI`) is cloned at `Retrieval-based-Voice-Conversion-WebUI/` and all its Python dependencies are installed in the `rvc` conda environment. Pretrained v2 model weights, HuBERT, and RMVPE are confirmed downloaded. No application code yet.

## Architecture / Key Patterns

- **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS — runs on port 3000
- **Backend**: FastAPI (Python, `rvc` conda env) — runs on port 8000
- **Audio Engine**: Python `sounddevice` loop in a background thread, driven by `infer/lib/rtrvc.py`
- **Training**: Subprocesses — `infer/modules/train/preprocess.py`, `extract_feature_print.py`, `train.py` — launched from FastAPI, log-streamed via WebSocket
- **Database**: SQLite via `aiofiles` + raw SQL (no ORM) — simple, zero-setup
- **Audio routing (macOS)**: BlackHole 2ch as virtual output device; user sets their DAW/app to use BlackHole as mic input
- **Waveform rendering**: Web Worker + OffscreenCanvas — never blocks the main thread
- **Service orchestration**: `scripts/start.sh` and `scripts/stop.sh` — starts conda backend + Next.js frontend

## Capability Contract

See `.gsd/REQUIREMENTS.md` for the explicit capability contract, requirement status, and coverage mapping.

## Milestone Sequence

- [ ] M001: RVC Web App — Voice samples, fine-tuning pipeline, and realtime VC with waveforms in a polished Next.js UI
