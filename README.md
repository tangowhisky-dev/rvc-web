# rvc-web

Realtime voice cloning web app with two training pipelines: **RVC v2** and **Beatrice 2**.

## Quick Start

```bash
./start.sh   # starts backend (port 8000) + frontend (port 3000)
./stop.sh    # clean shutdown
```

## Pipelines

| | RVC v2 | Beatrice 2 |
|---|---|---|
| Hardware | CPU / MPS / CUDA | CUDA required |
| Training unit | Epochs | Steps |
| Quality signal | loss_mel | UTMOS (MOS score) |
| Best model skip | First 50 epochs | First 1000 steps |

## Crash Diagnosis

Training subprocess output is written to a per-profile log file — not streamed to the UI. If training fails silently, check the log:

```bash
# RVC
cat data/profiles/<profile_id>/train.log

# Beatrice 2
cat data/profiles/<profile_id>/beatrice2/train.log
```

## Directory Layout

```
assets/          — pretrained weights (hubert, rmvpe, beatrice2)
backend/app/     — FastAPI app, DB, training orchestration
backend/beatrice2/ — Beatrice 2 pipeline + trainer
backend/rvc/     — RVC inference + training code
data/profiles/   — per-profile audio, checkpoints, logs
data/rvc.db      — SQLite database
frontend/        — Next.js (App Router), TypeScript, Tailwind
logs/            — RVC training workspace
```

## Environment

`PROJECT_ROOT` is the single source of truth for asset and data paths. Set automatically by `start.sh`.
