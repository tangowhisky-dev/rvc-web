# Project

## What This Is

A realtime voice cloning web application with two training pipelines: **RVC v2** (proven, fast) and **Beatrice 2** (higher quality, slower). Users train a voice model from uploaded audio samples, then use it for realtime voice conversion routed through a virtual audio device.

## Current State

Fully operational dual-pipeline system. Both RVC and Beatrice 2 training, inference, and realtime conversion work end-to-end.

**Active capabilities:**
- Voice profile management (create, delete, upload samples, record)
- RVC v2 training — preprocess → extract features → train; epoch loss chart; best-epoch tracking (skips first 50 epochs)
- Beatrice 2 training — preprocess → train (single subprocess); step loss chart (every 10 steps); UTMOS evaluation every 500 steps; best checkpoint skips first 1000 steps; latest = best during warmup
- Realtime voice conversion — sounddevice audio loop, RVC or Beatrice 2 inference, waveform viz via Web Worker
- Export trained Beatrice 2 models
- Voice analysis (speaker embedding comparison)

## Architecture

```
rvc-web/
  assets/              ← pretrained weights (hubert, rmvpe, beatrice2 pretrained)
  backend/
    app/               ← FastAPI app (main.py, db.py, training.py, realtime.py)
    app/routers/       ← profiles, training, realtime, devices, analysis, offline
    beatrice2/         ← Beatrice 2 pipeline (training.py, db_writer.py, inference.py)
    beatrice2/beatrice_trainer/  ← trainer source (train/loop.py is the main loop)
    rvc/               ← RVC Python code; assets/ and logs/ are symlinks into project root
  data/
    profiles/          ← per-profile audio, checkpoints, model files
    rvc.db             ← SQLite database
  frontend/            ← Next.js (App Router), TypeScript, Tailwind
  logs/                ← training workspace (rvc_finetune_active/)
  start.sh / stop.sh   ← always use these, never raw uvicorn/pnpm
```

**Stack:** FastAPI + aiosqlite (port 8000), Next.js 14 (port 3000), SQLite, sounddevice, Python subprocess for training.

## Key Files

| File | Role |
|------|------|
| `backend/app/db.py` | Schema, migrations, `get_db()`, `DB_PATH` |
| `backend/app/training.py` | RVC training orchestration, `_poll_rvc_epochs` |
| `backend/beatrice2/training.py` | Beatrice 2 orchestration, `_poll_beatrice_steps`, `beatrice_manager` |
| `backend/beatrice2/db_writer.py` | `BeatriceDBWriter` — sync sqlite3 writer called from training subprocess |
| `backend/beatrice2/beatrice_trainer/train/loop.py` | Beatrice 2 training loop; calls `_db_writer.insert_step_loss()` every 10 steps |
| `backend/app/routers/training.py` | Training REST + WS endpoints; `BeatriceStepPoint` model; `get_losses()` |
| `frontend/app/training/page.tsx` | Training UI — RVC epoch chart + Beatrice step chart + WS event handling |
| `frontend/app/realtime/page.tsx` | Realtime VC UI |

## Database Schema (key tables)

- `profiles` — id, name, pipeline ("rvc"/"beatrice2"), status, n_steps, etc.
- `epoch_losses` — RVC training: profile_id, epoch, loss_mel, loss_gen, loss_kl, loss_fm, loss_disc, best_epoch
- `beatrice_steps` — Beatrice training: profile_id, step, loss_mel, loss_loud, loss_ap, loss_adv, loss_fm, loss_d, utmos, is_best
- `audio_files` — uploaded/recorded samples per profile
