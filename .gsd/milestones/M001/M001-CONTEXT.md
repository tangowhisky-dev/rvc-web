# M001: RVC Web App — Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

## Project Description

A realtime voice cloning system with a polished Next.js web UI backed by a FastAPI Python service. The Python backend wraps the upstream RVC (Retrieval-based Voice Conversion) v2 pipeline — specifically `infer/lib/rtrvc.py` for realtime inference — and exposes REST + WebSocket APIs. Three user-facing features: voice sample library, model fine-tuning pipeline, and realtime voice conversion with live waveforms.

## Why This Milestone

The user has the RVC repo cloned, conda environment ready, and all model weights downloaded. The missing piece is a clean application layer that orchestrates the existing RVC machinery through a real UI rather than the Gradio-based web.py. This milestone delivers the complete end-to-end system.

## User-Visible Outcome

### When this milestone is complete, the user can:

- Open `http://localhost:3000`, upload a voice sample (e.g. a 5-minute recording of a target voice), and see it saved in the library.
- Select that profile, click "Train", watch the three-phase pipeline run with streamed logs, and see the model transition from "untrained" to "trained".
- Go to the Realtime tab, select MacBook Pro Microphone as input and BlackHole 2ch as output, click Start — and hear their voice converted to the target voice in real time through any app that uses BlackHole as its mic.
- See live input/output waveforms updating smoothly in the browser while speaking.

### Entry point / environment

- Entry point: `./scripts/start.sh` → opens `http://localhost:3000`
- Environment: local macOS dev (Apple Silicon, MPS), `rvc` conda env for backend
- Live dependencies: `sounddevice` (system audio), SQLite (local file), RVC model files in `Retrieval-based-Voice-Conversion-WebUI/`

## Completion Class

- Contract complete means: API endpoints return expected shapes, WebSocket delivers log lines and waveform data, SQLite records created/queried correctly.
- Integration complete means: realtime session actually converts audio through BlackHole; training actually produces a `.pth` weights file in `logs/<profile>/weights/`.
- Operational complete means: `start.sh` / `stop.sh` work cleanly; services survive restart.

## Final Integrated Acceptance

To call this milestone complete, we must prove:

- Upload a voice sample → train → start realtime session → speak → confirm audio output lands on BlackHole 2ch (verified by routing BlackHole to system speakers or monitoring app).
- Training completes for a short voice sample (≥60 sec audio) and produces a `.pth` file that loads without error.
- Waveform canvases animate without freezing the browser tab (verified by interacting with UI while session is active).

## Risks and Unknowns

- **MPS training stability** — RVC training on MPS (`torch.backends.mps.is_available() = True`) may hit device-specific bugs in upstream code. Fallback: force `device=cpu` for training, use `mps` for inference only.
- **rtrvc.py block latency** — Block size directly controls latency vs quality tradeoff. Need to tune `block_frame` during S03 execution; too small → artifacts, too large → unacceptable delay.
- **fairseq import path** — `rtrvc.py` uses `fairseq.checkpoint_utils.load_model_ensemble_and_task` with a hardcoded relative path. Must run backend from the `Retrieval-based-Voice-Conversion-WebUI/` directory or set `PYTHONPATH` and `rmvpe_root` env vars correctly.
- **SQLite concurrency** — FastAPI is async; SQLite with `aiofiles`+raw SQL via `aiosqlite` handles concurrent reads safely. Writes (training state updates) are serialized via a single writer pattern.
- **websockets package missing** — `websockets` and `sqlalchemy`/`aiosqlite` are not yet in the `rvc` conda env. Must install before running.

## Existing Codebase / Prior Art

- `Retrieval-based-Voice-Conversion-WebUI/infer/lib/rtrvc.py` — The core realtime inference class `RVC`. Import this directly; do not copy it. The constructor loads HuBERT + synthesizer; `RVC.infer()` processes one audio block.
- `Retrieval-based-Voice-Conversion-WebUI/infer/modules/vc/modules.py` — `VC` class for batch/single file inference. Not used for realtime but useful reference.
- `Retrieval-based-Voice-Conversion-WebUI/infer/modules/train/preprocess.py` — Launched as subprocess with args: `inp_root sr n_p exp_dir noparallel per`
- `Retrieval-based-Voice-Conversion-WebUI/infer/modules/train/extract_feature_print.py` — Launched as subprocess.
- `Retrieval-based-Voice-Conversion-WebUI/infer/modules/train/train.py` — Launched via `click_train()` pattern in `web.py`.
- `Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2/` — Contains `f0G48k.pth`, `f0D48k.pth` (base pretrained models). **Never overwrite these.**
- `Retrieval-based-Voice-Conversion-WebUI/assets/weights/` — Where fine-tuned `.pth` files land after `extract_small_model`. Our fine-tuned copy: `assets/weights/rvc_finetune_active.pth`.
- `Retrieval-based-Voice-Conversion-WebUI/configs/config.py` — `Config` singleton; sets device, half precision, python_cmd.

> See `.gsd/DECISIONS.md` for all architectural and pattern decisions.

## Relevant Requirements

- R001 — Voice sample library (S01 is the primary owner)
- R002 — Training pipeline (S02 is the primary owner)
- R003 — Realtime VC (S03 is the primary owner)
- R004 — Waveform visualization (S03)
- R005 — Audio device selection (S03)
- R006 — Modular audio processing chain (S03)
- R007 — Start/stop scripts (S04)
- R008 — BlackHole setup guide (S04)

## Scope

### In Scope

- FastAPI backend: REST API for profiles/samples/training, WebSocket for training logs and waveform data
- Next.js frontend: 3-tab UI (Library, Training, Realtime)
- SQLite database via `aiosqlite`
- RVC fine-tuning pipeline (preprocess → extract → train → extract_small_model) as background subprocess, log-streamed
- Realtime audio loop using `sounddevice` + `rtrvc.py`, output to user-selected device
- Live waveform rendering in Web Worker + OffscreenCanvas
- `start.sh` / `stop.sh` scripts
- In-app BlackHole setup instructions

### Out of Scope / Non-Goals

- Audio effects / post-processing chain (M002)
- Multiple fine-tuned model slots (M002)
- CUDA / Windows support
- Authentication / multi-user

## Technical Constraints

- Backend must run from `Retrieval-based-Voice-Conversion-WebUI/` working directory (or with correct `PYTHONPATH`/env vars) so `rtrvc.py` relative paths resolve.
- `rmvpe_root` env var must be set to `assets/rmvpe` before importing `rtrvc.py`.
- Use `aiosqlite` (not psycopg2/postgres) — zero setup, already compatible with `aiofiles` in env.
- Training device: try `mps` first, fallback to `cpu` if MPS errors occur.
- Inference device: `mps` (fast on Apple Silicon).
- Sample rate for RVC: 48k (v2 model, f0-enabled).
- Node.js 25.x, pnpm 10.x available.
- `psycopg2` / `sqlalchemy` / `websockets` / `aiosqlite` need to be pip-installed into `rvc` env.

## Integration Points

- `sounddevice` — macOS Core Audio bridge; device 1 = BlackHole 2ch, device 2 = MacBook Pro Microphone (indexes may shift if devices change — always query dynamically)
- BlackHole 2ch — virtual audio device; RVC writes output here; user routes this to target app
- `fairseq` — loaded by `rtrvc.py`; already in `rvc` env
- `faiss-cpu` — FAISS index search; already in `rvc` env
- RVC `logs/<exp_name>/` — training artifacts directory

## Open Questions

- **Block frame size for realtime**: Start with `block_frame=2048` at 48kHz (≈43ms). Tune during S03 verification. The `rtrvc.py` `infer()` signature requires `block_frame_16k` (at 16kHz), `skip_head`, and `return_length`. — *Need empirical tuning.*
- **Training cancellation**: When the user cancels, we kill the subprocess by PID. The training `train.py` does not have a clean checkpoint-on-cancel path by default. — *Best effort: kill subprocess, leave partial logs.*
