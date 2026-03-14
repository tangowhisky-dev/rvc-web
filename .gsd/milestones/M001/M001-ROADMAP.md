# M001: RVC Web App

**Vision:** A polished Next.js + FastAPI application that wraps the RVC v2 pipeline into three clean features: a voice sample library, a fine-tuning training pipeline with live log streaming, and a realtime voice conversion loop with live waveforms — all orchestrated from a browser tab, output routed through BlackHole 2ch.

## Success Criteria

- User can upload a voice sample, see it listed in the library, and select it for training.
- User can trigger fine-tuning and watch preprocessing → feature extraction → training logs stream live in the browser.
- User can start a realtime VC session selecting any input/output device; their voice is converted and plays on the chosen output device.
- Input and output waveforms update live in the browser without UI jank (rendering in Web Worker).
- `./scripts/start.sh` starts both services; `./scripts/stop.sh` stops them cleanly.
- The base pretrained v2 model is never modified; one fine-tuned copy is maintained in `assets/weights/`.

## Key Risks / Unknowns

- **MPS training stability** — RVC training code has CUDA assumptions; MPS may raise device errors on specific ops. May need to force CPU for training.
- **rtrvc.py block size tuning** — Too small = artifacts, too large = unacceptable latency. The right `block_frame` value is empirical and must be found during S03 execution.
- **fairseq import environment** — `rtrvc.py` must run with correct `PYTHONPATH` pointing to `Retrieval-based-Voice-Conversion-WebUI/` and `rmvpe_root` env var set.
- **aiosqlite not yet installed** — Several Python packages still need to be installed in the `rvc` conda env before any backend code runs.

## Proof Strategy

- **MPS training stability** → retire in S02 by running a real preprocess + extract + train cycle on a short sample and confirming `.pth` file is produced without crashes.
- **rtrvc.py block size** → retire in S03 by starting a real session with mic input and confirming audible output on BlackHole 2ch at an acceptable latency.
- **fairseq import environment** → retire in S01 (backend scaffolding) by confirming `rtrvc.py` imports without error in a test route.

## Verification Classes

- Contract verification: pytest tests for API endpoints, SQLite schema, WebSocket message shape
- Integration verification: real training run producing `.pth`; real realtime session outputting to BlackHole 2ch
- Operational verification: `start.sh` / `stop.sh` lifecycle; services restart cleanly
- UAT / human verification: audibly confirm voice conversion quality with target voice

## Milestone Definition of Done

This milestone is complete only when all are true:

- All four slices are complete with passing verification
- A voice sample can be uploaded, trained, and used for realtime VC end-to-end
- The realtime session actually routes audio to BlackHole 2ch (verified by routing BlackHole to speakers)
- Training produces a real `.pth` file (not stub) in `assets/weights/rvc_finetune_active.pth`
- `start.sh` and `stop.sh` work on a clean terminal session
- Waveform canvases animate without freezing the browser UI
- Base pretrained model files are untouched

## Requirement Coverage

- Covers: R001, R002, R003, R004, R005, R006, R007, R008
- Partially covers: none
- Leaves for later: R009 (effects chain), R010 (multi-slot models)
- Orphan risks: none

## Slices

- [x] **S01: Backend Foundation & Voice Sample Library** `risk:high` `depends:[]`
  > After this: FastAPI server runs from `rvc-web/`, SQLite schema initialized, REST API for creating/listing/deleting voice profiles is live, file upload stores audio to disk and records to DB — verified by curl.

- [x] **S02: Training Pipeline with Live Log Streaming** `risk:high` `depends:[S01]`
  > After this: User can POST to start training for a profile, watch preprocess → feature extract → train logs stream via WebSocket in a test client, and find a real `.pth` in `assets/weights/` when training completes.

- [ ] **S03: Realtime Voice Conversion & Waveform Visualization** `risk:high` `depends:[S01,S02]`
  > After this: User opens the Realtime tab in the browser, selects MacBook Pro Microphone → BlackHole 2ch, starts session, speaks, and hears their converted voice on BlackHole output; input/output waveforms animate live in the browser.

- [ ] **S04: Next.js Frontend Polish & Service Scripts** `risk:low` `depends:[S01,S02,S03]`
  > After this: All three features are accessible through a polished, responsive Next.js UI; `./scripts/start.sh` starts everything and opens the browser; `./scripts/stop.sh` stops cleanly; in-app BlackHole setup guide is visible.

## Boundary Map

### S01 → S02

Produces:
- `GET /api/profiles` → `[{id, name, status, created_at, sample_path}]`
- `POST /api/profiles` → create profile with uploaded audio, returns `{id, name, status}`
- `DELETE /api/profiles/:id` → removes profile + files
- `GET /api/profiles/:id` → single profile detail
- SQLite schema: `profiles(id TEXT PK, name TEXT, status TEXT, sample_path TEXT, model_path TEXT, index_path TEXT, created_at TEXT)`
- `data/samples/<profile_id>/` directory convention for audio storage
- FastAPI app module: `backend/app/main.py` with CORS configured for `localhost:3000`
- Env/config pattern: `RVC_ROOT` env var pointing to `Retrieval-based-Voice-Conversion-WebUI/`

Consumes:
- nothing (first slice)

### S01 → S03

Produces (same as S01 → S02, plus):
- `GET /api/devices` → `[{id, name, is_input, is_output}]` (sounddevice query)
- Profile `status` field values: `untrained | training | trained | failed`

### S02 → S03

Produces:
- `POST /api/training/start` body `{profile_id}` → `{job_id}` (triggers training subprocess chain)
- `POST /api/training/cancel` body `{profile_id}` → kills subprocess
- `GET /api/training/status/:profile_id` → `{phase, progress_pct, status}`
- `WS /ws/training/:profile_id` → streams `{type: "log"|"phase"|"done"|"error", message, phase}` JSON lines
- `assets/weights/rvc_finetune_active.pth` — fine-tuned model file produced by training
- `logs/<profile_id>/added_IVF*_Flat*.index` — FAISS index file produced by training
- Profile status updated to `trained` in SQLite when training completes

### S03 → S04

Produces:
- `POST /api/realtime/start` body `{profile_id, input_device_id, output_device_id, pitch, index_rate, protect}` → `{session_id}`
- `POST /api/realtime/stop` body `{session_id}` → stops audio loop
- `POST /api/realtime/params` body `{session_id, pitch?, index_rate?, protect?}` → hot-updates running session params
- `WS /ws/realtime/:session_id` → streams `{type: "waveform_in"|"waveform_out", samples: Float32Array as base64, sr: number}` at ~20fps
- `GET /api/realtime/status` → `{active: bool, session_id, profile_id, input_device, output_device}`
- `AudioProcessor` interface: `process(audio: np.ndarray, sample_rate: int) -> np.ndarray` — stageable pipeline

### S04 → (end)

Produces:
- `scripts/start.sh` — starts backend (conda run -n rvc uvicorn) + frontend (pnpm dev), writes PIDs
- `scripts/stop.sh` — kills by PID files
- Next.js pages: `/` (Library tab), `/training` (Training tab), `/realtime` (Realtime tab) — or single page with tab navigation
- In-app BlackHole setup guide component
