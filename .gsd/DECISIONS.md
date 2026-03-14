# Decisions Register

<!-- Append-only. Never edit or remove existing rows.
     To reverse a decision, add a new row that supersedes it.
     Read this file at the start of any planning or research phase. -->

| # | When | Scope | Decision | Choice | Rationale | Revisable? |
|---|------|-------|----------|--------|-----------|------------|
| D001 | M001 | arch | Database | SQLite via aiosqlite | PostgreSQL not installed; zero-setup; single-user local tool; trivially swappable later | Yes — if multi-user deployment needed |
| D002 | M001 | arch | Backend framework | FastAPI + uvicorn | Already in rvc conda env; async support for WebSocket log streaming; minimal overhead | No |
| D003 | M001 | arch | Frontend framework | Next.js 14 App Router + TypeScript + Tailwind | User specified Next.js; App Router is current standard | No |
| D004 | M001 | arch | Audio device routing | sounddevice (Python) loop → BlackHole 2ch output | User confirmed backend-only audio processing; browser only controls start/stop and displays waveforms | No |
| D005 | M001 | arch | Waveform rendering | Web Worker + OffscreenCanvas | Keeps main thread free; explicit user requirement | No |
| D006 | M001 | arch | Realtime inference | rtrvc.py RVC class directly (no copy) | Upstream code is well-maintained; wrapping avoids drift; run from RVC_ROOT working dir | No |
| D007 | M001 | arch | Fine-tuned model storage | Single copy at assets/weights/rvc_finetune_active.pth | User explicitly requested one copy; overwrite on re-train. Base pretrained models are never touched. | Yes — expand to per-profile slots in M002 |
| D008 | M001 | pattern | Audio processor pipeline | List of AudioProcessor objects (process: np.ndarray → np.ndarray) | Explicit user request for future effects extensibility; simple enough to not over-engineer now | Yes — could grow to a graph if needed |
| D009 | M001 | arch | Training subprocess pattern | Launch preprocess.py / extract_feature_print.py / train.py as subprocess, stream stdout via async queue | Matches how web.py orchestrates training; log lines available for WebSocket streaming | No |
| D010 | M001 | constraint | Training device | Try MPS first; fallback to CPU if MPS raises on training ops | MPS available (torch 2.10, mps.is_available=True) but training ops may be unsupported; inference uses MPS | Yes — revisit after S02 testing |
| D011 | M001 | constraint | Service startup | conda run -n rvc uvicorn backend.app.main:app for backend; pnpm dev for frontend | rvc conda env must be active for all Python deps; conda run activates inline | No |
| D012 | M001 | arch | Project structure | rvc-web/ contains backend/ and frontend/ at root alongside Retrieval-based-Voice-Conversion-WebUI/ | Keeps RVC repo self-contained; RVC_ROOT env var bridges backend to RVC code | No |
