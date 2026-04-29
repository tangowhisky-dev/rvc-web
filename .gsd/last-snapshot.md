# GSD context snapshot (2026-04-29T04:57:33.022Z)

## Top project memories
- [MEM001] (architecture) Database Chose: SQLite via aiosqlite. Rationale: PostgreSQL not installed; zero-setup; single-user local tool; trivially swappable later.
- [MEM002] (architecture) Backend framework Chose: FastAPI + uvicorn. Rationale: Already in rvc conda env; async support for WebSocket log streaming; minimal overhead.
- [MEM003] (architecture) Frontend framework Chose: Next.js 14 App Router + TypeScript + Tailwind. Rationale: User specified Next.js; App Router is current standard.
- [MEM004] (architecture) Audio device routing Chose: sounddevice (Python) loop → BlackHole 2ch output. Rationale: User confirmed backend-only audio processing; browser only controls start/stop and displays waveforms.
- [MEM005] (architecture) Waveform rendering Chose: Web Worker + OffscreenCanvas. Rationale: Keeps main thread free; explicit user requirement.
- [MEM006] (architecture) Realtime inference Chose: rtrvc.py RVC class directly (no copy). Rationale: Upstream code is well-maintained; wrapping avoids drift; run from RVC_ROOT working dir.
