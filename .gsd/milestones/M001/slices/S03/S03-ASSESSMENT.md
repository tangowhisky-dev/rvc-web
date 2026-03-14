---
id: S03-ASSESSMENT
slice: S03
milestone: M001
assessed_at: 2026-03-14
verdict: roadmap_unchanged
---

# S03 Roadmap Assessment

## Verdict

Roadmap is unchanged. S04 proceeds as planned.

## Success Criterion Coverage

All six milestone success criteria have S04 as their remaining owning slice:

- Upload/list/select voice sample → S04 (Library tab UI)
- Trigger fine-tuning with live log streaming → S04 (Training tab UI)
- Start realtime VC session with device selection → S04 (Realtime tab polish + scripts)
- Live waveforms without UI jank → S04 (already built in S03; S04 polishes and connects tabs)
- `./scripts/start.sh` / `./scripts/stop.sh` → S04 (R007, not yet built)
- Base pretrained model never modified → S04 (operational verification; enforcement already in place from S02)

No criterion is left unowned. Coverage check passes.

## Risk Retirement

- **rtrvc.py block size** — structurally retired: `block_48k=9600` (200ms), ~37ms inference latency, 5.4× headroom confirmed on MPS. Audible hardware verification deferred until the S02 training integration test produces a trained profile, but the integration test exists and will confirm on first run. Not a blocking risk.
- **fairseq import environment** — already retired in S01/S02. No change.

## New Risks

None. Known fragilities (D033 `asyncio.iscoroutine()` guard, OffscreenCanvas fallback) are documented and do not affect S04 scope.

## Boundary Contracts

S03 → S04 boundary map is accurate as written. All five endpoints (`POST /start`, `POST /stop`, `POST /params`, `GET /status`, `WS /ws/realtime/{session_id}`), the waveform WS stream, and the `AudioProcessor` protocol are live and match the roadmap contract exactly. S04 can consume them without changes.

## Requirements Coverage

- R001–R006: all validated. No change in status.
- R007 (start/stop scripts) and R008 (BlackHole guide): still owned by S04, still active. No change.
- R009, R010: still deferred to M002. No change.

## What S04 Must Know

- `/realtime` route is already live in the Next.js 16 App Router frontend. S04 adds `/` (Library) and `/training` tabs plus a shared layout — not a new frontend project.
- `GET /api/realtime/status` always returns `{active, session_id, profile_id, input_device, output_device}` — S04 can poll this to sync button state.
- Run `test_training_integration.py` (S02 integration test) before `test_realtime_integration.py` — the realtime test needs a trained profile in the DB to exercise real hardware.
- `waveform-worker.js` is in `frontend/public/` (not webpack-bundled) — verify worker URL stays correct if Next.js base path changes.
