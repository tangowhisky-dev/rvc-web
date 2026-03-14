# GSD State

**Active Milestone:** M001 — RVC Web App
**Active Slice:** S04 — Next.js Frontend Polish & Service Scripts
**Phase:** S03 complete; S04 not started

## Recent Decisions

- D035: Device API schema uses `is_input`/`is_output` booleans (not `kind` field)
- D036: Frontend bootstrapped fresh with Next.js 16 + App Router + Tailwind 4
- D037: Realtime tab includes Voice Profile selector (fetches /api/profiles, filters trained)

## Blockers

None.

## Milestone S01 Status

Complete. All 4 tasks done. pytest 17/17 green.

## Milestone S02 Status

Complete. All 4 tasks done.
- Contract tests: 25/25 passing
- Integration smoke test: 1 epoch in 24.5s, .pth produced, FAISS index built, DB status=trained
- MPS training stability risk retired

## Milestone S03 Status

Complete. All 4 tasks done.
- Contract suite: 35 passed, 2 skipped (both integration tests skip without hardware)
- 10 realtime contract tests: all passing
- Frontend: Next.js 16 at `frontend/`; `/realtime` page renders with device dropdowns, waveform canvases, sliders, Start/Stop
- Integration test exists; skips until trained profile available (run S02 training integration first)
- R003, R004, R005, R006 all validated

## S04 Scope Reminder

- Polished Next.js UI: Library tab (`/`), Training tab (`/training`), Realtime tab (`/realtime`) under unified layout
- `scripts/start.sh` and `scripts/stop.sh`
- In-app BlackHole setup guide

## Next Action

Begin S04: Next.js Frontend Polish & Service Scripts.
