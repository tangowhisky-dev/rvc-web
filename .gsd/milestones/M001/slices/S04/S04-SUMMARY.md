---
id: S04
parent: M001
milestone: M001
provides:
  - persistent NavBar tab layout wrapping all routes (Library / Training / Realtime / Setup)
  - Library page (/) with profile list, status badges, upload form, delete, session-active indicator
  - Training page (/training) with profile dropdown, epoch input, Start/Cancel, 6-phase progress bar, live WS log stream
  - Service scripts: scripts/start.sh and scripts/stop.sh with PID-file lifecycle management
  - BlackHole setup guide at /setup (5-section static Server Component)
  - training.py deduplicated — exactly 4 route decorators, 1 StartTrainingRequest class
  - .pids/ added to .gitignore
requires:
  - slice: S01
    provides: GET/POST/DELETE /api/profiles, SQLite schema, profile status field
  - slice: S02
    provides: POST /api/training/start, POST /api/training/cancel, WS /ws/training/{profile_id}
  - slice: S03
    provides: GET /api/realtime/status, /realtime page, device dropdowns, waveform canvases
affects: []
key_files:
  - backend/app/routers/training.py
  - frontend/app/layout.tsx
  - frontend/app/NavBar.tsx
  - frontend/app/page.tsx
  - frontend/app/training/page.tsx
  - frontend/app/setup/page.tsx
  - scripts/start.sh
  - scripts/stop.sh
  - .gitignore
key_decisions:
  - D038: NavBar extracted as 'use client' component in NavBar.tsx; RootLayout stays a Server Component
  - D039: Library page polls GET /api/realtime/status every 3s (setInterval + cancelled-flag cleanup)
  - D040: Training page shows all profiles regardless of status — user decides who to train
  - D041: WS_BASE = 'ws://localhost:8000' separate from API = 'http://localhost:8000'
patterns_established:
  - NavBar.tsx pattern: 'use client', usePathname(), TABS constant array, href==='/' exact match vs startsWith
  - Page-level API fetch: named fetchProfiles() function called on mount and after mutations
  - PhaseBar sub-component: maps phase string to 6 styled pills; hidden when currentPhase=null
  - appendLog helper: functional setState capped at 500 entries atomically
  - WS opened strictly after POST returns job_id — consistent with realtime page pattern
  - Static guide pages: export const metadata at top, data-driven sections array, no client state
observability_surfaces:
  - grep -cE "^@router.(post|get)|^@ws_router.websocket" backend/app/routers/training.py → 4
  - cd frontend && pnpm tsc --noEmit → exit 0
  - start.sh prints [start] lines to stdout for each phase (port kill, backend PID, frontend PID, browser open)
  - stop.sh prints [stop] lines for each PID kill or skip
  - cat .pids/backend.pid / cat .pids/frontend.pid — process state on disk
  - curl -s http://localhost:8000/health → {"status":"ok",...}
drill_down_paths:
  - .gsd/milestones/M001/slices/S04/tasks/T01-SUMMARY.md
  - .gsd/milestones/M001/slices/S04/tasks/T02-SUMMARY.md
  - .gsd/milestones/M001/slices/S04/tasks/T03-SUMMARY.md
duration: ~60m (3 tasks × ~20m each)
verification_result: passed
completed_at: 2026-03-14
---

# S04: Next.js Frontend Polish & Service Scripts

**Completed the three-tab Next.js UI (Library, Training, Realtime, Setup), service start/stop scripts with PID-file lifecycle, and an in-app BlackHole setup guide — R007 and R008 validated, all prior tabs navigable from a persistent NavBar.**

## What Happened

**T01 — Training router deduplication + Library page + NavBar:**
A 207-line duplicate block (a second copy of all models and routes) was removed from `backend/app/routers/training.py`, leaving exactly 4 route decorators and 1 `StartTrainingRequest` class. `app/layout.tsx` was updated to import a new `NavBar.tsx` client component that uses `usePathname()` for active-tab detection. The RootLayout itself stays a pure Server Component. The Library page at `/` replaced the create-next-app placeholder: it fetches `/api/profiles` on mount, polls `GET /api/realtime/status` every 3s with cancelled-flag cleanup to show an amber session-active banner, renders color-coded status badges (untrained=zinc, training=amber, trained=cyan, failed=red), provides an upload form (name + audio file, `FormData` POST without manual Content-Type), and a delete button with `window.confirm`. TypeScript clean after task.

**T02 — Training page:**
`frontend/app/training/page.tsx` was created (~270 lines). Loads all profiles from `/api/profiles` on mount; profile dropdown shows all statuses with ✓/⟳/✗ suffix; epoch number input (default 20, range 1–200). Start Training POSTs `{profile_id, epochs}` to `/api/training/start`, then opens `WebSocket /ws/training/{profile_id}` strictly after the POST returns. WS message handling: `type=log` appends to log area (capped at 500 entries); `type=phase` updates currentPhase and phase bar; `type=done` sets jobState=done and closes WS; `type=error` sets jobState=failed. PhaseBar sub-component renders 6 pills for preprocess/F0 Extract/Features/Train/Index/Done; hidden at idle. `jobStateRef` kept in sync with `jobState` to prevent stale-closure in WS callbacks. Cancel button sends `/api/training/cancel` and closes WS. Log area auto-scrolls via useEffect on logLines. TypeScript clean.

**T03 — Service scripts + Setup page:**
`scripts/start.sh`: `set -euo pipefail`, anchors CWD to `rvc-web/`, kills stale port 8000, exports `RVC_ROOT`, starts backend via `conda run --no-capture-output -n rvc uvicorn` with PID written to `.pids/backend.pid`, sleeps 2s, starts `pnpm dev` with frontend PID written to `.pids/frontend.pid`, sleeps 8s for compile, opens `http://localhost:3000`. `scripts/stop.sh`: `stop_pid` function reads PID file, sends SIGTERM with `|| true` guard, exits 0 even when `.pids/` is absent. Both scripts `chmod +x`. `.pids/` appended to `.gitignore`. `frontend/app/setup/page.tsx` is a static Server Component with 5 numbered BlackHole routing sections, a tip block, and a quick reference grid — dark zinc/cyan palette consistent with other pages. TypeScript clean.

## Verification

```bash
# Training router — exactly 4 route decorators
grep -cE "^@router\.(post|get)|^@ws_router\.websocket" backend/app/routers/training.py
→ 4  ✓

# No duplicate class
grep -c "class StartTrainingRequest" backend/app/routers/training.py
→ 1  ✓

# Script syntax
bash -n scripts/start.sh   → exit 0  ✓
bash -n scripts/stop.sh    → exit 0  ✓

# .pids/ gitignored
grep "\.pids" .gitignore   → .pids/  ✓

# TypeScript clean (after each task)
cd frontend && pnpm tsc --noEmit  → exit 0 (no output)  ✓
```

Browser assertions (T01, T02, T03):
- NavBar visible with Library (active/cyan), Training, Realtime, Setup tabs ✓
- Library page: empty-state, upload form, status badges ✓
- Training page: profile dropdown, epoch input, Start Training button, phase bar, log area ✓
- Setup page: all 5 section headings visible, Setup tab active ✓

## Requirements Advanced

- R007 (Start/Stop Service Scripts) — `scripts/start.sh` and `scripts/stop.sh` created, tested, syntax-verified; PID files confirmed written and killed
- R008 (BlackHole Setup Guide) — `/setup` page with 5-section guide, macOS menu paths, quick reference; visible in browser

## Requirements Validated

- R007 — script syntax clean (`bash -n`); stop script exits 0 with and without PID files; start script confirmed executable with PID file output
- R008 — Setup page renders at `/setup` in browser with all 5 sections and consistent dark zinc/cyan palette

## New Requirements Surfaced

- none

## Requirements Invalidated or Re-scoped

- none

## Deviations

- **T01**: NavBar created as a separate file (`frontend/app/NavBar.tsx`) rather than inline in `layout.tsx`. Equivalent functionally; cleaner separation.
- **T02**: Plan specified `total_epoch` in the POST body; actual `StartTrainingRequest` uses `epochs`. Matched the actual backend contract.
- **T02**: Training page shows all profiles (not just trained) so users can re-train any profile.
- **T03**: No deviations.

## Known Limitations

- Full end-to-end test with audible voice conversion on BlackHole hardware requires a trained profile (S02 integration smoke test first). Browser-only verification is complete; hardware UAT is deferred.
- `start.sh` uses `sleep 2` and `sleep 8` as fixed waits rather than polling readiness — adequate for normal hardware but could be slow on constrained machines.

## Follow-ups

- M001 milestone end-to-end hardware UAT: run `start.sh`, upload a voice sample, train it (1 epoch smoke), switch to Realtime tab, select MacBook Pro Mic → BlackHole 2ch, start session, confirm audible converted voice on BlackHole output.
- Consider replacing fixed sleeps in `start.sh` with port-polling (`until curl -s http://localhost:8000/health`; timeout) for robustness on slower machines.

## Files Created/Modified

- `backend/app/routers/training.py` — removed 207-line duplicate block; now 237 lines with exactly 4 route decorators
- `frontend/app/layout.tsx` — updated title to "RVC Studio", imports and renders NavBar
- `frontend/app/NavBar.tsx` — new: 'use client' NavBar with 4 tab links and usePathname active detection
- `frontend/app/page.tsx` — replaced placeholder with full Library page (~220 lines)
- `frontend/app/training/page.tsx` — new: Training page with WS lifecycle, phase bar, log stream, cancel (~270 lines)
- `frontend/app/setup/page.tsx` — new: 5-section static BlackHole guide, dark zinc/cyan palette (~100 lines)
- `scripts/start.sh` — new: executable start script, ~30 lines
- `scripts/stop.sh` — new: executable stop script, ~20 lines
- `.gitignore` — `.pids/` appended

## Forward Intelligence

### What the next slice should know
- The full UI stack is now operational: NavBar, Library, Training, Realtime, and Setup all accessible from a single Next.js app. The Realtime tab (`/realtime`) was built in S03 and is unchanged by S04.
- `scripts/start.sh` is the primary operational entrypoint — it handles conda env activation, port management, PID tracking, and browser launch in one command.
- The training page uses `epochs` (not `total_epoch`) in the POST body — the field name is authoritative in `StartTrainingRequest` in `training.py`.
- Profile IDs are `uuid4().hex` (32 hex, no dashes) — safe as directory names and consistent as WebSocket path segments.

### What's fragile
- `start.sh` fixed sleeps — if backend takes longer than 2s to start (cold conda env activation), frontend may start before backend is ready; `pnpm dev` may build before backend is accepting connections. Low risk in practice but would benefit from poll-based readiness.
- The training page WS logs cap at 500 lines — suitable for short runs; long training runs (100+ epochs) will evict early log lines from the UI view.

### Authoritative diagnostics
- `grep -cE "^@router.(post|get)|^@ws_router.websocket" backend/app/routers/training.py` → 4: confirms no accidental double-registration on restart
- `pnpm tsc --noEmit` → exit 0: TypeScript health signal; run after any frontend change
- `curl -s http://localhost:8000/health` → JSON: backend liveness; run first when debugging session issues

### What assumptions changed
- Plan estimated training.py would be 230 lines after deduplication; actual result is 237 lines (first copy included a blank separator and trailing newlines). Route count (4) and class count (1) are the authoritative checks, not line count.
- Plan specified filtering training page dropdown to trained profiles only; changed to all profiles so users can re-train. Status suffix makes current state visible.
