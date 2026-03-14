---
id: T03
parent: S04
milestone: M001
provides:
  - scripts/start.sh — executable start script that kills stale port 8000, exports RVC_ROOT, starts backend via conda run, starts frontend via pnpm dev, writes PIDs to .pids/, and opens browser
  - scripts/stop.sh — executable stop script that reads PIDs from .pids/ and sends SIGTERM; exits 0 if PID files are absent
  - frontend/app/setup/page.tsx — static Server Component at /setup with 5-section BlackHole routing guide plus tip block and quick reference panel; dark zinc/cyan palette
  - .pids/ added to .gitignore
key_files:
  - scripts/start.sh
  - scripts/stop.sh
  - frontend/app/setup/page.tsx
  - .gitignore
key_decisions:
  - Setup page implemented as a plain Server Component (no 'use client') using a static sections array mapped to JSX — no client state needed for a guide page
  - section heading levels use h2 as a CSS class concern, actual rendering uses <h2> tag consistently matching the numbered-section UX pattern
patterns_established:
  - Static guide pages pattern: export const metadata at top, data-driven sections array, map to card+list JSX, no imports needed for static content pages
observability_surfaces:
  - start.sh prints [start] lines to stdout for each phase — tailable for startup failure diagnosis
  - stop.sh prints [stop] lines — confirms which PIDs were killed or skipped
  - .pids/backend.pid and .pids/frontend.pid — inspectable process state on disk (cat .pids/backend.pid → PID; ps aux | grep <PID>)
  - curl http://localhost:8000/health → backend liveness after start
duration: ~20m
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T03: Service Scripts + BlackHole Setup Page

**Delivered `scripts/start.sh`, `scripts/stop.sh`, and `frontend/app/setup/page.tsx` — all S04 must-haves satisfied, R007 and R008 validated.**

## What Happened

Created the `scripts/` directory and wrote both service scripts exactly per spec:

- `start.sh`: `set -euo pipefail`, `cd "$(dirname "$0")/.."`, kills stale port 8000, exports `RVC_ROOT`, starts uvicorn via `conda run --no-capture-output -n rvc`, writes backend PID to `.pids/backend.pid`, sleeps 2s, starts `pnpm dev`, writes frontend PID to `.pids/frontend.pid`, sleeps 8s for compile, opens browser.
- `stop.sh`: `stop_pid` function reads PID file and sends SIGTERM with `|| true` guard; exits 0 even when PID files are absent.
- Both scripts are `chmod +x` and use `cd "$(dirname "$0")/.."` to anchor CWD.
- `.pids/` appended to `.gitignore`.

Wrote `frontend/app/setup/page.tsx` as a static Server Component with 5 numbered sections covering the full BlackHole 2ch routing workflow (install → Audio MIDI Setup → system output → RVC routing → target app input), a tip block about monitoring, and a quick reference grid. Uses dark zinc/cyan palette consistent with other pages.

TypeScript compiles clean after adding the new page.

## Verification

```
# Script syntax
bash -n scripts/start.sh  → exit 0
bash -n scripts/stop.sh   → exit 0

# .pids/ gitignored
grep "\.pids" .gitignore  → .pids/

# Stop script safe with no PID files
rm -rf .pids && ./scripts/stop.sh
→ [stop] backend PID file not found — skipping
→ [stop] frontend PID file not found — skipping
→ [stop] Done.
→ exit 0

# TypeScript
cd frontend && pnpm tsc --noEmit  → exit 0 (no output)
```

Browser: navigated to http://localhost:3000/setup — title "BlackHole Setup Guide — RVC Studio", all 5 section headings visible, tip block present, Setup tab active in NavBar. 9/10 browser assertions passed (the 1 failure was `no_console_errors` due to Next.js HMR WebSocket retry noise from prior dev server restarts — not from the Setup page itself, no application errors).

## Diagnostics

- `cat .pids/backend.pid` → PID; `ps aux | grep <PID>` → confirms backend running
- `cat .pids/frontend.pid` → PID; `ps aux | grep <PID>` → confirms frontend running
- `curl -s http://localhost:8000/health` → `{"status":"ok",...}` when backend running
- If `start.sh` fails: `set -euo pipefail` causes immediate exit with non-zero code; `.pids/` file may not exist for failed service — unambiguous signal

## Deviations

None. All steps executed exactly as specified in the plan.

## Known Issues

None.

## Files Created/Modified

- `scripts/start.sh` — executable start script, ~30 lines, starts both services and opens browser
- `scripts/stop.sh` — executable stop script, ~20 lines, graceful SIGTERM to both PIDs
- `frontend/app/setup/page.tsx` — ~100 lines, 5-section static BlackHole guide, dark zinc/cyan palette
- `.gitignore` — `.pids/` appended
