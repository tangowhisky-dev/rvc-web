---
estimated_steps: 5
estimated_files: 5
---

# T03: Service Scripts + BlackHole Setup Page

**Slice:** S04 — Next.js Frontend Polish & Service Scripts
**Milestone:** M001

## Description

Delivers R007 (start/stop scripts) and R008 (BlackHole setup guide) — the two remaining unvalidated operability requirements. `scripts/start.sh` is the single command a user runs from a clean terminal to get everything going. `scripts/stop.sh` tears it down cleanly. The Setup page at `/setup` is the in-app guide explaining exactly how to route BlackHole 2ch so the converted voice actually reaches other apps.

## Steps

1. **Create `scripts/` directory** and write `scripts/start.sh`:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail

   cd "$(dirname "$0")/.."

   echo "[start] Killing any stale process on port 8000..."
   lsof -ti:8000 | xargs kill -9 2>/dev/null || true

   mkdir -p .pids

   export RVC_ROOT=Retrieval-based-Voice-Conversion-WebUI

   echo "[start] Starting backend (conda rvc env)..."
   conda run --no-capture-output -n rvc \
     uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
   BACKEND_PID=$!
   echo "$BACKEND_PID" > .pids/backend.pid
   echo "[start] Backend PID $BACKEND_PID"

   sleep 2

   echo "[start] Starting frontend..."
   (cd frontend && pnpm dev) &
   FRONTEND_PID=$!
   echo "$FRONTEND_PID" > .pids/frontend.pid
   echo "[start] Frontend PID $FRONTEND_PID"

   echo "[start] Waiting for frontend to compile (8s)..."
   sleep 8

   echo "[start] Opening browser..."
   open http://localhost:3000
   echo "[start] Done — http://localhost:3000"
   ```

2. **Write `scripts/stop.sh`**:
   ```bash
   #!/usr/bin/env bash

   cd "$(dirname "$0")/.."

   stop_pid() {
     local name=$1
     local pidfile=".pids/${name}.pid"
     if [ -f "$pidfile" ]; then
       local pid
       pid=$(cat "$pidfile")
       echo "[stop] Stopping $name (PID $pid)..."
       kill "$pid" 2>/dev/null || true
       rm -f "$pidfile"
     else
       echo "[stop] $name PID file not found — skipping"
     fi
   }

   stop_pid backend
   stop_pid frontend

   echo "[stop] Done."
   ```

3. **`chmod +x scripts/start.sh scripts/stop.sh`**

4. **Add `.pids/` to `.gitignore`** — append the line `.pids/` to `.gitignore`

5. **Write `frontend/app/setup/page.tsx`** — static Server Component (no `'use client'`). Five numbered sections:
   - Section 1: "Install BlackHole 2ch" — download from existingrepository `BlackHole` (Existential Audio), install `.pkg`, restart Mac audio
   - Section 2: "Open Audio MIDI Setup" — Finder → Applications → Utilities → Audio MIDI Setup; click `+` bottom-left → Create Multi-Output Device; check BlackHole 2ch AND your speakers (e.g. MacBook Pro Speakers) in the list; set the speaker as the master (clock source)
   - Section 3: "Set System Output to Multi-Output Device" — System Settings → Sound → Output → select the Multi-Output Device you created; this lets you hear both monitor and routed audio simultaneously
   - Section 4: "Start RVC and route output to BlackHole" — in the Realtime tab, set Output Device to "BlackHole 2ch"; BlackHole 2ch is typically device id=1 on a default install; RVC audio engine writes converted voice to this virtual device
   - Section 5: "Configure your target app's input" — in your DAW or app (e.g. Discord, Zoom, GarageBand), set the microphone/input to "BlackHole 2ch"; that app now receives the converted voice as its input
   - Tip block: "To monitor what BlackHole is receiving, add BlackHole 2ch to the Multi-Output Device and route your speakers there too."
   - Use same dark zinc/cyan palette as other pages. Each section: numbered `<h2>` in cyan, body in `text-zinc-300`. Page title: "BlackHole Setup Guide". No inline SVGs needed — numbered sections are sufficient.

6. **Run `pnpm tsc --noEmit`** from `frontend/` and fix any type errors.

## Must-Haves

- [ ] `scripts/start.sh` sets `RVC_ROOT` before starting uvicorn
- [ ] `scripts/start.sh` uses `--no-capture-output` in `conda run`
- [ ] `scripts/start.sh` writes PIDs to `.pids/backend.pid` and `.pids/frontend.pid`
- [ ] `scripts/stop.sh` exits 0 even if PID files don't exist (`|| true` guard)
- [ ] Both scripts use `cd "$(dirname "$0")/.."` to anchor CWD to `rvc-web/`
- [ ] `.pids/` added to `.gitignore`
- [ ] `frontend/app/setup/page.tsx` renders all 5 sections with actionable macOS steps
- [ ] `pnpm tsc --noEmit` exits 0

## Verification

```bash
# Script syntax check
bash -n scripts/start.sh
bash -n scripts/stop.sh
# → both exit 0 (no syntax errors)

# .pids/ gitignored
grep "\.pids" .gitignore
# → .pids/

# TypeScript clean
cd frontend && pnpm tsc --noEmit
# → (no output, exit 0)

# Stop script is safe with no PID files
rm -rf .pids && ./scripts/stop.sh
# → exits 0, prints "not found" for both

# Full start (functional test — run manually):
./scripts/start.sh
# → .pids/backend.pid and .pids/frontend.pid exist
# → browser opens to http://localhost:3000
# → curl -s http://localhost:8000/health → {"status":"ok",...}
./scripts/stop.sh
# → both PIDs terminated
# → curl -s http://localhost:8000/health → connection refused
```

Browser: navigate to `http://localhost:3000/setup` → Setup tab visible in nav, all 5 sections render with correct content.

## Observability Impact

- Signals added/changed:
  - `start.sh` prints timestamped `[start]` lines to stdout — future agent can tail or inspect for startup failures
  - `stop.sh` prints `[stop]` lines — confirms which PIDs were killed or skipped
  - `.pids/backend.pid` / `.pids/frontend.pid` — inspectable process state on disk
- How a future agent inspects this: `cat .pids/backend.pid` → PID; `ps aux | grep <PID>` → confirms running; `curl http://localhost:8000/health` → backend liveness
- Failure state exposed: if `start.sh` fails (e.g. uvicorn port collision not resolved), `set -euo pipefail` causes immediate exit with non-zero code; PID file may not be written, so `stop.sh` will print "not found" — unambiguous signal that start failed

## Inputs

- `S04-RESEARCH.md` — start.sh/stop.sh constraints: `--no-capture-output`, `conda run -n rvc`, `RVC_ROOT` must be exported, port 8000 stale-kill, `sleep 2` before frontend, `sleep 8` before open, `.pids/` directory
- `.gitignore` — append `.pids/` (check it's not already present)
- T01 `frontend/app/layout.tsx` — Setup tab link to `/setup` already in nav; page just needs to exist
- T02 `frontend/app/training/page.tsx` — TypeScript must remain clean after adding new page

## Expected Output

- `scripts/start.sh` — executable, ~30 lines, starts both services and opens browser
- `scripts/stop.sh` — executable, ~20 lines, graceful SIGTERM to both PIDs
- `frontend/app/setup/page.tsx` — ~80 lines, 5-section static guide, dark zinc/cyan palette
- `.gitignore` — `.pids/` appended
- TypeScript compiles clean
- All S04 must-haves satisfied; R007 and R008 requirements validated
