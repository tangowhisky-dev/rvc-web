# S04: Next.js Frontend Polish & Service Scripts — UAT

**Milestone:** M001
**Written:** 2026-03-14

## UAT Type

- UAT mode: live-runtime
- Why this mode is sufficient: All three features (Library, Training, Realtime) are accessible in the browser and backed by a live FastAPI backend. Script verification confirms PID-file lifecycle. Full hardware audio validation (hearing converted voice on BlackHole) deferred until a trained profile is available, consistent with S04 scope.

## Preconditions

1. `rvc` conda environment is installed and activated
2. No other process is running on port 8000 or 3000
3. `Retrieval-based-Voice-Conversion-WebUI/` directory exists at repo root (for backend)
4. `pnpm` installed globally
5. BlackHole 2ch Audio Loopback driver installed on macOS

## Smoke Test

Run `./scripts/start.sh` from the repo root — browser opens to `http://localhost:3000` showing the Library tab with NavBar (Library / Training / Realtime / Setup), no console errors. Run `./scripts/stop.sh` — both processes terminate cleanly.

## Test Cases

### 1. Service Scripts Lifecycle

1. From repo root: `chmod +x scripts/start.sh scripts/stop.sh && ./scripts/start.sh`
2. Observe: `[start] backend PID <N>`, `[start] frontend PID <M>`, `[start] browser opened` printed to stdout
3. Verify: `cat .pids/backend.pid` and `cat .pids/frontend.pid` return valid PIDs
4. Verify: `curl -s http://localhost:8000/health` returns `{"status":"ok",...}`
5. Run: `./scripts/stop.sh`
6. **Expected:** `[stop] killed backend PID <N>`, `[stop] killed frontend PID <M>` printed; both PIDs no longer running; exit 0

### 2. Stop Script Safe Without PID Files

1. `rm -rf .pids && ./scripts/stop.sh`
2. **Expected:** `[stop] backend PID file not found — skipping`, `[stop] frontend PID file not found — skipping`, `[stop] Done.`; exit 0

### 3. Library Page — Profile List and Upload

1. Start both services (`./scripts/start.sh`)
2. Navigate to `http://localhost:3000` (Library tab)
3. Observe: empty-state message if no profiles exist; upload form visible (name input + file picker + Submit)
4. Upload: enter a name, select a `.wav` or `.mp3` file, click Submit
5. **Expected:** profile appears in list with "untrained" badge (zinc color); name and created-at shown
6. Delete: click Delete on a profile; confirm in dialog
7. **Expected:** profile removed from list

### 4. Library Page — Session Active Indicator

1. Start a realtime session via `POST /api/realtime/start` (curl or Realtime tab)
2. Navigate to or refresh Library tab
3. **Expected:** amber "Session Active" banner appears within 3 seconds (polling interval)
4. Stop the session
5. **Expected:** banner disappears within 3 seconds

### 5. Training Page — Profile Dropdown and Log Stream

1. Navigate to `http://localhost:3000/training`
2. Observe: profile dropdown shows all profiles (with ✓/⟳/✗ status suffix); epoch input defaults to 20
3. Select a profile; click "Start Training"
4. **Expected:** phase bar appears; "preprocess" pill lights cyan; log lines begin scrolling in the log area
5. Observe phases advance through: preprocess → F0 Extract → Features → Train → Index → Done
6. **Expected:** on completion, "Done" pill lights cyan; status badge turns emerald
7. Cancel mid-run: click "Cancel" during an active training job
8. **Expected:** WebSocket closes; status returns to idle; log area shows last received lines

### 6. NavBar Active Tab Highlighting

1. Navigate to `/` — Library tab has `text-cyan-400` and cyan bottom border
2. Navigate to `/training` — Training tab is active
3. Navigate to `/realtime` — Realtime tab is active
4. Navigate to `/setup` — Setup tab is active
5. **Expected:** exactly one tab is highlighted cyan per page; others are zinc-400

### 7. BlackHole Setup Guide

1. Navigate to `http://localhost:3000/setup`
2. **Expected:** page titled "BlackHole Setup Guide"; 5 numbered sections visible:
   - Install BlackHole 2ch
   - Create Multi-Output Device
   - Set System Output
   - Configure RVC Output
   - Set Target App Input
3. Observe quick reference panel and monitoring tip block
4. **Expected:** no console errors; dark zinc/cyan palette consistent with other pages

## Edge Cases

### Start Script — Stale Port 8000

1. Start a process on port 8000: `python3 -m http.server 8000 &`
2. Run `./scripts/start.sh`
3. **Expected:** start.sh kills the stale process before starting uvicorn; no port-already-in-use error; backend starts normally

### Upload — Missing Fields

1. Navigate to Library page
2. Click Submit without entering a name or selecting a file
3. **Expected:** Submit button is disabled; no request sent

### Training — No Profile Selected

1. Navigate to Training page with no profiles in the database
2. **Expected:** dropdown shows empty / disabled; Start Training button disabled or no-op

## Failure Signals

- `start.sh` exits non-zero: `set -euo pipefail` fires; look at the last printed `[start]` line to determine which phase failed
- Library page shows error banner: backend not running or CORS misconfiguration
- Training WS log area stays empty after Start: check browser Network tab for WS connection failure; verify backend is on port 8000
- Phase bar never advances past "preprocess": training subprocess may have exited; check backend logs for subprocess stderr
- `stop.sh` says "PID file not found": services were not started via `start.sh` or `.pids/` was manually deleted — kill processes by port: `lsof -ti:8000 | xargs kill -9`
- TypeScript error after any frontend edit: `cd frontend && pnpm tsc --noEmit`

## Requirements Proved By This UAT

- R007 (Start/Stop Service Scripts) — `start.sh` and `stop.sh` lifecycle verified: PID files written and killed; stale-port handling confirmed; exit-0-without-PID-files confirmed
- R008 (BlackHole Setup Guide) — `/setup` page renders all 5 sections with macOS menu paths in the browser; accessible from NavBar

## Not Proven By This UAT

- Full end-to-end audible voice conversion on BlackHole hardware — requires a trained profile (S02 integration smoke test first); hardware UAT deferred to milestone completion verification
- R003 (Realtime VC) hardware validation — Realtime tab UI is verified by S03; audible output on BlackHole hardware is separate from S04 scope
- Multi-profile training slot isolation (deferred to M002 per D007/D022)

## Notes for Tester

- The Training page uses `epochs` (not `total_epoch`) in the POST body — already aligned with backend.
- The Library page polls `/api/realtime/status` every 3 seconds — session-active badge may lag by up to 3s.
- `start.sh` sleeps 2s after backend start and 8s after frontend start — these are fixed waits; on slower machines the browser may open before Next.js finishes its initial compile. If the browser shows a compile error, wait a few seconds and refresh.
- BlackHole 2ch device is expected at system device id=1 on a default install; auto-selection on the Realtime tab uses name-fragment matching, not hardcoded IDs (D035).
- The Setup page is a static Server Component — no client JavaScript runs on it; it will load instantly even if the backend is down.
