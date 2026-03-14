# S04: Next.js Frontend Polish & Service Scripts

**Goal:** Complete the Next.js UI with Library and Training tabs alongside the existing Realtime tab; write `scripts/start.sh` / `scripts/stop.sh`; add an in-app BlackHole setup guide. Every owned requirement (R007, R008) is proven. R001 and R002 gain their front-end delivery surface.
**Demo:** User runs `./scripts/start.sh` from a clean terminal, browser opens automatically to `/` (Library tab). They can upload a voice sample, see it listed with status badge, navigate to Training, select a profile, start a training job and watch logs stream live, then switch to Realtime and use the existing VC tab. `./scripts/stop.sh` kills both processes cleanly. The BlackHole setup guide is one click away in the nav.

## Must-Haves

- `app/layout.tsx` renders a persistent tab nav (Library / Training / Realtime / Setup) with active-tab highlight; Geist fonts and zinc/cyan dark theme preserved
- `app/page.tsx` (Library): lists profiles with status badges, upload form (`name` + `.wav`/`.mp3` file, `FormData` POST), delete with confirm, shows realtime session active state from `GET /api/realtime/status`
- `app/training/page.tsx` (Training): profile dropdown (all profiles), Start Training / Cancel button, live WebSocket log stream, phase progress bar (6 phases), epoch counter input
- `backend/app/routers/training.py` duplicate body (lines 231–446) removed — clean file with exactly 4 route registrations
- `scripts/start.sh`: kills stale port 8000, sets `RVC_ROOT`, starts backend via `conda run --no-capture-output -n rvc uvicorn`, writes `.pids/`, waits 2s, starts `pnpm dev`, waits 8s, opens browser
- `scripts/stop.sh`: reads `.pids/` and sends SIGTERM; exits 0 even if PIDs are stale
- `app/setup/page.tsx` (Setup): step-by-step BlackHole Audio MIDI Setup guide as a static page with numbered sections
- `pnpm tsc --noEmit` passes after every task
- `.pids/` added to `.gitignore`

## Proof Level

- This slice proves: **final-assembly** (all three user-facing features accessible from one UI; scripts proven against live processes)
- Real runtime required: **yes** — browser verification against live backend + frontend servers
- Human/UAT required: **no** — functional verification is script/browser-based; audible quality check deferred (no trained profile in CI)

## Verification

**T01 (training router fix + Library page):**
```bash
# Training router — exactly 4 route decorators
grep -cE "^@router\.(post|get)|^@ws_router\.websocket" backend/app/routers/training.py
# → 4

# TypeScript clean
cd frontend && pnpm tsc --noEmit
# → no output (exit 0)

# Library page loads and lists profiles in browser
# → GET /api/profiles returns list; upload form visible; delete buttons present
```

**T02 (Training page):**
```bash
# TypeScript clean
cd frontend && pnpm tsc --noEmit

# Training page renders profile dropdown, Start Training button, log area in browser
# → WS connects after POST /api/training/start; log lines appear in UI
```

**T03 (Service scripts + Setup page):**
```bash
# start.sh is executable and runs cleanly
chmod +x scripts/start.sh scripts/stop.sh
./scripts/start.sh
# → .pids/backend.pid and .pids/frontend.pid written
# → browser opens to http://localhost:3000

# stop.sh kills processes
./scripts/stop.sh
# → both PIDs no longer running (kill exits 0)

# TypeScript clean after Setup page
cd frontend && pnpm tsc --noEmit

# .pids/ in .gitignore
grep ".pids" .gitignore
```

## Observability / Diagnostics

- Runtime signals: `start.sh` prints timestamped lines to stdout for each phase (killing stale port, starting backend, starting frontend, opening browser); `stop.sh` prints which PIDs it killed or skipped
- Inspection surfaces:
  - `cat .pids/backend.pid` / `cat .pids/frontend.pid` — quick liveness check
  - `curl -s http://localhost:8000/health` — backend liveness
  - `curl -s http://localhost:3000` — frontend liveness
  - Browser console — Next.js page errors, WS connection failures
- Failure visibility: `start.sh` exits non-zero if `conda run ... uvicorn` start fails within 5s timeout; `stop.sh` logs "PID file not found" if `.pids/` doesn't exist
- Redaction constraints: none (no secrets involved)

## Integration Closure

- Upstream surfaces consumed:
  - `GET /api/profiles` — Library page list + delete
  - `POST /api/profiles` (multipart) — Library page upload
  - `DELETE /api/profiles/{id}` — Library page delete
  - `GET /api/realtime/status` — Library page session indicator
  - `POST /api/training/start`, `POST /api/training/cancel`, `GET /api/training/status/{id}`, `WS /ws/training/{id}` — Training page
  - `backend/app/routers/training.py` (lines 231–446) — duplicate removed
  - `frontend/app/realtime/page.tsx` — unchanged; nav added around it
- New wiring introduced in this slice:
  - `app/layout.tsx` — shared nav bar wrapping all routes including existing `/realtime`
  - `scripts/start.sh` + `scripts/stop.sh` — operational entrypoints for the whole stack
  - `app/setup/page.tsx` — static BlackHole guide linked from nav
- What remains before the milestone is truly usable end-to-end:
  - End-to-end with real audio (requires trained profile): run S02 integration smoke test, then open Realtime tab and start a session → BlackHole produces audible output (human UAT)

## Tasks

- [x] **T01: Fix training router duplicate + build Library page** `est:90m`
  - Why: The training router duplicate (lines 231–446) risks fragile double-registration; must be fixed before the Training UI depends on it. The Library page (`/`) is the landing experience — lists profiles, upload, delete. The nav layout wraps all three existing routes.
  - Files: `backend/app/routers/training.py`, `frontend/app/layout.tsx`, `frontend/app/page.tsx`
  - Do:
    1. Delete lines 231–446 from `backend/app/routers/training.py` (the duplicate body starting with second `class StartTrainingRequest`). Leave lines 1–230 intact. Verify exactly 4 decorators remain: `@router.post("/start")`, `@router.post("/cancel")`, `@router.get("/status/{profile_id}")`, `@ws_router.websocket("/ws/training/{profile_id}")`.
    2. Replace `app/layout.tsx`: update title to "RVC Studio", add persistent `<nav>` with 4 tab links (Library=`/`, Training=`/training`, Realtime=`/realtime`, Setup=`/setup`) using `usePathname` for active-tab styling. Use `'use client'` on a child `NavBar` component; keep `layout.tsx` itself as a Server Component. Dark zinc theme: `bg-zinc-950` body, `bg-zinc-900/80` nav, `text-cyan-400` active tab, `text-zinc-400` inactive.
    3. Write `app/page.tsx` (Library): `'use client'`; `const API = 'http://localhost:8000'`; fetch `/api/profiles` on mount; poll `GET /api/realtime/status` every 3s to show a "Session Active" badge if `active=true`; render profile list with name, status badge (color-coded: untrained=zinc, training=amber, trained=cyan, failed=red), created_at; upload form with `name` text input + file picker (`accept=".wav,.mp3,.ogg,.flac"`); on submit build `FormData` and POST — do NOT set Content-Type header; delete button with `window.confirm`; empty-state message if no profiles.
    4. `pnpm tsc --noEmit` must pass.
  - Verify: `grep -cE "^@router\.(post|get)|^@ws_router\.websocket" backend/app/routers/training.py` → `4`; `cd frontend && pnpm tsc --noEmit` exits 0; Library page renders in browser with profile list from live backend
  - Done when: training router has exactly 4 decorators; Library page lists real profiles, upload form posts successfully, delete works; TypeScript clean

- [x] **T02: Build Training page** `est:75m`
  - Why: R002 needs a front-end delivery surface. Users must be able to select a profile, start training, watch log lines scroll live via WebSocket, track phase progress, and cancel.
  - Files: `frontend/app/training/page.tsx`
  - Do:
    1. Create `frontend/app/training/` directory and `page.tsx`. `'use client'`. `const API = 'http://localhost:8000'`.
    2. State: `profiles: Profile[]`, `selectedProfileId: string | null`, `jobPhase: string | null`, `logLines: string[]`, `wsState: 'disconnected'|'connecting'|'connected'|'done'`, `trainingStatus: 'idle'|'running'|'done'|'failed'`, `epochs: number` (default 20).
    3. Fetch `/api/profiles` on mount; render profile dropdown (show all profiles, not just trained — user picks who to train).
    4. Phase progress bar: 6 phases `['preprocess','extract_f0','extract_feature','train','index','done']`; highlight current phase in cyan, completed in zinc-600, upcoming in zinc-800.
    5. Start Training: POST `/api/training/start` body `{profile_id: selectedProfileId, total_epoch: epochs}`; on success open `WebSocket(${WS_BASE}/ws/training/${selectedProfileId})`; `onmessage` appends `msg.message` to `logLines` (cap at 500 lines); update `jobPhase` from `msg.phase`; on `{type:'done'}` close WS and set `trainingStatus='done'`; on `{type:'error'}` set `trainingStatus='failed'`.
    6. Cancel: POST `/api/training/cancel` body `{profile_id: selectedProfileId}`; close WS ref; set `trainingStatus='idle'`.
    7. Log area: dark `bg-zinc-950` scrollable box (`max-h-80`), monospace text, auto-scroll to bottom on new lines via `useEffect` + `scrollTop = scrollHeight`.
    8. Connect WS *after* POST returns `job_id` (not before). Use `useRef<WebSocket>` + cleanup in `useEffect` return.
    9. `WS_BASE = 'ws://localhost:8000'` constant (not derived from API).
    10. `pnpm tsc --noEmit` must pass.
  - Verify: `cd frontend && pnpm tsc --noEmit` exits 0; in browser: Training page renders profile dropdown and Start Training button; with live backend, selecting a profile and clicking Start shows WS log lines appearing and phase bar advancing
  - Done when: Training page renders; TypeScript clean; WS log stream populates UI with real messages from backend (verified in browser against live server)

- [x] **T03: Service scripts + BlackHole Setup page** `est:60m`
  - Why: R007 (start/stop scripts) and R008 (BlackHole guide) are both unvalidated. This task completes the milestone operability requirements.
  - Files: `scripts/start.sh`, `scripts/stop.sh`, `frontend/app/setup/page.tsx`, `.gitignore`
  - Do:
    1. Create `scripts/` directory. Write `scripts/start.sh`:
       - Shebang `#!/usr/bin/env bash` + `set -euo pipefail`
       - `cd "$(dirname "$0")/.."` to anchor CWD to `rvc-web/`
       - Kill stale port: `lsof -ti:8000 | xargs kill -9 2>/dev/null || true`
       - `mkdir -p .pids`
       - Export `RVC_ROOT=Retrieval-based-Voice-Conversion-WebUI`
       - Start backend: `conda run --no-capture-output -n rvc uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 & BACKEND_PID=$! && echo $BACKEND_PID > .pids/backend.pid`
       - Print: `echo "[start] backend PID $BACKEND_PID"`
       - `sleep 2`
       - Start frontend: `cd frontend && pnpm dev & FRONTEND_PID=$! && echo $FRONTEND_PID > ../.pids/frontend.pid && cd ..`
       - Print: `echo "[start] frontend PID $FRONTEND_PID"`
       - `sleep 8`
       - `open http://localhost:3000`
       - Print: `echo "[start] browser opened — http://localhost:3000"`
    2. Write `scripts/stop.sh`:
       - Shebang `#!/usr/bin/env bash`
       - `cd "$(dirname "$0")/.."` to anchor CWD
       - For each PID file (`backend.pid`, `frontend.pid`): if file exists, read PID, `kill $PID 2>/dev/null || true`, print what was done; else print "not found"
    3. `chmod +x scripts/start.sh scripts/stop.sh`
    4. Add `.pids/` to `.gitignore`
    5. Write `frontend/app/setup/page.tsx` — static Server Component (no `'use client'`). BlackHole Audio MIDI Setup guide with 5 numbered sections: (1) Install BlackHole 2ch, (2) Create Multi-Output Device in Audio MIDI Setup, (3) Set system output to Multi-Output Device, (4) Configure RVC to output to BlackHole 2ch, (5) Set your target app's mic to BlackHole 2ch. Include exact macOS menu paths. Use same dark zinc/cyan palette as other pages. Add a note about BlackHole being device id=1 on a default install.
    6. Verify Setup page appears in nav (already wired in T01 layout).
    7. `pnpm tsc --noEmit` must pass.
  - Verify: `bash -n scripts/start.sh && bash -n scripts/stop.sh` (syntax check); `.pids/` in `.gitignore`; `cd frontend && pnpm tsc --noEmit` exits 0; `./scripts/start.sh` writes PID files and opens browser; `./scripts/stop.sh` terminates processes
  - Done when: both scripts executable and syntactically valid; `.pids/` gitignored; setup page renders in browser at `/setup`; `./scripts/start.sh` starts both services and opens browser; `./scripts/stop.sh` stops them cleanly; TypeScript clean

## Files Likely Touched

- `backend/app/routers/training.py` — remove duplicate body (lines 231–446)
- `frontend/app/layout.tsx` — add persistent tab nav
- `frontend/app/page.tsx` — Library page (replace create-next-app placeholder)
- `frontend/app/training/page.tsx` — new: Training page with WS log stream
- `frontend/app/setup/page.tsx` — new: BlackHole setup guide
- `scripts/start.sh` — new
- `scripts/stop.sh` — new
- `.gitignore` — add `.pids/`
