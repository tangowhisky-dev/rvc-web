# S04: Next.js Frontend Polish & Service Scripts — Research

**Date:** 2026-03-14

## Summary

S04 is the polish and operability slice. Three deliverables: (1) complete the Next.js UI with Library and Training tabs alongside the existing Realtime tab, (2) write `scripts/start.sh` + `scripts/stop.sh`, and (3) add an in-app BlackHole setup guide. All backend APIs already exist and are contract-tested — this slice is purely frontend + scripts. Risk is low.

The frontend was bootstrapped in S03 with Next.js 16 + App Router + TypeScript + Tailwind 4. Only one page exists: `/realtime/page.tsx` (~650 lines). The root `page.tsx` is still the create-next-app placeholder. The layout `app/layout.tsx` is the bare default with no navigation. No icon library is installed — inline SVGs are used throughout the realtime page. This sets the pattern to follow for new pages.

The training router `backend/app/routers/training.py` has a silent code-duplication bug: the entire file body appears twice starting at line 231, meaning all four REST endpoints and the WS route are each registered twice. FastAPI silently accepts this (last registration wins for REST; WS may silently deduplicate). This is not a regression risk for S04 but should be cleaned up as part of T01 since the training router file is being read during frontend work anyway.

## Recommendation

**Tab navigation in the root layout, three routes.** Use a shared `<nav>` in `app/layout.tsx` with three tab links (`/`, `/training`, `/realtime`). The root `/` route becomes the Library page. This matches the D003 decision (App Router), keeps each page file self-contained, and requires no routing library. The existing Realtime page is already at `/realtime` — add the nav bar without touching its logic.

For scripts: `conda run --no-capture-output -n rvc uvicorn backend.app.main:app --host 0.0.0.0 --port 8000` in the background, `pnpm dev` in the frontend dir. Write PIDs to `.pids/backend.pid` and `.pids/frontend.pid`. Stop by reading PID files and killing.

Do **not** install an icon library — the codebase uses inline SVGs. Keep that pattern for consistency.

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| File upload UI | `<input type="file">` + fetch FormData (no library) | Matches what `POST /api/profiles` multipart endpoint expects; no lib needed |
| WebSocket in training tab | Same `useRef<WebSocket>` + `ws.onmessage` pattern as `/realtime/page.tsx` | Already proven working; copy the connection + cleanup pattern exactly |
| Status polling | `setInterval` + `GET /api/training/status/{profile_id}` | Training status endpoint exists; use polling while job is active as fallback if WS disconnects |
| Tailwind 4 | Already configured via `@tailwindcss/postcss` in `postcss.config.mjs` | Tailwind 4 uses `@import "tailwindcss"` in globals.css — no `tailwind.config.ts` needed |
| Tab navigation | `usePathname` from `next/navigation` in layout | Clean active-tab detection without any routing lib |

## Existing Code and Patterns

- `frontend/app/realtime/page.tsx` — The canonical pattern for all new pages: `'use client'`, typed interfaces at top, `const API = 'http://localhost:8000'`, `useEffect` for API fetches with `cancelled` flag cleanup, `useRef` for WS + worker, dark zinc+cyan color palette. **Follow this exactly for Library and Training pages.**
- `frontend/public/waveform-worker.js` — Web Worker lives in `public/` (not bundled). Training tab does NOT need a worker — log lines render directly on the main thread.
- `frontend/app/layout.tsx` — Currently bare default. S04 must add nav tabs here while keeping the Geist font variables and `antialiased` body class.
- `frontend/app/globals.css` — Tailwind 4 `@import "tailwindcss"` at top; `@theme inline` block for CSS vars. Custom styles go here if needed, but the realtime page proves Tailwind utility classes are sufficient.
- `backend/app/routers/profiles.py` — `ProfileOut`: `{id, name, status, created_at, sample_path}`. Status values: `"untrained" | "training" | "trained" | "failed"`. POST is `multipart/form-data` with `name` (Form field) + `file` (UploadFile).
- `backend/app/routers/training.py` — **BUG**: entire file is duplicated from line 231. Clean up in T01. The first copy (lines 1–230) is correct: `ws_router` uses `ws_router.websocket(...)`, REST on `router`. The second copy (lines 231–446) erroneously uses `@router.websocket(...)` for the WS route.
- `backend/app/routers/training.py` — Training phases emitted by WS: `"preprocess"`, `"extract_f0"`, `"extract_feature"`, `"train"`, `"index"`, `"done"`, `"error"`. Message types: `"log" | "phase" | "done" | "error"`.
- `backend/app/routers/realtime.py` — `GET /api/realtime/status` returns `{active, session_id, profile_id, input_device, output_device}`. The Library page can poll this to show whether a session is active.
- `backend/app/db.py` — `profiles` table has columns: `id, name, status, sample_path, model_path, index_path, created_at`.

## Constraints

- **CORS is locked to `http://localhost:3000`** in `main.py`. Frontend must run on port 3000 (`pnpm dev` default). `start.sh` must not pass `--port` to pnpm dev (or explicitly pass `--port 3000`).
- **Hardcoded `API = 'http://localhost:8000'`** in `realtime/page.tsx`. New pages must use the same hardcoded constant (local-only tool, no env var needed).
- **`conda run --no-capture-output -n rvc`** for backend startup — the `--no-capture-output` flag is needed so uvicorn stdout is not buffered, which matters for the startup JSON log line.
- **`RVC_ROOT` env var** must be exported before backend starts. `start.sh` must set it to `Retrieval-based-Voice-Conversion-WebUI` (relative path from `rvc-web/` works because uvicorn CWD is `rvc-web/`).
- **Tailwind 4 syntax**: uses `@import "tailwindcss"` in CSS, NOT `@tailwind base/components/utilities`. No `tailwind.config.ts` — configuration is via `@theme` blocks in CSS. Arbitrary values still work (`bg-zinc-950`, `text-cyan-400`, etc.).
- **TypeScript strict mode** is enabled. All component props must be typed. `pnpm tsc --noEmit` must pass after every task.
- **No `<script>` tags or CDN imports** in Next.js pages — use `next/font/google` for fonts and inline SVGs for icons.
- **`start.sh` CWD** must be `rvc-web/` (the project root) because backend imports use `backend.app.main` package path (D012).
- **PID file location**: `.pids/` directory at `rvc-web/` root. Must be created by `start.sh` if absent. Already in `.gitignore` baseline should cover it; confirm or add.
- **`open` command** on macOS: `open http://localhost:3000` after a brief sleep to let frontend start.

## Common Pitfalls

- **Duplicate training router endpoints** — The `training.py` router has the entire file pasted twice. Running with this file causes FastAPI to register `/api/training/start` twice and `/ws/training/{profile_id}` once via `ws_router` and once via `router`. The first `ws_router` registration is correct; the second `@router.websocket` is the bug. Clean the file in T01 before building the training UI — do not leave it duplicated.
- **WS connection before job exists** — Training WS at `/ws/training/{profile_id}` polls up to 30s for the job to appear. Connect the WS *after* POST `/api/training/start` returns the `job_id`, not before. The realtime page does POST-then-WS correctly; training page must do the same.
- **`transferControlToOffscreen` is one-time and irreversible** — The realtime page transfers canvas control to the worker. Do NOT call `transferControlToOffscreen` on the same canvas element twice (e.g. if the user stops and restarts). The realtime page handles this correctly by terminating and recreating the worker on stop. Training tab has no canvases so this is not a concern there.
- **start.sh backgrounding + PID capture**: Use `cmd & echo $!` pattern. Do NOT use `nohup` with `$!` (nohup may alter PID). Pattern: `conda run ... uvicorn ... &\nBACKEND_PID=$!\necho $BACKEND_PID > .pids/backend.pid`. Wait 2s before starting frontend so the backend port binds first.
- **stop.sh kill signal**: Use `kill` (SIGTERM) not `kill -9` (SIGKILL) to allow uvicorn and Next.js to clean up. Add `-f || true` so stop.sh doesn't fail if PIDs are stale.
- **Training page — profile status display**: `status` field in `ProfileOut` is a raw string (`"untrained"`, `"training"`, `"trained"`, `"failed"`). Map to human labels in the UI. A profile in `"training"` state means a training job is running (or was interrupted); the training page should handle this gracefully.
- **Tailwind dark mode**: The realtime page uses `bg-zinc-950` / `text-zinc-100` without `dark:` prefix — it's hardcoded dark theme. Follow this — do not add dark/light mode toggle; it's out of scope.
- **Library page file upload**: `POST /api/profiles` expects `multipart/form-data` with both `name` (text field) and `file` (binary). Use `FormData` API directly; do not set `Content-Type` header manually (let the browser set it with the boundary).

## Open Risks

- **training.py router duplicate**: If left unfixed, FastAPI will register POST `/api/training/start` twice. In practice FastAPI uses the last-registered handler, which happens to be functionally identical — so training still works. But this is fragile. Fix it in T01 before building the training UI.
- **`pnpm dev` startup time before `open`**: The `sleep` duration in `start.sh` before `open http://localhost:3000` needs to be long enough for Next.js compilation (typically 5–8 seconds on first boot with Turbopack). Use `sleep 8` as a safe default; power users can reduce it.
- **BlackHole guide accuracy**: The macOS Audio MIDI Setup steps depend on the macOS version (Ventura / Sonoma). The core steps (create Multi-Output Device, set BlackHole as output, route app to BlackHole) are version-stable. The screenshots differ but the step sequence is the same since at least macOS 12. The guide in-app should describe steps not reference screenshots.
- **Port 8000 already in use**: If a prior uvicorn process wasn't stopped cleanly, `start.sh` will fail silently (uvicorn binds the port, but start.sh has already written the new PID). Add a port check with `lsof -ti:8000 | xargs kill -9 2>/dev/null || true` at the top of `start.sh` to evict stale processes.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| Next.js 16 / React frontend | `frontend-design` | Installed — available at `~/.gsd/agent/skills/frontend-design/SKILL.md` |
| Bash / shell scripts | (none needed) | n/a — simple start/stop scripts |

## Task Breakdown (Proposed)

**T01 — Fix training router + Library page (`/`)**
- Clean up `training.py` router duplicate (20 lines)
- Implement `app/layout.tsx` with 3-tab nav (Library / Training / Realtime)
- Implement `app/page.tsx` — Library: list profiles, upload form, delete, status badges
- Verification: `pnpm tsc --noEmit` passes; Library page renders in browser with real API

**T02 — Training page (`/training`)**
- Implement `app/training/page.tsx`: profile selector, Start Training, epochs input, live WS log stream, phase progress bar, Cancel button
- Verification: `pnpm tsc --noEmit` passes; connect to live backend, submit training job, see log stream in browser

**T03 — Service scripts + BlackHole guide**
- Write `scripts/start.sh` and `scripts/stop.sh`
- Add BlackHole setup guide as a collapsible component in the Realtime page (or as a `/setup` route linked from the nav)
- Verification: `./scripts/start.sh` from clean terminal opens browser; `./scripts/stop.sh` kills both processes

## Sources

- Training WS message shapes confirmed from `backend/app/training.py` (lines 16–19 docstring + queue.put calls at lines 109, 356, 366, 526, 545)
- Training phases confirmed: `preprocess → extract_f0 → extract_feature → train → index → done`
- Duplicate router bug confirmed by `grep -n "@router\|@ws_router" backend/app/routers/training.py` showing 8 route registrations (4 pairs)
- Frontend constraints confirmed by reading `realtime/page.tsx`, `layout.tsx`, `globals.css`, `package.json`, `tsconfig.json`
- conda/uvicorn invocation confirmed by `conda run -n rvc which uvicorn` → `/opt/miniconda3/envs/rvc/bin/uvicorn`
- pnpm location confirmed: `/opt/homebrew/bin/pnpm`, version 10.28.1
- Node version: 25.8.0 (supports all ES2017+ features referenced in tsconfig)
