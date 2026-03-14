---
id: T01
parent: S04
milestone: M001
provides:
  - training router deduplicated (exactly 4 route decorators)
  - persistent NavBar tab component wired into layout.tsx
  - Library page (/) with profile list, status badges, upload form, delete
key_files:
  - backend/app/routers/training.py
  - frontend/app/layout.tsx
  - frontend/app/NavBar.tsx
  - frontend/app/page.tsx
key_decisions:
  - NavBar extracted to a separate file (frontend/app/NavBar.tsx) as a 'use client' component, keeping layout.tsx a pure Server Component
  - Library page polls GET /api/realtime/status every 3s via setInterval with cancelled-flag cleanup to show amber session-active banner
patterns_established:
  - NavBar.tsx pattern: 'use client', usePathname(), TABS constant array, href==='/' exact-match vs startsWith for active detection
  - Page-level API fetch pattern: async fetchProfiles() extracted as named function (not inline arrow), so upload/delete handlers can call it after mutations
observability_surfaces:
  - grep -cE "^@router.(post|get)|^@ws_router.websocket" backend/app/routers/training.py → must return 4
  - grep -c "class StartTrainingRequest" backend/app/routers/training.py → must return 1
  - cd frontend && pnpm tsc --noEmit → exit 0
duration: ~20m
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T01: Fix Training Router Duplicate + Build Library Page

**Removed 207-line duplicate from training.py and built the Library page with profile list, upload, delete, and session indicator; NavBar wired into all routes.**

## What Happened

**Step 1 — training.py deduplication:** Lines 238–446 (the second copy of all models, routes, and WebSocket handler) were removed using a Python script that kept lines 1–237 and stripped trailing blank lines. Result: 237-line file with exactly 4 route decorators and 1 `StartTrainingRequest` class.

**Step 2 — layout.tsx + NavBar:** `app/layout.tsx` updated to import a new `NavBar` component, update metadata title to "RVC Studio", and add `bg-zinc-950` to body. The NavBar itself was created as `frontend/app/NavBar.tsx` — a `'use client'` component using `usePathname()` for active-tab detection. Tab links: Library (`/`), Training (`/training`), Realtime (`/realtime`), Setup (`/setup`). Active tab uses `text-cyan-400 border-cyan-500`; inactive uses `text-zinc-400 hover:text-zinc-200`.

**Step 3 — Library page:** `frontend/app/page.tsx` replaced the create-next-app placeholder with a full `'use client'` Library page. Features: fetch `/api/profiles` on mount with a named `fetchProfiles()` function so upload/delete handlers can refetch; `setInterval` polling of `GET /api/realtime/status` every 3s with cancelled-flag cleanup; upload via `FormData` POST (no manual Content-Type); delete with `window.confirm` then `DELETE /api/profiles/{id}`; status badge with color-coded variants for untrained/training/trained/failed; empty state with dashed border; upload form with both name and file required before submit enabled.

**Step 4 — TypeScript:** `pnpm tsc --noEmit` exits 0 with no output.

## Verification

```
# Route decorators
grep -cE "^@router.(post|get)|^@ws_router.websocket" backend/app/routers/training.py
→ 4  ✓

# No duplicate class
grep -c "class StartTrainingRequest" backend/app/routers/training.py
→ 1  ✓

# TypeScript clean
cd frontend && pnpm tsc --noEmit
→ (no output, exit 0)  ✓
```

Browser verification against running frontend (no backend):
- Title: "RVC Studio" ✓
- Nav tabs: Library (active, cyan border), Training, Realtime, Setup all visible ✓
- Library tab has `text-cyan-400 border-cyan-500` CSS classes (confirmed via browser_evaluate) ✓
- Empty-state message "No voice profiles yet — upload one below." rendered ✓
- Upload form: name input, file input, submit button (disabled without both fields) visible ✓
- Error banner displayed (expected — no backend running) ✓

## Diagnostics

- `grep -cE "^@router.(post|get)|^@ws_router.websocket" backend/app/routers/training.py` → 4: confirms no double-registration
- `grep -c "class StartTrainingRequest" backend/app/routers/training.py` → 1: confirms no duplicate model
- `cd frontend && pnpm tsc --noEmit` → exit 0: TypeScript health signal

## Deviations

- NavBar was created as a separate file `frontend/app/NavBar.tsx` rather than inline in `layout.tsx`. This is equivalent functionally (task plan said "or in the same file") but cleaner for the T02 pattern reference.
- training.py ended up 237 lines (not 230 as the plan estimated) because the first copy included a blank comment separator and trailing newlines. The route count (4) and class count (1) are the authoritative checks, not the line count.

## Known Issues

None.

## Files Created/Modified

- `backend/app/routers/training.py` — removed 207-line duplicate block (lines 238–446); now 237 lines with exactly 4 route decorators
- `frontend/app/layout.tsx` — updated title to "RVC Studio", added bg-zinc-950 to body, imports and renders NavBar
- `frontend/app/NavBar.tsx` — new: 'use client' NavBar component with 4 tab links and usePathname active detection
- `frontend/app/page.tsx` — replaced create-next-app placeholder with full Library page (~220 lines)
