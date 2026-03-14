---
estimated_steps: 4
estimated_files: 3
---

# T01: Fix Training Router Duplicate + Build Library Page

**Slice:** S04 — Next.js Frontend Polish & Service Scripts
**Milestone:** M001

## Description

Two independent changes that unblock the rest of the slice: (1) remove the duplicated body in `training.py` (lines 231–446) before the Training UI depends on stable endpoints, and (2) replace the create-next-app placeholder at `/` with the real Library page — profile list, status badges, upload form, and delete. Also adds the persistent tab nav to `app/layout.tsx` so all three existing routes appear under one shell. This is the first task, so no test file is created here — S04's verification is browser + TypeScript-compile based (not a new test suite), and the library page IS the verified output.

## Steps

1. **Fix `backend/app/routers/training.py`**: Delete the duplicate block starting at line 231 (second `class StartTrainingRequest:` through end of file, line 446). Only lines 1–230 should remain. Confirm with `grep -cE "^@router\.(post|get)|^@ws_router\.websocket" backend/app/routers/training.py` — must return `4`.

2. **Update `frontend/app/layout.tsx`**: Extract a `NavBar` client component inline (or in the same file with `'use client'` scoped to the NavBar function). Add tab links: Library (`/`), Training (`/training`), Realtime (`/realtime`), Setup (`/setup`). Use `usePathname()` for active-tab detection. Dark theme: `bg-zinc-950` for `<body>`, `bg-zinc-900/80 border-b border-zinc-800` for `<nav>`, `text-cyan-400 border-b-2 border-cyan-500` for active tab, `text-zinc-400 hover:text-zinc-200` for inactive. Update `<title>` to "RVC Studio". Keep Geist font variables and `antialiased` class. The layout itself stays a Server Component — only the NavBar sub-component is `'use client'`.

3. **Write `frontend/app/page.tsx`** (Library page):
   - `'use client'` at top
   - `const API = 'http://localhost:8000'` constant
   - TypeScript interfaces: `Profile { id: string; name: string; status: string; created_at: string; sample_path: string }`, `RealtimeStatus { active: boolean }`
   - State: `profiles: Profile[]`, `loading: boolean`, `uploading: boolean`, `sessionActive: boolean`, `nameInput: string`, `fileInput: File | null`, `error: string | null`
   - On mount: fetch `/api/profiles` + start 3s polling of `GET /api/realtime/status` (use `setInterval`, clear in cleanup)
   - Upload handler: build `FormData` with `name` and `file`; POST to `/api/profiles`; do NOT manually set `Content-Type`; on success refetch profiles; clear form
   - Delete handler: `window.confirm` → `DELETE /api/profiles/{id}` → refetch profiles
   - Status badge colors: `untrained` → `bg-zinc-700 text-zinc-300`, `training` → `bg-amber-900/50 text-amber-300 animate-pulse`, `trained` → `bg-cyan-900/50 text-cyan-300`, `failed` → `bg-red-900/50 text-red-300`
   - Session active banner at top if `sessionActive` is true (amber strip)
   - Empty state: "No voice profiles yet — upload one below." centered in a dashed zinc-800 border box
   - Form validation: both name (non-empty) and file must be set before submit button is enabled

4. **Run `pnpm tsc --noEmit`** from `frontend/` and fix any type errors until it exits 0.

## Must-Haves

- [ ] `backend/app/routers/training.py` has exactly 4 route decorators (2 `@router.post`, 1 `@router.get`, 1 `@ws_router.websocket`); no duplicate class definitions
- [ ] `app/layout.tsx` renders tab nav visible on all routes; active tab highlighted with cyan border; Geist fonts and `antialiased` preserved
- [ ] `app/page.tsx` fetches real profiles from `/api/profiles` and renders them (not hardcoded)
- [ ] Upload uses `FormData` without manual `Content-Type` header
- [ ] Delete calls `DELETE /api/profiles/{id}` after `window.confirm`
- [ ] `pnpm tsc --noEmit` exits 0

## Verification

```bash
# Training router has exactly 4 route decorators
grep -cE "^@router\.(post|get)|^@ws_router\.websocket" backend/app/routers/training.py
# → 4

# No duplicate class definitions
grep -c "class StartTrainingRequest" backend/app/routers/training.py
# → 1

# TypeScript clean
cd frontend && pnpm tsc --noEmit
# → (no output, exit 0)
```

Then open browser at `http://localhost:3000` (with backend running) and verify:
- Tab nav visible with 4 tabs; Library tab highlighted
- Profile list renders (or empty-state message)
- Upload form accepts name + file and POSTs successfully
- Delete button removes a profile

## Observability Impact

- Signals added/changed: none (this task does not add new backend signals; it fixes fragile double-registration which could cause silent routing confusion)
- How a future agent inspects this: `grep -c "class StartTrainingRequest" backend/app/routers/training.py` must return 1; TypeScript compile is the frontend health signal
- Failure state exposed: None new — the removed duplicate was silent; its absence is correct

## Inputs

- `backend/app/routers/training.py` — lines 231–446 are the duplicate block to remove; lines 1–230 are the correct single copy
- `frontend/app/layout.tsx` — bare default layout to augment with nav
- `frontend/app/page.tsx` — create-next-app placeholder to replace entirely
- `frontend/app/realtime/page.tsx` — pattern reference: `'use client'`, `const API`, typed interfaces, `useEffect` with `cancelled` flag, dark zinc/cyan palette

## Expected Output

- `backend/app/routers/training.py` — 230 lines, exactly 4 route decorators
- `frontend/app/layout.tsx` — persistent tab nav with `NavBar` client component; Server Component layout preserved
- `frontend/app/page.tsx` — Library page (~200 lines): profile list, status badges, upload form, delete, session-active indicator, empty state
- TypeScript compiles clean
