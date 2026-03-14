---
estimated_steps: 5
estimated_files: 1
---

# T02: Build Training Page with Live WebSocket Log Stream

**Slice:** S04 — Next.js Frontend Polish & Service Scripts
**Milestone:** M001

## Description

Creates `frontend/app/training/page.tsx` — the Training tab. Users select any profile, set an epoch count, click Start Training, and watch log lines appear in real time as the WebSocket delivers them. A 6-phase progress bar advances as training moves through preprocess → extract_f0 → extract_feature → train → index → done. Cancel button sends POST `/api/training/cancel`. This task closes R002's front-end delivery gap.

## Steps

1. **Create `frontend/app/training/` and `page.tsx`**:
   - `'use client'` at top
   - Constants: `const API = 'http://localhost:8000'`, `const WS_BASE = 'ws://localhost:8000'`
   - TypeScript interfaces: `Profile { id: string; name: string; status: string }`, `TrainingMsg { type: 'log' | 'phase' | 'done' | 'error'; message?: string; phase?: string }`
   - Define phases array: `const PHASES = ['preprocess','extract_f0','extract_feature','train','index','done']`

2. **State and data fetching**:
   - `profiles: Profile[]` — fetched from `/api/profiles` on mount via `useEffect` with `cancelled` flag cleanup (mirror realtime page pattern exactly)
   - `selectedId: string` — `''` initially; set to first profile id when profiles load
   - `epochs: number` — default `20`; input `type="number"` min=1 max=200
   - `logLines: string[]` — append-only from WS messages; cap at 500 entries
   - `currentPhase: string | null` — updated from `msg.phase` on each WS message
   - `jobState: 'idle' | 'running' | 'done' | 'failed'` — drives button state
   - `wsRef: useRef<WebSocket | null>` — holds live WS connection

3. **Phase progress bar**:
   - Render 6 phase pills side by side; for each phase:
     - If phase === currentPhase → `bg-cyan-600 text-white` (current)
     - If PHASES.indexOf(phase) < PHASES.indexOf(currentPhase ?? '') → `bg-zinc-700 text-zinc-400` (done)
     - Else → `bg-zinc-900 text-zinc-600` (upcoming)
   - Phase labels are human-readable: `preprocess` → "Preprocess", `extract_f0` → "F0 Extract", `extract_feature` → "Features", `train` → "Train", `index` → "Index", `done` → "Done"
   - Render nothing if `currentPhase === null`

4. **Start Training handler**:
   - Guard: `if (!selectedId || jobState === 'running') return`
   - POST `${API}/api/training/start` with body `{ profile_id: selectedId, total_epoch: epochs }`
   - On success: set `jobState='running'`, clear `logLines`, reset `currentPhase` to null
   - Open `new WebSocket(\`${WS_BASE}/ws/training/${selectedId}\`)`; store in `wsRef.current`
   - `ws.onmessage`: parse JSON → if `type==='log'` and `msg.message` → append to logLines (slice to last 500); update `currentPhase` from `msg.phase` if present
   - `ws.onmessage`: if `type==='done'` → set `jobState='done'`, `currentPhase='done'`, close WS
   - `ws.onmessage`: if `type==='error'` → set `jobState='failed'`, append error message to logLines, close WS
   - `ws.onerror` / `ws.onclose` (when `jobState` is still `'running'`): append "(connection closed)" to logLines
   - Connect WS **after** POST returns job_id — not before

5. **Cancel handler**: POST `${API}/api/training/cancel` with body `{ profile_id: selectedId }`; close `wsRef.current` if open; set `jobState='idle'`

6. **Log area**: `<div ref={logRef}>` with `overflow-y-auto max-h-80 bg-zinc-950 rounded-lg p-3 font-mono text-[11px]`; each line in `text-zinc-300`; `useEffect` on `logLines` to auto-scroll: `logRef.current.scrollTop = logRef.current.scrollHeight`

7. **Cleanup**: `useEffect` return closes `wsRef.current` if open (component unmount safety)

8. **Run `pnpm tsc --noEmit`** from `frontend/` and fix all type errors.

## Must-Haves

- [ ] WS connection opened **after** POST `/api/training/start` returns (not before)
- [ ] `wsRef` closed on component unmount via `useEffect` cleanup
- [ ] `logLines` capped at 500 to prevent unbounded memory growth
- [ ] Phase bar renders and updates as `currentPhase` changes
- [ ] Cancel posts to `/api/training/cancel` and closes WS
- [ ] `pnpm tsc --noEmit` exits 0

## Verification

```bash
# TypeScript clean
cd frontend && pnpm tsc --noEmit
# → (no output, exit 0)
```

Browser verification against live backend:
1. Navigate to `http://localhost:3000/training`
2. Profile dropdown populates from `/api/profiles`
3. Select a trained profile; click "Start Training"
4. Log area shows incoming messages; phase bar advances
5. "Cancel" button appears and is clickable while running
6. After `{type:"done"}`, "Done" phase lights up cyan; log area shows final messages

## Observability Impact

- Signals added/changed: Training page surfaces WS message stream in the browser UI — phase and log lines are now visible without curl/wscat
- How a future agent inspects this: navigate to `/training` in browser; check browser console for WS errors or JSON parse failures; check Network tab for WS frame content
- Failure state exposed: `jobState='failed'` renders a red error banner at top; last error message from `{type:'error'}` appended to log area

## Inputs

- `frontend/app/realtime/page.tsx` — canonical pattern: `'use client'`, `const API`, typed interfaces, `useEffect` with `cancelled` flag, `useRef<WebSocket>`, WS onmessage/onerror/onclose, dark zinc/cyan palette
- `backend/app/routers/training.py` (post-T01 fix) — 4 clean routes; WS at `/ws/training/{profile_id}`
- `S04-RESEARCH.md` — WS message shapes: `{type:'log'|'phase'|'done'|'error', message?, phase?}`; phases: `preprocess → extract_f0 → extract_feature → train → index → done`
- T01 `app/layout.tsx` — tab nav already wired; Training tab link to `/training` already present

## Expected Output

- `frontend/app/training/page.tsx` — ~230 lines: profile dropdown, epoch input, Start/Cancel buttons, phase progress bar, scrollable log area, full WS lifecycle
- TypeScript compiles clean
- Training page accessible at `http://localhost:3000/training` with real WS log stream from live backend
