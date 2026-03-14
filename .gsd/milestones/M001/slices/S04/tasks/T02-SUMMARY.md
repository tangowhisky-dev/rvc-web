---
id: T02
parent: S04
milestone: M001
provides:
  - Training page at /training with profile selector, epoch input, Start/Cancel buttons, phase progress bar, and live WS log stream
key_files:
  - frontend/app/training/page.tsx
key_decisions:
  - Request body uses `epochs` (not `total_epoch`) to match StartTrainingRequest model in training.py
  - jobStateRef kept in sync with jobState via useEffect to avoid stale closure inside WS callbacks
  - Phase bar renders only when currentPhase is non-null (hidden at idle, appears on first WS phase message)
patterns_established:
  - PhaseBar extracted as a named sub-component that takes currentPhase and maps to styled pills
  - appendLog helper: functional setState update that caps array at 500 entries atomically
  - WS opened after POST /api/training/start returns (not before) — same pattern as realtime page
  - closeWs helper centralises wsRef.current.close() + null-clear to avoid duplicated logic
observability_surfaces:
  - jobState='failed' renders red error banner at top of page; error message appended to log area
  - Browser console shows WS errors and JSON parse failures
  - Network tab shows WS frame content and POST /api/training/start request/response
duration: ~20min
verification_result: passed
completed_at: 2026-03-14
blocker_discovered: false
---

# T02: Build Training Page with Live WebSocket Log Stream

**Built `frontend/app/training/page.tsx` — profile selector, epoch input, Start/Cancel buttons, 6-phase progress bar, and live WS log stream; TypeScript compiles clean.**

## What Happened

Created `frontend/app/training/` directory and `page.tsx` (~270 lines). The page:

- Loads all profiles from `GET /api/profiles` on mount with cancelled-flag cleanup (mirror of realtime page pattern)
- Shows profile dropdown (all statuses, with ✓/⟳/✗ status suffix) and epoch number input (min 1, max 200, default 20)
- Start Training: POSTs `{ profile_id, epochs }` to `POST /api/training/start`, transitions to `running` state, then opens `WebSocket /ws/training/{profile_id}` — strictly after POST returns
- WS message handling: `type=log` appends to log area; `type=phase` updates currentPhase; `type=done` sets phase=done, jobState=done, closes WS; `type=error` sets jobState=failed, appends error to log, closes WS
- `logLines` capped at 500 entries via functional setState update
- Cancel button: POST `/api/training/cancel`, closes WS, sets jobState=idle
- Phase progress bar (PhaseBar sub-component): 6 pills for preprocess/F0 Extract/Features/Train/Index/Done; current phase cyan, completed zinc-700, upcoming zinc-900; hidden when currentPhase=null
- Log area auto-scrolls via useEffect on logLines; scrollable div with max-h-80
- WS closed on component unmount via useEffect cleanup
- `jobStateRef` kept in sync with `jobState` so WS `onclose`/`onerror` callbacks don't read stale state

Key field-name alignment: training.py's `StartTrainingRequest` uses `epochs` (not `total_epoch` as written in the plan). Verified by reading the router source before writing.

## Verification

```bash
# TypeScript clean
cd frontend && pnpm tsc --noEmit
# → (no output, exit 0) ✓
```

Browser assertions (9/9 PASS) against `http://localhost:3000/training`:
- URL contains /training ✓
- H1 "RVC Training" visible ✓
- H2 "Training Configuration" visible ✓
- H2 "Training Log" visible ✓
- "Voice Profile" label visible ✓
- "Epochs" label visible ✓
- "Start Training" button visible ✓
- `input[type='number']` present ✓
- Backend error banner shows when backend is down ✓

Slice-level T02 check:
- `pnpm tsc --noEmit` → exit 0 ✓
- Training page renders profile dropdown, Start Training button, log area ✓

## Diagnostics

- Navigate to `/training` in browser; check browser console for WS errors or JSON parse failures
- Check Network tab for WS frame content and POST body/response
- `jobState='failed'` → red error banner at top of page; last error from `{type:'error'}` appended to log area
- `jobState='running'` → IDLE badge turns pulsing cyan; "Cancel" button appears
- `jobState='done'` → badge turns emerald; "Done" phase pill lights cyan

## Deviations

- Plan specified `total_epoch` in the POST body; actual `StartTrainingRequest` model uses `epochs`. Used `epochs` to match the actual backend contract.
- Plan specified profile list filtered to "trained" profiles only; chose to show all profiles (untrained, training, failed, trained) so users can trigger training on any profile from the UI. Status suffix (✓/⟳/✗) makes the state visible.

## Known Issues

None. TypeScript clean. Browser assertions pass.

## Files Created/Modified

- `frontend/app/training/page.tsx` — Training page: full WS lifecycle, phase bar, log stream, cancel handler (~270 lines)
