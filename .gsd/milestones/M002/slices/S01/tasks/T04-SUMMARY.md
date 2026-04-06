---
id: T04
parent: S01
milestone: M002
key_files:
  - backend/app/routers/training.py
key_decisions:
  - adv_loss and optimizer not read from profile DB row (not locked) — taken directly from request body each run
duration: 
verification_result: passed
completed_at: 2026-04-03T20:32:31.278Z
blocker_discovered: false
---

# T04: Extended StartTrainingRequest and /start endpoint in routers/training.py

**Extended StartTrainingRequest and /start endpoint in routers/training.py**

## What Happened

Extended StartTrainingRequest with adv_loss, kl_anneal, kl_anneal_epochs, optimizer fields. Passed all four to manager.start_job().

## Verification

py_compile passes

## Verification Evidence

| # | Command | Exit Code | Verdict | Duration |
|---|---------|-----------|---------|----------|
| 1 | `python3 -m py_compile backend/app/routers/training.py` | 0 | ✅ pass | 80ms |

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `backend/app/routers/training.py`
