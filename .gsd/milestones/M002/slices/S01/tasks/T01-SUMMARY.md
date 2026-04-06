---
id: T01
parent: S01
milestone: M002
key_files:
  - backend/rvc/infer/lib/train/losses.py
  - backend/rvc/infer/lib/train/adamspd.py
key_decisions:
  - AdamSPD weight_decay=0.1 as default stiffness (configurable); 0.5 in fork-4 was aggressive
  - TPRLS tau=0.04 matches fork-4 reference exactly
duration: 
verification_result: passed
completed_at: 2026-04-03T20:32:04.475Z
blocker_discovered: false
---

# T01: Added TPRLS disc/gen loss functions to losses.py and AdamSPD optimizer module

**Added TPRLS disc/gen loss functions to losses.py and AdamSPD optimizer module**

## What Happened

Added discriminator_tprls_loss and generator_tprls_loss to losses.py using median-centering approach from fork-4 reference. Created adamspd.py with a clean AdamSPD implementation that snapshots pretrain anchor weights and applies selective projection decay only when the gradient would push parameters further from the anchor.

## Verification

python3 -m py_compile on both files passes

## Verification Evidence

| # | Command | Exit Code | Verdict | Duration |
|---|---------|-----------|---------|----------|
| 1 | `python3 -m py_compile backend/rvc/infer/lib/train/losses.py && python3 -m py_compile backend/rvc/infer/lib/train/adamspd.py` | 0 | ✅ pass | 120ms |

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `backend/rvc/infer/lib/train/losses.py`
- `backend/rvc/infer/lib/train/adamspd.py`
