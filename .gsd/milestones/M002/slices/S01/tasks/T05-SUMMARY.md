---
id: T05
parent: S01
milestone: M002
key_files:
  - backend/rvc/infer/modules/train/train.py
key_decisions:
  - KL annealing uses epoch-level position (epoch-1) % cycle, not step-level — cleaner signal
  - _adv_loss computed once per batch before D step and reused in G step
duration: 
verification_result: passed
completed_at: 2026-04-03T20:32:43.373Z
blocker_discovered: false
---

# T05: Wired train.py: adv_loss branching, KL beta schedule, AdamSPD optimizer init

**Wired train.py: adv_loss branching, KL beta schedule, AdamSPD optimizer init**

## What Happened

Imported TPRLS functions. Added optimizer branching at init (AdamW vs AdamSPD with anchor snapshot). Added discriminator loss branching (lsgan vs tprls). Added cyclic KL beta schedule. Added generator loss branching. Startup log line confirms chosen adv_loss and optimizer.

## Verification

py_compile passes

## Verification Evidence

| # | Command | Exit Code | Verdict | Duration |
|---|---------|-----------|---------|----------|
| 1 | `python3 -m py_compile backend/rvc/infer/modules/train/train.py` | 0 | ✅ pass | 200ms |

## Deviations

TPRLS generator loss uses y_d_hat_r (real outputs) in addition to y_d_hat_g since TPRLS is a paired relative loss — both are needed. Moved _adv_loss assignment outside the inner autocast block so it's shared between disc and gen sections.

## Known Issues

None.

## Files Created/Modified

- `backend/rvc/infer/modules/train/train.py`
