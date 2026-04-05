---
id: S01
parent: M002
milestone: M002
provides:
  - TPRLS and AdamSPD usable when frontend sends correct fields
  - DB columns ready for frontend to store/read settings
requires:
  []
affects:
  - S02
key_files:
  - backend/rvc/infer/lib/train/losses.py
  - backend/rvc/infer/lib/train/adamspd.py
  - backend/app/db.py
  - backend/app/training.py
  - backend/app/routers/training.py
  - backend/rvc/infer/modules/train/train.py
key_decisions:
  - New params flow via config.json not CLI args
  - adv_loss/optimizer not locked per-run (user can change freely)
  - KL annealing epoch-level cycle position
  - AdamSPD stiffness 0.1 default
patterns_established:
  - New training config knobs go via config.json patch in _write_config, not CLI args — consistent with c_spk/loss_mode pattern
observability_surfaces:
  - train.py logs optimizer= adv_loss= kl_anneal= at epoch 1 startup
drill_down_paths:
  []
duration: ""
verification_result: passed
completed_at: 2026-04-03T20:33:04.869Z
blocker_discovered: false
---

# S01: Backend: loss functions, optimizer, config plumbing, train.py wiring

**Full Python stack plumbing for TPRLS, KL annealing, and AdamSPD complete and syntax-clean**

## What Happened

All Python-side plumbing implemented. TPRLS loss functions, AdamSPD module, DB migrations, config.json patching, subprocess command, and train.py loss/optimizer branching all in place and syntax-verified.

## Verification

All 5 Python files pass py_compile. train.py branches on hps.train.adv_loss and hps.train.optimizer. KL beta schedule implemented.

## Requirements Advanced

-  — TPRLS, KL annealing, AdamSPD all implemented end-to-end in Python stack

## Requirements Validated

None.

## New Requirements Surfaced

None.

## Requirements Invalidated or Re-scoped

None.

## Deviations

None.

## Known Limitations

None.

## Follow-ups

None.

## Files Created/Modified

- `backend/rvc/infer/lib/train/losses.py` — Added discriminator_tprls_loss and generator_tprls_loss
- `backend/rvc/infer/lib/train/adamspd.py` — New AdamSPD optimizer module with stiffness projection decay
- `backend/app/db.py` — Added adv_loss/kl_anneal/kl_anneal_epochs/optimizer column migrations
- `backend/app/training.py` — Extended _write_config, _run_pipeline, start_job with 4 new params
- `backend/app/routers/training.py` — Extended StartTrainingRequest and start_job call
- `backend/rvc/infer/modules/train/train.py` — Loss branching, KL beta schedule, AdamSPD optimizer init, TPRLS imports
