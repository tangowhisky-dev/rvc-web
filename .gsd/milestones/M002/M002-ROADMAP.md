# M002: Training Quality Enhancements: TPRLS, KL Annealing, AdamSPD

## Vision
Expose three training quality levers — TPRLS adversarial loss, cyclic KL annealing, and AdamSPD optimizer — through the full stack from training UI to subprocess, improving voice cloning convergence quality without changing the inference pipeline.

## Slice Overview
| ID | Slice | Risk | Depends | Done | After this |
|----|-------|------|---------|------|------------|
| S01 | Backend: loss functions, optimizer, config plumbing, train.py wiring | medium | — | ✅ | Training log shows 'adv_loss=tprls optimizer=adamspd kl_anneal=True' when configured |
| S02 | Frontend: UI controls and POST wiring | low | S01 | ✅ | Training page shows all three new controls; starting training with non-default settings works |
