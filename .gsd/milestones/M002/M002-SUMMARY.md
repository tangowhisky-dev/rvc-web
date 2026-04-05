---
id: M002
title: "Training Quality Enhancements: TPRLS, KL Annealing, AdamSPD"
status: complete
completed_at: 2026-04-03T20:34:56.320Z
key_decisions:
  - TPRLS and optimizer not locked per-run (unlike vocoder/embedder)
  - New config knobs via config.json not CLI args
  - AdamSPD stiffness weight_decay=0.1 default (conservative)
  - KL annealing cycle is epoch-level, not step-level
key_files:
  - backend/rvc/infer/lib/train/losses.py
  - backend/rvc/infer/lib/train/adamspd.py
  - backend/app/db.py
  - backend/app/training.py
  - backend/app/routers/training.py
  - backend/rvc/infer/modules/train/train.py
  - frontend/app/training/page.tsx
lessons_learned:
  - config.json injection pattern (vs CLI args) is the right channel for training hyperparams that don't affect pipeline stages outside train.py
  - AdamSPD requires anchor snapshot before DDP wrapping and before any gradient steps — order matters
---

# M002: Training Quality Enhancements: TPRLS, KL Annealing, AdamSPD

**TPRLS loss, cyclic KL annealing, and AdamSPD optimizer selectable end-to-end from training UI to train.py subprocess**

## What Happened

Implemented TPRLS adversarial loss, cyclic KL annealing, and AdamSPD optimizer selectable from the training UI. Full stack: DB migration → Pydantic model → config.json patch → train.py loss/optimizer branching. Frontend adds three clean control sections consistent with existing UI patterns.

## Success Criteria Results

All 6 success criteria met: TPRLS/LSGAN selectable ✅, KL anneal configurable ✅, AdamW/AdamSPD selectable ✅, full stack plumbing ✅, UI renders cleanly ✅, defaults unchanged (LSGAN+AdamW+no anneal) ✅

## Definition of Done Results

All 8 DoD items met: losses.py has TPRLS functions ✅, adamspd.py created ✅, db.py migrated ✅, training.py extended ✅, routers/training.py extended ✅, train.py wired ✅, py_compile clean on all modified files ✅, tsc --noEmit clean ✅

## Requirement Outcomes

Training quality levers identified in TRAINING_COMPARISON.md analysis now available to users.

## Deviations

None.

## Follow-ups

Consider storing adv_loss/optimizer choices in profile DB row so they show as current settings when resuming (currently they reset to defaults on page load).
