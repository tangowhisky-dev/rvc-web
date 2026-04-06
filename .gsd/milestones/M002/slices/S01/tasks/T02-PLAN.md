---
estimated_steps: 5
estimated_files: 1
skills_used: []
---

# T02: DB migration: adv_loss, kl_anneal, kl_anneal_epochs, optimizer columns

Add 4 new columns to profiles table via _MIGRATIONS in db.py:
- adv_loss TEXT NOT NULL DEFAULT 'lsgan'
- kl_anneal INTEGER NOT NULL DEFAULT 0
- kl_anneal_epochs INTEGER NOT NULL DEFAULT 40
- optimizer TEXT NOT NULL DEFAULT 'adamw'

## Inputs

- `backend/app/db.py existing migration pattern`

## Expected Output

- `4 new entries in _MIGRATIONS list`

## Verification

python3 -m py_compile backend/app/db.py
