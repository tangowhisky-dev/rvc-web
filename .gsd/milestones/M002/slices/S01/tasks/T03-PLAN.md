---
estimated_steps: 4
estimated_files: 1
skills_used: []
---

# T03: Extend _write_config and _run_pipeline in training.py

1. Add adv_loss, kl_anneal, kl_anneal_epochs, optimizer params to _write_config; write them into cfg['train']
2. Add same params to _run_pipeline signature
3. Pass them through to _write_config call inside _run_pipeline
4. Add -adv_loss, -kl_anneal etc are NOT CLI args - they go via config.json

## Inputs

- `backend/app/training.py _write_config at line ~272`
- `backend/app/training.py _run_pipeline at line ~431`

## Expected Output

- `_write_config writes adv_loss/kl_anneal/kl_anneal_epochs/optimizer into config.json train block`
- `_run_pipeline accepts and threads new params`

## Verification

python3 -m py_compile backend/app/training.py
