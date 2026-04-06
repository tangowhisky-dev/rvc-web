---
estimated_steps: 3
estimated_files: 1
skills_used: []
---

# T04: Extend StartTrainingRequest and /start endpoint in routers/training.py

1. Add adv_loss, kl_anneal, kl_anneal_epochs, optimizer fields to StartTrainingRequest Pydantic model
2. Read adv_loss/kl_anneal/kl_anneal_epochs/optimizer from profile DB row (read from profile, passed through)
3. Pass new fields to manager.start_job() call

## Inputs

- `backend/app/routers/training.py StartTrainingRequest at line ~37`
- `/start endpoint logic`

## Expected Output

- `StartTrainingRequest has 4 new optional fields with defaults`
- `start_job call includes new params`

## Verification

python3 -m py_compile backend/app/routers/training.py
