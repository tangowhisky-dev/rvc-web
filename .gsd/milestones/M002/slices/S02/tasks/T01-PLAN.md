---
estimated_steps: 7
estimated_files: 1
skills_used: []
---

# T01: Frontend: adv_loss selector, KL anneal toggle, optimizer selector

1. Add state: advLoss ('lsgan'|'tprls'), klAnneal (bool), klAnnealEpochs (number), optimizer ('adamw'|'adamspd')
2. Add adv_loss, kl_anneal, kl_anneal_epochs, optimizer to handleStart POST body
3. Render in training config section (after overtraining controls, before Start button):
   - 'Adversarial Loss' row: two tile buttons LSGAN / TPRLS with brief tooltip
   - 'KL Annealing' row: toggle switch; when on show cycle epochs number input (default 40)
   - 'Optimizer' row: two tile buttons AdamW / AdamSPD with brief tooltip
4. Style consistent with existing BatchSizeSelector / loss mode tiles

## Inputs

- `frontend/app/training/page.tsx handleStart at line ~914`
- `frontend/app/training/page.tsx loss mode tiles at lines ~1130-1205`

## Expected Output

- `3 new UI control groups in training config`
- `handleStart POST body includes 4 new fields`
- `TypeScript compiles clean`

## Verification

cd frontend && npx tsc --noEmit 2>&1 | head -20
