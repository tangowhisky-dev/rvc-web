# S02: Frontend: UI controls and POST wiring

**Goal:** Add three training config controls to the frontend training page: adversarial loss selector (LSGAN/TPRLS), KL annealing toggle + cycle input, optimizer selector (AdamW/AdamSPD). Wire all to handleStart POST body.
**Demo:** After this: Training page shows all three new controls; starting training with non-default settings works

## Tasks
- [x] **T01: Added adv_loss selector, KL anneal toggle, optimizer selector to training page UI** — 1. Add state: advLoss ('lsgan'|'tprls'), klAnneal (bool), klAnnealEpochs (number), optimizer ('adamw'|'adamspd')
2. Add adv_loss, kl_anneal, kl_anneal_epochs, optimizer to handleStart POST body
3. Render in training config section (after overtraining controls, before Start button):
   - 'Adversarial Loss' row: two tile buttons LSGAN / TPRLS with brief tooltip
   - 'KL Annealing' row: toggle switch; when on show cycle epochs number input (default 40)
   - 'Optimizer' row: two tile buttons AdamW / AdamSPD with brief tooltip
4. Style consistent with existing BatchSizeSelector / loss mode tiles
  - Estimate: 30min
  - Files: frontend/app/training/page.tsx
  - Verify: cd frontend && npx tsc --noEmit 2>&1 | head -20
