# S01: Backend: loss functions, optimizer, config plumbing, train.py wiring

**Goal:** Implement TPRLS loss, AdamSPD optimizer, and cyclic KL annealing through the full Python stack: losses.py → adamspd.py → db.py migration → training.py _write_config+_run_pipeline → routers/training.py → utils.py → train.py
**Demo:** After this: Training log shows 'adv_loss=tprls optimizer=adamspd kl_anneal=True' when configured

## Tasks
- [x] **T01: Added TPRLS disc/gen loss functions to losses.py and AdamSPD optimizer module** — 1. Add discriminator_tprls_loss and generator_tprls_loss to backend/rvc/infer/lib/train/losses.py
2. Create backend/rvc/infer/lib/train/adamspd.py with AdamSPD implementation (stiffness penalty toward pretrain anchor weights)
  - Estimate: 20min
  - Files: backend/rvc/infer/lib/train/losses.py, backend/rvc/infer/lib/train/adamspd.py
  - Verify: python3 -m py_compile backend/rvc/infer/lib/train/losses.py && python3 -m py_compile backend/rvc/infer/lib/train/adamspd.py
- [x] **T02: DB migration: added adv_loss, kl_anneal, kl_anneal_epochs, optimizer columns** — Add 4 new columns to profiles table via _MIGRATIONS in db.py:
- adv_loss TEXT NOT NULL DEFAULT 'lsgan'
- kl_anneal INTEGER NOT NULL DEFAULT 0
- kl_anneal_epochs INTEGER NOT NULL DEFAULT 40
- optimizer TEXT NOT NULL DEFAULT 'adamw'
  - Estimate: 10min
  - Files: backend/app/db.py
  - Verify: python3 -m py_compile backend/app/db.py
- [x] **T03: Extended _write_config and _run_pipeline in training.py with new params** — 1. Add adv_loss, kl_anneal, kl_anneal_epochs, optimizer params to _write_config; write them into cfg['train']
2. Add same params to _run_pipeline signature
3. Pass them through to _write_config call inside _run_pipeline
4. Add -adv_loss, -kl_anneal etc are NOT CLI args - they go via config.json
  - Estimate: 20min
  - Files: backend/app/training.py
  - Verify: python3 -m py_compile backend/app/training.py
- [x] **T04: Extended StartTrainingRequest and /start endpoint in routers/training.py** — 1. Add adv_loss, kl_anneal, kl_anneal_epochs, optimizer fields to StartTrainingRequest Pydantic model
2. Read adv_loss/kl_anneal/kl_anneal_epochs/optimizer from profile DB row (read from profile, passed through)
3. Pass new fields to manager.start_job() call
  - Estimate: 15min
  - Files: backend/app/routers/training.py
  - Verify: python3 -m py_compile backend/app/routers/training.py
- [x] **T05: Wired train.py: adv_loss branching, KL beta schedule, AdamSPD optimizer init** — 1. In train_and_evaluate: branch discriminator_loss / generator_loss on hps.train.adv_loss ('lsgan'|'tprls')
2. Import tprls functions from losses.py
3. Compute kl_beta: if hps.train.kl_anneal: beta = cyclic cosine over kl_anneal_epochs, else 1.0
4. Apply kl_beta: loss_kl = kl_loss(...) * hps.train.c_kl * kl_beta
5. In optimizer init block: branch on hps.train.optimizer ('adamw'|'adamspd'); for adamspd import and init AdamSPD with pretrain anchor
6. Log chosen adv_loss and optimizer at startup
  - Estimate: 30min
  - Files: backend/rvc/infer/modules/train/train.py
  - Verify: python3 -m py_compile backend/rvc/infer/modules/train/train.py
