---
estimated_steps: 6
estimated_files: 1
skills_used: []
---

# T05: Wire train.py: adv_loss branching, KL beta schedule, AdamSPD optimizer

1. In train_and_evaluate: branch discriminator_loss / generator_loss on hps.train.adv_loss ('lsgan'|'tprls')
2. Import tprls functions from losses.py
3. Compute kl_beta: if hps.train.kl_anneal: beta = cyclic cosine over kl_anneal_epochs, else 1.0
4. Apply kl_beta: loss_kl = kl_loss(...) * hps.train.c_kl * kl_beta
5. In optimizer init block: branch on hps.train.optimizer ('adamw'|'adamspd'); for adamspd import and init AdamSPD with pretrain anchor
6. Log chosen adv_loss and optimizer at startup

## Inputs

- `train.py discriminator loss at lines ~644-651`
- `train.py generator loss at lines ~660-730`
- `train.py optimizer init at lines ~273-284`
- `backend/rvc/infer/lib/train/losses.py (tprls functions)`
- `backend/rvc/infer/lib/train/adamspd.py`

## Expected Output

- `train.py uses hps.train.adv_loss to pick loss fn`
- `kl_beta cyclic schedule implemented`
- `AdamSPD branch in optimizer init`

## Verification

python3 -m py_compile backend/rvc/infer/modules/train/train.py
