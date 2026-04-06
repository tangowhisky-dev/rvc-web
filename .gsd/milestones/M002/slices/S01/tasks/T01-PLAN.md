---
estimated_steps: 2
estimated_files: 2
skills_used: []
---

# T01: Add TPRLS functions to losses.py and AdamSPD module

1. Add discriminator_tprls_loss and generator_tprls_loss to backend/rvc/infer/lib/train/losses.py
2. Create backend/rvc/infer/lib/train/adamspd.py with AdamSPD implementation (stiffness penalty toward pretrain anchor weights)

## Inputs

- `codename-rvc-fork-4/rvc/train/losses.py (TPRLS reference)`
- `codename-rvc-fork-4/rvc/train/custom_optimizers/adamspd/adamSPD.py (AdamSPD reference)`

## Expected Output

- `losses.py with TPRLS functions`
- `adamspd.py new file`

## Verification

python3 -m py_compile backend/rvc/infer/lib/train/losses.py && python3 -m py_compile backend/rvc/infer/lib/train/adamspd.py
