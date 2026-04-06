# S01: Backend: loss functions, optimizer, config plumbing, train.py wiring — UAT

**Milestone:** M002
**Written:** 2026-04-03T20:33:04.869Z

## S01 UAT

- [ ] Start training with default settings (LSGAN + AdamW + no KL anneal) — log shows `adv_loss=lsgan optimizer=adamw kl_anneal=False`
- [ ] Start training with TPRLS selected — log shows `adv_loss=tprls`
- [ ] Start training with AdamSPD selected — log shows `optimizer=adamspd`
- [ ] Start training with KL annealing enabled — log shows `kl_anneal=True`
- [ ] Training completes 2 epochs without error for all four combinations
