# Training Pipeline — Open Bugs

Status audit performed 2026-04-02 against current codebase.

---

## Bug 1 — RefineGAN resume corrupts speaker embedding (CRITICAL) ✅ FIXED

**Location:** `_write_config()` in `backend/app/training.py`

Fixed at line 372:
```python
_ckpt_to_probe = pretrain_g if pretrain_g else os.path.join(exp_dir, "G_latest.pth")
```
On resume (`pretrain_g=""`), reads `emb_g.weight.shape[0]` from `exp_dir/G_latest.pth`
instead of falling through to the 32k.json default of 109. RefineGAN checkpoints with
`spk_embed_dim=129` are now correctly preserved across continuation runs.

---

## Bug 2 — Overtraining early-stop produces no inference model (HIGH) ✅ FIXED

**Location:** Artifact copy phase in `_run_pipeline`

Fixed via `_make_infer_model()` helper (lines ~1490–1545). When `weights_pth` is absent
(overtraining SIGTERM before natural end), falls back to generating from `G_latest.pth`.
When neither exists (SIGTERM before first save), logs a clear error with guidance to
raise `overtrain_threshold`. `G_best.pth` is also generated into `model_best.pth`
independently.

---

## Bug 5 — `profile_embedding.pt` re-extracted every continuation run (LOW) ✅ FIXED

**Location:** Cleanup block in `_run_pipeline` (lines 583, 613)

Fixed — `profile_embedding.pt` added to `_KEEP_FILES` in both cleanup branches:
```python
_KEEP_FILES = ({"G_latest.pth", "D_latest.pth", "G_best.pth"} if seeded_checkpoint else set()) | {"profile_embedding.pt"}
```

---

## Bug 6 — Wrong default embedder path + non-fatal exit (LOW) ✅ FIXED

**Location:** `extract_feature_print.py`

Fixed:
- `assets/embedders/spin-v2` → `assets/spin-v2` in both path candidates
- Fairseq file-not-found `exit(0)` → `exit(1)` so `_run_subprocess` detects failure

---

## G_best.pth gap (LOW) ✅ FIXED

Fixed as part of Bug 5:
- `G_best.pth` added to `_KEEP_FILES` — preserved across continuation runs
- `G_best.pth` re-seeded into `exp_dir` after audio-changed wipe (alongside `G_latest.pth`)
- Already copied to `profile_dir/checkpoints/G_best.pth` in artifact phase (was working before)
- Already in fallback chain for `_make_infer_model` (Bug 2 fix)

---

## All bugs resolved. No open items.
