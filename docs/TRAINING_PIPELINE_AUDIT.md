# Training Pipeline Audit

**Date:** 2026-03-31  
**Status:** Validated findings from external audit + additional findings

---

## Bug 1 — RefineGAN resume corrupts speaker embedding table (CRITICAL)

**Location:** `_write_config()` in `backend/app/training.py`

### What Happens

On resume (prior_epochs > 0), `pretrain_g = ""` so `_write_config` has no checkpoint to read `spk_embed_dim` from. It writes `config.json` with the 32k.json default of `spk_embed_dim=109`. But a RefineGAN-trained `G_latest.pth` has `emb_g.weight` shape `[129, 256]`. When `train.py` calls `load_checkpoint`, it detects the shape mismatch (109 vs 129), logs a warning, and resets `emb_g.weight` to random init. The entire speaker embedding table is discarded on every continuation run.

**HiFi-GAN is unaffected** — its pretrain also uses `spk_embed_dim=109`, matching the default.

### Validation

**CONFIRMED.** Looking at `_write_config()` lines 326-333:
```python
if pretrain_g and os.path.isfile(pretrain_g):
    try:
        ckpt = _torch.load(pretrain_g, map_location="cpu", weights_only=True)
        emb_shape = ckpt.get("model", {}).get("emb_g.weight")
        if emb_shape is not None:
            cfg["model"]["spk_embed_dim"] = emb_shape.shape[0]
    except Exception:
        pass
```

This block only runs when `pretrain_g` is provided. On resume, `pretrain_g = ""` (line 891), so the block is skipped entirely. The config.json retains the default `spk_embed_dim=109` from 32k.json.

### Impact

Without this fix, every continuation training run for a RefineGAN profile destroys the speaker embedding layer. The model effectively re-learns speaker identity from scratch each time, making multi-run training useless for RefineGAN.

### Suggested Fix

In `_write_config`, when `pretrain_g` is empty (resume), look for `G_latest.pth` inside `exp_dir` and read `emb_g.weight.shape[0]` from it instead:

```python
# in _write_config, after the existing pretrain_g block:
if not pretrain_g:
    g_resume = os.path.join(exp_dir, "G_latest.pth")
    if os.path.isfile(g_resume):
        try:
            ckpt = _torch.load(g_resume, map_location="cpu", weights_only=True)
            emb_shape = ckpt.get("model", {}).get("emb_g.weight")
            if emb_shape is not None:
                cfg["model"]["spk_embed_dim"] = emb_shape.shape[0]
        except Exception:
            pass
```

---

## Bug 2 — Overtraining early-stop produces no inference model (HIGH)

**Location:** Artifact copy phase in `_run_pipeline` + `train.py` design

### What Happens

`save_small_model()` in `train.py` is only called at two points:
1. At `epoch >= hps.total_epoch` (the natural end)
2. Inside the `save_every_weights == "1"` block

We pass `-sw 0`, so the second path is disabled. When the overtraining detector sends SIGTERM, `train.py` exits immediately — `save_small_model` never runs. `assets/weights/rvc_finetune_active.pth` does not exist. `training.py` detects this, warns, and sets `dest_infer_path = None`. The DB update writes `model_path = NULL`. Inference (realtime and offline) fails for that profile — both routers check for `model_infer.pth` and return 400 if missing.

**Compound risk:** If the overtraining trigger fires before `effective_save_every` epochs have elapsed (e.g. `overtrain_threshold=3`, `total_epoch=100` → fires at epoch 3, first save at epoch 10), `G_latest.pth` also doesn't exist. No checkpoint, no inference model. The entire training run produces nothing usable.

### Validation

**CONFIRMED.** Looking at lines 1180-1196:
```python
is_overtrain_stop = overtrain_threshold > 0 and train_proc.returncode in (
    -_signal.SIGTERM,
    _signal.SIGTERM,
)
if not is_overtrain_stop:
    await _fail("train", err_msg)
    return
# Overtraining stop — log it and continue to artifact phase
```

Then at lines 1287-1304:
```python
if os.path.exists(weights_pth):
    shutil.copy2(weights_pth, dest_infer_path)
else:
    dest_infer_path = None  # ← No inference model!
```

The DB update at line 1343 will write `model_path = NULL` when `dest_infer_path` is None.

### Impact

Overtraining-stopped runs leave the profile in a "trained" status with no usable model. The user sees success in the UI but inference fails.

### Suggested Fix (Two Parts)

**Part A** — After the overtraining stop, if `weights_pth` is missing but `G_latest.pth` exists, call `save_small_model` directly from `training.py`:

```python
if not os.path.exists(weights_pth) and os.path.exists(g_latest):
    # Generate inference model from resume checkpoint
    import sys as _sys
    _sys.path.insert(0, rvc_pkg_dir)
    try:
        import json as _json
        from infer.lib.train.process_ckpt import save_small_model as _ssm
        _g_ckpt = torch.load(g_latest, map_location="cpu", weights_only=True)
        _state = _g_ckpt.get("model", _g_ckpt)
        # Build minimal hps from config.json
        with open(os.path.join(exp_dir, "config.json")) as _f:
            _cfg_d = _json.load(_f)
        # ... populate HParams and call _ssm(...)
    except Exception as _e:
        logger.warning(f"Could not generate inference model from checkpoint: {_e}")
```

**Part B** — Log a clear error when `G_latest.pth` is also missing (overtrain fired before first save), so the user understands what happened and can adjust `overtrain_threshold`.

---

## Bug 3 — `_last_completed_epoch` initialization (N/A — Not a Bug)

**Location:** `_run_pipeline`, line ordering

### Validation

**NOT A BUG.** The variable `_last_completed_epoch` is initialized to `prior_epochs` at the top of the function. The `prior_epochs` reset guard (lines 470-480) runs before this initialization. So `_last_completed_epoch` correctly picks up the post-reset value. No action needed.

---

## Bug 4 — `new_total_epochs` uses `_last_completed_epoch` (N/A — Not a Bug)

**Location:** DB update at end of `_run_pipeline`

### Validation

**NOT A BUG.** `_last_completed_epoch` is seeded at `prior_epochs`. If `train.py` crashes immediately (e.g. OOM on the first batch, before any epoch completes), no `epoch_done` events are emitted. `_last_completed_epoch` stays at `prior_epochs`. The DB `total_epochs_trained` is written as `prior_epochs` — unchanged, which is correct. No regression.

---

## Bug 5 — `profile_embedding.pt` deleted on every continuation run (LOW)

**Location:** Same-profile, no-files-changed cleanup block

### What Happens

Every continuation run re-extracts the profile speaker embedding even though the audio files haven't changed. The extraction takes ~20-60s depending on audio length. It's not wrong — the output is deterministic — just wasteful.

### Validation

**CONFIRMED.** Looking at lines 564-568:
```python
_KEEP_DIRS = {"0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"}
_KEEP_FILES = {"G_latest.pth", "D_latest.pth"} if seeded_checkpoint else set()
```

`profile_embedding.pt` is not in `_KEEP_FILES`, so it gets deleted during cleanup and re-extracted.

### Impact

Minor performance improvement on continuation runs. No correctness impact.

### Suggested Fix

Add `"profile_embedding.pt"` to `_KEEP_FILES`:

```python
_KEEP_FILES = ({"G_latest.pth", "D_latest.pth"} if seeded_checkpoint else set()) | {"profile_embedding.pt"}
```

---

## Bug 6 — Feature extraction uses wrong default embedder path (LOW)

**Location:** `backend/rvc/infer/modules/train/extract_feature_print.py`, `_default_embedder_path()` function

### What Happens

When `embedder_path` arg is missing or empty, the fallback resolves to `assets/embedders/spin-v2` — but the actual directory is `assets/spin-v2` (no `embedders/` subdirectory).

`training.py` always passes the explicit embedder path, so this fallback never fires in production. But if someone invokes `extract_feature_print.py` directly without the arg, it silently uses a non-existent path, which causes `_use_hf = False` → falls through to the fairseq path → file-not-found → `exit(0)` (not `exit(1)`) → silent empty feature extraction.

### Validation

**CONFIRMED.** Looking at lines 77-89:
```python
def _default_embedder_path() -> str:
    project_root = os.environ.get("PROJECT_ROOT", "")
    if project_root:
        candidate = os.path.join(project_root, "assets", "embedders", "spin-v2")  # WRONG
        if os.path.isdir(candidate):
            return candidate
        return os.path.join(project_root, "assets", "hubert", "hubert_base.pt")
    return os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "..", "assets", "embedders", "spin-v2",  # WRONG
    )
```

The actual path is `assets/spin-v2/`, not `assets/embedders/spin-v2/`.

### Impact

Minor — only affects manual invocation of `extract_feature_print.py`. Production training always passes the explicit path.

### Suggested Fix

Update `_default_embedder_path()` to use `assets/spin-v2` not `assets/embedders/spin-v2`. Also change the non-fatal `exit(0)` to `exit(1)` so `_run_subprocess` detects the failure.

---

## Bug 7 — `extract_feature_print.py` argument parsing is confusing (LOW)

**Location:** `extract_feature_print.py`, top-level argument parsing

### What Happens

The script parses `argv` with three branches:
- `len(sys.argv) == 7` → 6-arg legacy form (no `i_gpu`)
- `len(sys.argv) == 8` → new form without `i_gpu`
- `else` → old form with `i_gpu` at position 4

`training.py` passes 9 args (`device, n_part, i_part, i_gpu, exp_dir, version, is_half, embedder_path`). That hits the `else` branch which assigns correctly. The branching is confusing but functionally correct for the actual call path.

### Validation

**CONFIRMED — Not a runtime bug.** The comment in the `len==8` branch is misleading but the actual 9-arg call path works correctly. This is a maintenance hazard only.

### Impact

Maintenance only. No runtime bug.

---

## Summary Table

| # | Severity | Component | Description | Affects |
|---|----------|-----------|-------------|---------|
| 1 | **Critical** | `_write_config` | RefineGAN resume resets `emb_g` to random via shape mismatch | RefineGAN, all continuation runs |
| 2 | **High** | artifact copy + `train.py` | Overtraining stop produces no `model_infer.pth`; possibly no `G_latest.pth` either | All vocoder/embedder combos with overtraining enabled |
| 3 | N/A | `_last_completed_epoch` init | Apparent ordering issue but actually correct after prior-session guard | — |
| 4 | N/A | `new_total_epochs` | Correctly uses observed epochs, not requested epochs | — |
| 5 | Low | cleanup | `profile_embedding.pt` re-extracted every continuation run unnecessarily | All profiles with `c_spk > 0` |
| 6 | Low | `extract_feature_print.py` | Wrong default embedder path + non-fatal exit code | Manual invocation only |
| 7 | Low | `extract_feature_print.py` | Three-branch arg parsing is confusing but functionally correct | Maintenance only |

**Bugs 1 and 2 are the only ones that corrupt training or silently break inference.** Bug 1 is the most severe because it's silent and affects every continuation run for RefineGAN profiles. Bug 2 is triggered only when overtraining detection is enabled.
