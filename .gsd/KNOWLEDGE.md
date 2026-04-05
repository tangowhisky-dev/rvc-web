# KNOWLEDGE — rvc-web

Append-only register of project-specific rules, patterns, and lessons learned.
Read at the start of every unit. Never remove or edit existing entries.

---

## Service Management

**Rule: Always use `start.sh` and `stop.sh` (in the project root `rvc-web/`) to start and stop services.**

Never start the backend or frontend manually (no raw `uvicorn`, `conda run`, `pnpm dev`, or `bg_shell` commands).
Never kill services manually with `lsof | kill` or `pkill` — use `stop.sh` instead.

Reasons:
- `start.sh` exports required env vars (`PROJECT_ROOT`, `KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) that the backend and training pipeline depend on.
- `start.sh` kills stale ports and dangling training/inference processes before starting.
- `stop.sh` shuts down worker subprocesses (`spawn_main`, `_realtime_worker`) before killing the backend, preventing orphaned GPU processes.
- `start.sh` writes PIDs to `.pids/` so `stop.sh` can do a clean shutdown.

Usage:
```bash
./start.sh   # start backend + frontend (from rvc-web/ root)
./stop.sh    # stop everything cleanly
```

---

## Path Handling

**Rule: Always convert `profile_dir` and `sample_dir` to absolute paths before passing to subprocesses.**

RVC training subprocesses (`preprocess.py`, `extract_f0_print.py`, `extract_feature_print.py`, `train.py`) run with `cwd=rvc_root`. Any relative path passed to them resolves against `rvc_root`, not the project root, causing `FileNotFoundError`.

Fix: call `os.path.abspath()` on `profile_dir` and `audio_dir` in `routers/training.py` immediately after construction.

---

## Audio Preprocessing

**Rule: Always pass `format="WAV"` explicitly to `soundfile.write()` when the output path has a non-standard extension.**

`soundfile` infers format from the file extension. Temp files like `moeed.wav.cleaning` have an unrecognised extension and cause `TypeError: No format specified`. Always pass `format="WAV"` (or the appropriate format) explicitly.

---

## Training Workspace

**Rule: The shared `logs/rvc_finetune_active/` directory is NOT profile-scoped — it must be treated as a temp workspace.**

When a profile is deleted, `routers/profiles.py` wipes `exp_dir` (`logs/rvc_finetune_active/`) entirely.
When training starts for a new profile, `training.py` checks a sentinel file (`.profile_id`) and wipes the workspace if the profile ID changed.
A fresh profile (no own checkpoints) must never inherit `G_latest.pth` / `D_latest.pth` from a prior profile — `_KEEP_FILES` is empty when `seeded_checkpoint=False`.

---

## Epoch Loss Tracking

**Rule: `epoch_losses` rows must be deleted explicitly on profile delete** — `aiosqlite` does not enforce `ON DELETE CASCADE` by default.

The delete handler in `routers/profiles.py` explicitly runs:
```sql
DELETE FROM epoch_losses WHERE profile_id = ?
DELETE FROM audio_files WHERE profile_id = ?
DELETE FROM profiles WHERE id = ?
```

## Directory Layout (post-refactor)

**Rule: `RVC_ROOT` env var is gone. Use `PROJECT_ROOT` instead.**

After the refactor the project root `rvc-web/` is the single source of truth:

```
rvc-web/
  assets/              ← pretrained weights, hubert, rmvpe (was RVC submodule/assets/)
  backend/
    app/               ← FastAPI application
    rvc/               ← RVC Python code (was Retrieval-based-Voice-Conversion-WebUI/)
      assets → ../../assets   (symlink so train.py relative writes work)
      logs   → ../../logs     (symlink so get_hparams() relative read works)
      configs/inuse/   ← v1/v2 JSON configs
      infer/           ← rtrvc.py, slicer2.py, train scripts
      rvc/             ← f0/, layers/, synthesizer.py, hubert.py
  data/
    profiles/          ← per-profile audio, checkpoints, model
    rvc.db             ← SQLite database
  frontend/            ← Next.js frontend
  logs/                ← training workspace (rvc_finetune_active/)
  start.sh             ← start backend + frontend
  stop.sh              ← stop everything
```

Training subprocess cwd = `backend/rvc/`. The `assets/` and `logs/` symlinks inside `backend/rvc/` make all relative paths in the upstream training scripts resolve correctly.

---

## MPS Training: 4D Constant Padding is Not Natively Supported

**Rule: Never use `F.pad(..., mode='constant')` on tensors with `ndim > 3` in any code that runs on MPS.**

### Background

PyTorch's MPS backend (Apple Silicon GPU) does not natively support constant-mode padding of tensors with more than 3 dimensions. When called, it silently falls back to a CPU-based "View Ops" implementation:

```
UserWarning: MPS: The constant padding of more than 3 dimensions is not currently
supported natively. It uses View Ops default implementation to run.
This may have performance implications.
```

This fires on **every training step** — not just at startup — because the attention layers call these functions in the forward pass. The cost is a silent MPS→CPU→MPS round-trip for every attention computation, degrading training throughput.

### Where it occurs in this codebase

`backend/rvc/rvc/layers/attentions.py` — `MultiHeadAttention._relative_position_to_absolute_position` and `_absolute_position_to_relative_position`. Both methods operate on 4D tensors `[batch, heads, length, ...]` and originally used `F.pad` with `mode='constant'` (default).

### Fix applied

Replace 4D `F.pad` constant-pad calls with `torch.cat` against a zero tensor of the required shape. This is semantically identical and uses only native MPS ops:

```python
# Before (triggers MPS fallback):
x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])  # pad last dim by 1

# After (native on MPS):
x = torch.cat(
    [x, torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)],
    dim=-1,
)
```

The 3D `F.pad` calls remaining in those same methods (on `x_flat` which is `[b, h, L]`) are fine and were left unchanged — MPS handles ≤3D constant padding natively.

**Verified**: full forward pass on MPS after the fix produces zero padding warnings and numerically identical outputs.

### General rule for future model changes

Any time you add a new layer or attention variant that uses `F.pad` on 4D+ tensors with `mode='constant'`, replace it with `torch.cat` + zeros. Reflect/replicate modes on 4D+ tensors are also unsupported on MPS and should be avoided similarly. Prefer `torch.cat`, index-based operations, or pre-allocate and fill with `torch.zeros`/`torch.ones`.

---

## MPS Training: Learning Rate Must Be Scaled with Batch Size

**Rule: When training on MPS with a reduced batch size, always scale the learning rate using the sqrt rule. Do NOT use the same LR as the CUDA baseline.**

### Background

The pretrained RVC weights were optimised at `batch_size=32, LR=1e-4`. For AdamW (and Adam generally), the appropriate batch-size scaling rule is:

```
LR_eff = LR_base × sqrt(batch_size / reference_batch_size)
```

Running the same LR with a smaller batch means each gradient update carries 4× more noise (at bs=8 vs bs=32) while the per-epoch LR decay schedule is unchanged. The result is a loss curve that oscillates at a higher floor and converges more slowly — observed as `loss_mel ≈ 36` at epoch 70 on MPS vs `≈ 23` on CUDA with the same epoch count.

### Effective LR by batch size

| batch_size | LR (sqrt-scaled) |
|---|---|
| 4  | 3.54e-5 |
| 8  | 5.00e-5 |
| 16 | 7.07e-5 |
| 32 | 1.00e-4 (unchanged) |

### Implementation

`_write_config` in `backend/app/training.py` auto-scales LR based on the batch size argument:

```python
_REFERENCE_BATCH_SIZE = 32
_BASE_LR = 1e-4
_scaled_lr = _BASE_LR * math.sqrt(batch_size / _REFERENCE_BATCH_SIZE)
cfg["train"]["learning_rate"] = round(_scaled_lr, 7)
```

This runs on every training start (fresh and continuation), so changing batch size mid-project takes effect immediately on the next run.

### Why sqrt and not linear?

The linear scaling rule (`LR ∝ batch_size`) is correct for SGD with momentum. For Adam/AdamW the adaptive per-parameter scaling partially compensates for gradient noise, so the optimal correction is closer to sqrt. Empirically, sqrt is the standard recommendation for fine-tuning transformer-style models with AdamW at varying batch sizes.


---

## MPS Training: fp16 AMP Must Be Disabled (Use fp32)

**Rule: Never train with `fp16_run=True` on MPS. Always use fp32 on Apple Silicon.**

### Background

MPS fp16 AMP has two compounding instabilities that compound over training:

**1. weight_norm underflow → NaN gradients → scale collapse**

This model has 137 `weight_norm` hooks. `weight_norm` computes:

```
weight = weight_g × (weight_v / ‖weight_v‖)
```

In fp16, `‖weight_v‖` can underflow to zero for small weights (common early in fine-tuning). This produces a division-by-zero → inf/NaN in the weight tensor → NaN gradient on that module's output → `GradScaler` detects the inf and halves its scale (65536 → 32768 → ...).

`GradScaler` defaults to `growth_interval=2000`: the scale only grows back after 2000 *consecutive* clean steps. At bs=8 with ~125 steps/epoch on MPS, that's ~16 epochs without a scale increase after each NaN event. During those 16 epochs the lower scale makes further underflow *more* likely — a positive feedback loop. Observed symptom: `loss_mel` oscillates at a high floor (~36) and refuses to converge, even as the loss steadily improves on CUDA.

**2. MPS fp16 fused op coverage is narrower than CUDA**

CUDA's fp16 implementation uses hardware-fused ops for reductions (softmax, layer_norm, exp) that maintain higher precision in intermediate accumulation. MPS does not provide equivalent fused paths — it executes the same ops in full fp16, producing more rounding error per step.

### Why fp16 isn't needed on MPS

fp16 on CUDA primarily saves **off-chip VRAM bandwidth** — tensors are half the size when transferred between DRAM and the GPU. MPS uses **unified memory** (CPU and GPU share the same physical RAM), so there is no separate off-chip transfer to optimize. fp32 on MPS uses the same memory bus as fp16 would. The throughput benefit is negligible; the stability cost is significant.

### Implementation

`_write_config` in `backend/app/training.py` sets `fp16_run` based on device availability:

```python
# fp16 only when CUDA is present; fp32 on MPS (unified memory, experimental AMP)
cfg["train"]["fp16_run"] = torch.cuda.is_available()
```

`torch.cuda.is_available()` is `True` on CUDA machines (fp16 enabled, no change) and `False` on MPS-only machines (fp32, stable).

### Interaction with GradScaler in train.py

When `fp16_run=False`, `train.py` still constructs `GradScaler(device, enabled=fp16_run)` and `autocast(device, enabled=fp16_run)`. Both are no-ops when `enabled=False` — no code changes needed in `train.py`.

---

## Best-Epoch Criterion: Use `loss_mel`, NOT `loss_gen`

**Rule: `G_best.pth` and `best_epoch` in DB must track the epoch with lowest epoch-average `loss_mel`, not `loss_gen`.**

### Root cause of the bug
`train.py` accumulated `_epoch_gen_sum` (raw adversarial `loss_gen`) and compared it against `lowest_loss_g`. `loss_gen` reaches equilibrium by epoch ~12 (bottoms at ~3.13 for this dataset) and then stays flat at 3.1–3.4 for the entire remainder of training. Using it as the best-epoch signal means `G_best.pth` freezes at epoch 12 and never updates again, even as mel loss continues dropping from 20 → 17 over the next 100+ epochs.

### Why `loss_mel`
- `loss_mel` is the L1 spectrogram reconstruction error — the direct proxy for inference output quality. The vocoder reconstructs audio from mel, so lower mel = better output.
- `loss_mel` decreases monotonically throughout training.
- `loss_gen` (discriminator-fooling term) and `loss_fm` (feature matching) are important for training stability but plateau early and add noise to epoch ranking.
- `loss_kl` (latent alignment) matters in early epochs (drops from ~9 to ~1.5) but provides little discrimination between late-stage epochs.

### Fix applied
- `lowest_loss_g` renamed to `lowest_mel` in `train.py`
- `avg_loss_g` replaced with `avg_mel = _epoch_mel_sum / _epoch_gen_count`
- Log line changed from `avg_loss_g=X` to `avg_mel=X`
- `_drain_stdout` regex updated from `avg_loss_g=` to `avg_mel=`
- OT detection updated to compare `loss_mel` (not `loss_gen`) against `_best_avg_gen`
- DB column `best_avg_gen_loss` now stores `avg_mel` value (column name unchanged for backward compat)

### For existing running jobs
The fix only takes effect on new training runs. For a run already in progress, manually copy `G_latest.pth` to `G_best.pth` and update the DB `best_epoch`/`best_avg_gen_loss` to match the current best mel epoch from `epoch_losses`.
