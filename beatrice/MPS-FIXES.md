# Beatrice Trainer — MPS Fixes

Surgical fixes applied to `trainer/beatrice_trainer/__main__.py` to make
training stable on Apple Silicon (MPS backend).

Legend:
- ✅ confirmed — stopped triggering after fix applied
- 🔬 in validation — fix applied, still accumulating evidence
- 🔍 diagnosing — probe in place, root cause not yet fully identified

---

## Overview

All failures share the same cascade: a single NaN enters any tensor in the
forward pass → propagates through loss → NaN gradients → corrupted Adam
moments → NaN weight update → all subsequent outputs NaN (non-recoverable
without the root cause fix).

NaN sources found so far fall into two categories:

**Forward-pass numerical instability** (fixes 1, 2, 4): specific operations
that produce NaN on MPS due to float32 precision limits or async buffer reuse.

**Weight corruption from NaN gradients** (fixes 3, 5, 6): once a single NaN
gradient escapes, it corrupts Adam state and then weights. These fixes either
repair corruption at load time or prevent it from accumulating per-step.

The investigation is ongoing — new NaN sources keep appearing as earlier ones
are plugged. Each fix removes one path; the next-weakest path then triggers.

---

## Fix 1 — `overlap_add`: Zero-denominator NaN in `fractional_part` ✅

### Location
`overlap_add()`, `fractional_part` computation.

### Root cause
`overlap_add` places pitched IRs at sub-sample positions. For each phase-wrap
pitchmark it computes:

```python
numer = 1.0 - phase[indices0, indices1]
denom = numer + phase[indices0, indices1 + 1]
fractional_part = numer / denom
```

On CUDA/CPU, `cumsum` runs in float64 — accumulated phase almost never lands
exactly on an integer boundary, so `denom > 0` always. On MPS, float64 is
unsupported so `cumsum` runs in float32 (~7 digits). Over a long sequence the
accumulated value can land *exactly* on an integer → `phase % 1.0 == 0.0` →
`numer == denom == 0` → `0.0 / 0.0 = NaN`.

### Fix
```python
fractional_part = numer / denom.clamp(min=1e-7)
```

When `denom == 0`, result is `0.0` — no sub-sample delay, physically correct
for an exact-boundary pitchmark. Inaudible.

### CUDA/CPU impact
No-op — float64 prevents exact zero. Negligible overhead.

---

## Fix 2 — Vocoder: `ir_amp` overflow from unclamped `.exp()` ✅

### Location
`Vocoder.forward()`, IR amplitude computation.

### Root cause
```python
ir_amp = ir[:, :ir.size(1)//2+1, :].exp()
```

`.exp()` overflows fp32 for `ir > ~88`. Early in training, before the network
stabilises, `ir` values can grow large from gradient noise (small batch sizes
on MPS: bs=4–8 vs bs=32 on CUDA). `inf` propagates through `irfft` →
NaN in the signal path → NaN gradients → corrupted Adam state.

More likely on MPS: smaller batch = noisier gradients; fp32 has more dynamic
range than fp16 so values that fp16 clips at 65504 grow unchecked.

### Fix
```python
ir_amp = ir[:, :ir.size(1)//2+1, :].clamp(max=10.0).exp()
```

`exp(10) ≈ 22026` — ~87 dB amplitude ratio, physically unreachable. Inaudible
at convergence; only activates during the unstable early phase.

### CUDA/CPU impact
Active on all devices — valid correctness guard for CUDA too. Negligible overhead.

---

## Fix 3 — Checkpoint repair: corrupted `weight_norm` parameters at load ✅

### Location
Checkpoint loading block (after `load_state_dict`).

### Root cause
If a prior session was terminated mid-step after a NaN gradient event, the saved
checkpoint can contain `weight_v` rows driven to zero or NaN by Adam.
`weight_norm` computes `weight = weight_g × (weight_v / ‖weight_v‖)` — zero or
NaN `weight_v` → NaN weight → NaN output on iteration 1 of the resumed run.
The per-iteration fixes (1, 2) prevent new NaN but cannot repair saved state.

### Fix
After `load_state_dict`, scan all `weight_v`/`weight_g` tensors in `net_g` and
`net_d`. Reinitialise non-finite or near-zero (`< 1e-12`) values with small
random noise (`1e-3`). Also zero out non-finite Adam `exp_avg`/`exp_avg_sq`
state for those parameters — otherwise the first optimizer step re-corrupts them.

### CUDA/CPU impact
Runs at checkpoint load time only. No per-step overhead. Valid on all devices.

---

## Fix 4 — Generator: `.detach()` + `.where()` on aliased MPS buffer ✅

### Location
`GeneratorNetwork.forward_and_compute_loss()`, after `self()` returns.

### Root cause
```python
y_hat_all = y_hat_all.detach().where(y_all == 0.0, y_hat_all)
```

`y_hat_all.detach()` shares underlying storage with `y_hat_all`. On MPS,
`torch.where()` with aliased input/output buffers can produce NaN
intermittently due to buffer reuse during async kernel dispatch.

Evidence: adding `.isfinite()` diagnostic probes (which force MPS sync + keep
tensor references alive) suppressed the crash; removing them reactivated it.
`torch.mps.synchronize()` alone did *not* fix it — ruling out a pure race
condition, pointing instead to buffer lifetime / aliasing.

### Fix
```python
_y_hat_detached = y_hat_all.detach() if torch.cuda.is_available() \
                  else y_hat_all.detach().clone()
y_hat_all = _y_hat_detached.where(y_all == 0.0, y_hat_all)
```

`.clone()` forces a new MPS buffer, breaking the aliasing. CUDA path unchanged.

### CUDA/CPU impact
MPS-only `.clone()`. No-op on CUDA.

---

## Fix 5 — Discriminator: `weight_norm` norm collapse after ~40 iterations 🔬

### Location
Training loop, after `grad_scaler.step(optim_d)`.

### Root cause
After ~40 iterations of unconstrained Adam updates on MPS float32, `weight_v`
rows in the discriminator's `weight_norm` Conv2d layers drift toward zero. No
gradient clipping was applied to the discriminator. Once `‖weight_v‖ < ε`,
`weight_norm` produces a NaN weight → NaN discriminator output → NaN
`loss_discriminator`.

Evidence:
- NaN appeared only in `loss_discriminator`, not in `y_hat` or generator losses
- Crash iteration was non-deterministic (4, 7, ~40 in different runs)
- `.isfinite()` probe intermittently suppressed it (MPS sync timing side effect)
- `torch.mps.synchronize()` before the discriminator call did not fix it

### Fix
Two complementary guards after the optimizer step:

**1. Gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(net_d.parameters(), 1.0)
```

**2. Post-step `weight_v` norm guard (MPS only):**
```python
if not _is_cuda:
    for module in net_d.modules():
        if hasattr(module, "weight_v"):
            wv = module.weight_v
            norms = wv.flatten(1).norm(dim=1)
            bad = norms < 1e-6
            if bad.any():
                with torch.no_grad():
                    module.weight_v[bad] = torch.randn_like(module.weight_v[bad]) * 1e-2
```

### CUDA/CPU impact
Grad clipping at 1.0 is active on all devices — discriminator training doesn't
require norms above 1.0. Per-step `weight_v` scan is MPS-only.

---

## Fix 6 — Generator weights NaN after ~13 iterations 🔍

### Location
`GeneratorNetwork.forward()`, before the vocoder call.

### Observation
At iteration ~13 (after fixes 1–5), the vocoder check fires with
`x_ok=False, speaker_emb_ok=False`. Both `x` (hidden state) and
`speaker_embedding` are NaN entering the vocoder, while `pitch` is clean.

`x` is built from:
```python
x = embed_phone(phone)
  + embed_quantized_pitch(quantized_pitch)
  + embed_pitch_features(pitch_features)
  + embed_speaker(target_speaker_id)
  + embed_formant_shift(formant_shift_indices)
```

`speaker_embedding` comes from `key_value_speaker_embedding(target_speaker_id)`.

Both share `target_speaker_id` as input — but integer indices can't be NaN.
The NaN must be in the *weights* of one or more of these embedding/conv modules,
corrupted by a NaN gradient propagating back through the generator after ~13
Adam steps.

### Current state
An extended probe is in place that lists every non-finite parameter by name when
this condition fires. Awaiting the next crash to identify the specific module.

### Hypothesis
The gradient balancer (`GradBalancer`) rescales gradients from multiple loss
terms. On MPS float32, rescaled gradients can overflow or underflow in specific
parameter groups. The ConvNeXt blocks in `embed_pitch_features` or the prenet
are the most likely candidates given their depth and use of `weight_standardization`.

---

## Diagnostic probes (temporary, currently in code)

Two RuntimeError probes remain pending confirmation:

1. **Vocoder input check** (`GeneratorNetwork.forward`, before `self.vocoder()`):
   Reports `x_ok`, `pitch_ok`, `speaker_emb_ok` and lists all non-finite
   parameter names when inputs are bad.

2. **Generator output check** (`forward_and_compute_loss`, after `self()`):
   Fires if `y_hat_all` is non-finite immediately after the generator forward.

To be removed once training runs stably for 500+ iterations.

---

## Summary

| # | Fix | Location | Trigger | Status |
|---|-----|----------|---------|--------|
| 1 | `denom.clamp(min=1e-7)` | `overlap_add` | float32 cumsum exact-zero | ✅ |
| 2 | `.clamp(max=10.0).exp()` | `Vocoder.forward` | exp overflow, noisy gradients | ✅ |
| 3 | `_repair_weight_norm` at load | checkpoint loader | corrupted `weight_v` in saved state | ✅ |
| 4 | `.detach().clone()` before `.where()` | `forward_and_compute_loss` | MPS buffer aliasing | ✅ |
| 5 | grad clip + post-step `weight_v` guard | training loop | discriminator `weight_v` norm collapse | 🔬 |
| 6 | TBD — generator weight NaN after ~13 iter | `GeneratorNetwork.forward` | generator weights corrupted by NaN gradient | 🔍 |

**Pattern**: each fix removes one NaN path; the next-weakest path then activates.
The root problem is that MPS float32 training has no fp16 GradScaler to bound
gradient magnitudes, making the model sensitive to numerical instabilities that
CUDA's fp16 training naturally suppresses.
