# Target-Speaker F0 Prior — Analysis

How to move from "voice color" (current: mean pitch shift) to "voice behavior"
(target: the full pitch character of the target speaker).

---

## The Problem with Mean-Std Normalization

What we currently implement:

```
log_f0_out = log(μ_t) + (σ_t / σ_s) × (log_f0_in − log(μ_s))
```

This is a linear transform in log-Hz space. It correctly maps:
- The central pitch (μ_s → μ_t)
- The spread of pitch variation (σ_s → σ_t)

It does **not** map:
- **Distribution shape** — whether the speaker has one habitual register or two, whether their pitch is skewed toward high or low
- **Intonation dynamics** — how fast pitch moves between frames (slow/smooth vs choppy/expressive)
- **Speaking range** — the hard floor and ceiling the speaker actually uses
- **Voiced/unvoiced patterns** — how often the speaker goes silent vs sustains speech

A flat-speaking male mapped to an expressive female target will still be flat — just at the right center pitch. The output has the target's voice color but the source speaker's voice character.

---

## The Four Transforms

### Layer 1: Affine (current implementation)

```
log_f0_out = log(μ_t) + (σ_t/σ_s) × (log_f0_in − log(μ_s))
```

Controls: central pitch, distribution width.  
Storage: 2 floats per speaker (mean, std).  
Works well when source and target have similar distribution shapes. Fails when shapes differ substantially (e.g. bimodal target, flat source).

---

### Layer 2: Histogram Equalization

```
f0_out = CDF_t⁻¹ ( CDF_s(f0_in) )
```

Where CDF is the empirical cumulative distribution of voiced F0 values.

**What it does:** Maps the source frame to its rank within the source distribution, then maps that rank to the corresponding value in the target distribution. Since it's a monotone rank-preserving transform, a smooth pitch contour in produces a smooth pitch contour out — just in the target's register.

**What it fixes that affine cannot:**
- Bimodal targets (e.g. a speaker with a "neutral" register and an "emphasis" register): flat source maps correctly to the lower (habitual) mode rather than the wrong Hz level
- Skewed distributions: a target whose pitch is skewed toward high values is represented correctly — the output spends proportionally more time high
- Any higher-order distributional shape is exactly matched

**Key property:** Rank-preserving → monotone → if source is locally smooth, output is locally smooth. HistEQ does not introduce jaggedness.

**Storage:** An empirical CDF stored as a 256-bucket histogram over the log-Hz range 50–1400 Hz. Bucket width = 0.23 semitones. Size = 1 KB per speaker.

**Inference cost:** One `np.searchsorted` call per voiced frame. For a 30-second file (~6000 voiced frames), this is microseconds — negligible.

**Subsumes affine:** When both CDF distributions are Gaussian, HistEQ reduces exactly to affine. HistEQ is strictly more powerful.

---

### Layer 3: Velocity Normalization

```
Δlog_f0_out[t] = (σ_vel_t / σ_vel_s) × Δlog_f0_in[t]
log_f0_out[t]  = log_f0_out[t-1] + Δlog_f0_out[t]
```

Then recentered to preserve the mean established by Layer 1 or 2.

Where Δlog_f0[t] = log_f0[t] − log_f0[t-1] (log-Hz first difference, computed only over voiced frames).

**What it does:** Scales the frame-to-frame rate of pitch change to match the target speaker's intonation speed.

**Why it's independent of Layer 1/2:** σ_vel and σ_dist are not simply proportional. For an AR(1) speech model `x[t] = a·x[t-1] + (1-a)·μ + noise`:

```
σ_vel / σ_dist = √(2(1-a))
```

The AR coefficient `a` varies between speakers:
- Very smooth, slow speaker: a ≈ 0.95 → ratio = 0.32
- Normal conversational: a ≈ 0.85 → ratio = 0.55
- Fast/choppy speaker: a ≈ 0.70 → ratio = 0.77

A flat-speaking target (high a) and an expressive target (low a) have the same μ and σ but very different σ_vel. Affine normalization applied σ_t/σ_s to the distribution width but cannot independently control velocity. Velocity normalization is a genuinely orthogonal degree of freedom.

**Perceptually:** Controls whether the converted voice sounds deliberate (slow, smooth pitch contour) or animated (fast, frequent pitch excursions). A mismatch here is what produces "monotone" or "unnaturally expressive" conversion artifacts even when the mean pitch is correct.

**Implementation note:** Must be applied causally (only to voiced frames, gaps reset the accumulator). After scaling, drift-correct by subtracting `(running_mean − target_mean)` every N frames.

**Storage:** One float per speaker (σ_vel).

---

### Layer 4: Range Soft-Clip

```
center    = (log(P5_t) + log(P95_t)) / 2
half_range = (log(P95_t) − log(P5_t)) / 2
f0_out = exp( center + tanh( (log_f0 − center) / half_range × k ) × half_range / tanh(k) )
```

where k controls softness (k=2 gives a gentle squashing; k=5 approaches hard clip).

**What it does:** Applies a soft sigmoidal compression that confines output F0 to the target's speaking range without hard cutting. Values well within the range are unaffected; values far outside are asymptotically pulled toward the boundary.

**Why it matters:** After normalization, extreme frames can still produce out-of-character output — e.g. a source speaker who occasionally drops to a very low creaky voice will map to something that sounds wrong on a high-pitched target even if the mean is correct. Soft-clipping these outliers makes the conversion sound more consistently in-character.

**Soft-clip test (target range 170–380 Hz):**
```
100 Hz → 167 Hz  (far below floor, pulled up strongly)
150 Hz → 168 Hz  (below floor, pulled up)
200 Hz → 180 Hz  (just inside range, gently nudged)
250 Hz → 246 Hz  (center, nearly unchanged)
300 Hz → 337 Hz  (inside range, gently nudged)
350 Hz → 373 Hz  (near ceiling, gently pulled down)
450 Hz → 385 Hz  (above ceiling, pulled down)
```

**Storage:** Two floats per speaker (P5, P95). Already partially computable from the DIO pass.

---

## What Each Layer Adds Perceptually

| Transform stack | What's correct | What still sounds wrong |
|---|---|---|
| No transform | Timbre | Pitch is source speaker's entirely |
| Affine only (current) | Center pitch, spread | Distribution shape, dynamics, range |
| + HistEQ | Full distribution shape | Dynamics, extreme range excursions |
| + Velocity norm | Intonation speed / energy | Extreme range excursions |
| + Range soft-clip | Speaking range bounds | — |

The full stack produces a voice that has the target's **pitch behavior**: the characteristic registers (habitual vs emphasis), the rate of intonation movement, and the speaking range. Not just the right average pitch.

---

## Feasibility and Architecture

### What's buildable without retraining

All four transforms are **post-processing operations** on the pitch tensor after extraction. They require no changes to model weights, no new training, and no architectural changes. They're applied inside the `_patched_get_f0` (RVC) and `_apply_f0_norm_patch → sample_pitch` (B2) hooks we already have.

### What's not buildable without retraining

**Voiced/unvoiced rate matching** — the source speaker's voicing decisions (when to pause, when to sustain) come from RMVPE/PitchEstimator. Changing these post-hoc would silence or add audio segments. Not addressable without either retraining or a separate voicing model.

**Phrase-level declination / reset** — natural speech has an F0 trend that falls across a phrase and resets at phrase boundaries. Targets differ in how steeply they declinate. Would require phrase boundary detection (not currently in the pipeline). Research territory.

**Autoregressive F0 prior** — condition each F0 decision on a small LM trained on target F0 sequences. Would produce truly speaker-like intonation trajectories, not just matching distributions. Requires new training infrastructure. Not currently buildable.

**Diffusion-based F0 resynthesis** — replace the converted F0 curve with a sample drawn from a diffusion model trained on target speaker trajectories. Highest quality but requires new model training. Not currently buildable.

---

## Statistics to Store in `speaker_f0.json`

All computable from a single pass over training audio with DIO (same pass that already computes mean and std):

```json
{
  "mean_f0":    195.3,      // geometric mean Hz (already stored)
  "std_f0":     0.198,      // log-space std (already stored)
  "p5_f0":      142.0,      // 5th percentile Hz — speaking floor
  "p25_f0":     170.0,      // 25th percentile — habitual pitch lower bound
  "p50_f0":     193.0,      // median — modal pitch (most stable central tendency)
  "p75_f0":     222.0,      // 75th percentile — habitual pitch upper bound
  "p95_f0":     290.0,      // 95th percentile — speaking ceiling
  "vel_std":    0.072,      // log-Hz velocity std — intonation dynamics
  "voiced_rate": 0.68,      // fraction of frames voiced (diagnostic)
  "f0_hist":    [...]       // 256 floats — normalized CDF histogram for HistEQ
}
```

The `f0_hist` array holds the empirical CDF: `f0_hist[i]` = fraction of voiced frames with log(F0) ≤ `lo_log + (i+0.5) * bucket_width`. The bucket grid is fixed (log 50–1400 Hz, 256 buckets = 0.23 semitones/bucket) so histograms from different profiles are directly comparable.

---

## Recommended Implementation Order

The layers are ordered by impact-to-complexity ratio:

### 1. Percentiles + velocity std (extend `compute_speaker_f0`)

Single extra pass over the already-collected voiced F0 array. No new dependencies. Adds P5, P25, P50, P75, P95, vel_std, voiced_rate to `speaker_f0.json`. Backward compatible.

### 2. Range soft-clip (inference hook)

One tanh operation per voiced frame. Extend the existing `_patched_get_f0` (RVC) and `_normed_sample_pitch` (B2) with a final soft-clip stage. Requires: P5, P95 from speaker_f0.json.

### 3. Velocity normalization (inference hook)

Causal first-difference accumulator. Extends the same hooks. Requires: vel_std from speaker_f0.json for both target and source (source comes from `input_f0` endpoint). Handle voiced/unvoiced gaps: reset the accumulator when the silence spans more than N frames (don't propagate the gap as a "pitch change").

### 4. Histogram equalization (offline only)

Requires full source CDF — only available for offline inference where the input file is known upfront. Extend `compute_speaker_f0` to also compute and store the 256-bucket histogram. Extend `input_f0` endpoint to return the source histogram. Implement `histeq_transform(src_hist, tgt_hist, f0_hz)` as a vectorized numpy operation on the whole file before the block loop.

### 5. Rolling CDF for realtime (optional, post-HistEQ)

After ~5 seconds of voiced speech, a rolling buffer of ~1000 voiced F0 values is enough to estimate a reasonable source CDF. At that point, switch from affine to HistEQ mid-session. Adds latency and complexity to the realtime path; skip unless realtime quality is specifically a priority.

---

## Realtime vs Offline

The available information differs:

| | Offline | Realtime |
|---|---|---|
| Source CDF known | ✓ (full file before conversion) | ✗ (streaming, future unknown) |
| HistEQ applicable | ✓ | ✗ (or after warm-up delay) |
| Velocity norm | ✓ (src vel_std from input_f0) | ✓ (causal, 1-frame lag) |
| Range soft-clip | ✓ | ✓ |
| Affine | ✓ | ✓ |

**Offline:** Full pipeline. HistEQ + velocity + soft-clip.

**Realtime:** Affine + velocity + soft-clip. Optionally add rolling CDF after 5-second warm-up if latency is acceptable.

The realtime upgrade from current (affine only) to affine + velocity + soft-clip is substantial and requires no file-scope data. Velocity normalization and soft-clipping are both causal operations.

---

## Why Velocity Std is NOT Redundant with Distribution Std

For AR(1) speech:

```
σ_vel / σ_dist = √(2(1 − a))
```

Two speakers can have identical μ and σ_dist but different AR coefficients — one speaks smoothly (high a, low σ_vel) and one speaks choppily (low a, high σ_vel). Affine normalization scales the distribution correctly but cannot independently control the velocity because it treats each frame independently — it has no memory of the previous frame.

Velocity normalization is the first layer in this pipeline that is **temporally aware**. It introduces one frame of state (the previous output value) and scales the intonation trajectory accordingly.

---

## Summary

The full transformation stack treats pitch as a **time series drawn from a speaker-specific process**, not just a series of independent samples from a distribution. Each layer addresses one dimension of that process:

- **Affine**: where the distribution lives (center, spread)
- **HistEQ**: what shape the distribution has (skew, modality)
- **Velocity norm**: how fast the process moves through that distribution (dynamics)
- **Range soft-clip**: where the process is bounded (floor, ceiling)

The result is a pitch contour that sounds like it was produced by a speaker with the target's habitual registers, speaking energy, and range — not just the target's average pitch.
