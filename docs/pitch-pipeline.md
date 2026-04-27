# Pitch Handling in rvc-web

How pitch is extracted, shifted, and conditioned in both the RVC and Beatrice 2 pipelines, plus how auto pitch-shift works for offline inference.

---

## 1. RVC Pitch Pipeline

### 1.1 Training — what the model learns

During preprocessing, F0 is extracted from every training audio file (same extractor configured at training time — RMVPE by default). The raw Hz values are converted to a mel-scale coarse bin (1–255, 0 = unvoiced) and stored alongside the HuBERT content feature frames on disk.

The synthesizer (`SynthesizerTrnMsNSFsid`) is trained to receive **both** content features and the F0 curve as separate conditioning signals. The NSF generator (`NSFGenerator` / `SourceModuleHnNSF`) uses the F0 directly to synthesize a harmonic excitation signal — it builds sinewaves at the fundamental frequency and harmonics — then the HiFi-GAN or RefineGAN residual stack shapes that excitation into the target voice's timbre.

**The model never learns the target speaker's pitch distribution.** It learns: given this content embedding + this F0 curve + speaker embedding → produce audio that sounds like speaker 0 at exactly that F0.

### 1.2 Inference — pitch extraction and shift

Code path: `rtrvc.py → infer()` → `_get_f0()` → `rvc/f0/gen.py → post_process()`

**Step 1 — F0 extraction:**  
`RMVPE` runs on the most recent `f0_extractor_frame` samples of the rolling input buffer (16 kHz). RMVPE is a deep neural network trained specifically for robust pitch estimation; it handles noisy and whispery speech better than traditional YIN/DIO approaches. Returns `pitch` (coarse mel-bin integers 1–255) and `pitchf` (raw Hz floats).

**Step 2 — Pitch shift (applied before the model sees it):**  
In `gen.py → post_process()`:
```python
f0 = f0 * pow(2, f0_up_key / 12)   # f0_up_key = user_semitones - formant_semitones
```
This is a pure frequency multiplication on the **source speaker's** F0 curve. The model receives the already-shifted values — it never touches the raw source pitch. `f0_up_key = key - formant_shift` where `key` is the user's semitone slider and `formant_shift` compensates for formant shifting.

After the Hz shift, values are re-quantized to mel-bin integers (1–255) for the content encoder and kept as raw Hz for the NSF vocoder.

**Step 3 — NSF vocoder conditions on the shifted F0:**  
`pitchf` (Hz, shifted) is passed directly into `NSFGenerator`. It upsamples the F0 sequence to audio sample rate, synthesizes a harmonic sinusoid train via `SourceModuleHnNSF`, and the residual blocks shape the timbre. The vocoder outputs audio at exactly the frequencies it was given.

**Step 4 — FAISS index retrieval (timbre, not pitch):**  
Content embeddings are replaced with a weighted average of nearest-neighbor embeddings from the training set. This pulls phonetic codes toward the target speaker's vocal style but does not affect pitch at all.

**Step 5 — Speaker embedding `g`:**  
`emb_g(sid=0)` injects a single learned speaker vector into all decoder layers. Conditions the vocoder on target timbre, not pitch.

### 1.3 Protect parameter

`protect` (0–0.5) controls how much of the raw (pre-index-retrieval) content embedding is blended back for unvoiced frames. At `protect=0.33`, voiced frames (pitchf > 0) use the retrieved embedding fully; unvoiced frames blend 33% of original back. This preserves consonant quality in fricatives/stops where RMVPE would otherwise assign a spurious pitch.

---

## 2. Beatrice 2 Pitch Pipeline

### 2.1 Training — pitch-invariant content features

Beatrice 2 explicitly disentangles pitch from content during training through **pitch augmentation** (`converter.py → forward()` training branch):

1. Each batch item is resampled to a random pitch-shifted version (integer-ratio resampling, which shifts pitch without changing speed — a phase-vocoder-free trick using torchaudio's `Resample`).
2. `PitchEstimator` runs on the pitch-shifted audio to get pitch logits.
3. The pitch logits are **circularly shifted** in the pitch-bin dimension by the negative shift amount, realigning them back to the original pitch scale.
4. The `PhoneExtractor` therefore sees the same content at many different pitches across training, while the pitch conditioning always reflects the actual pitch of the audio given. The content codes learn to be pitch-independent.

The VQ codebooks in `PhoneExtractor` are conditioned on `target_speaker_id` so the quantized codes pull toward examples from that speaker's training data — capturing timbre and articulation style.

### 2.2 Pitch estimation — PitchEstimator architecture

`PitchEstimator` (`models/pitch_estimator.py`) is a custom neural pitch estimator based on a ConvNeXt stack. It does not use RMVPE, DIO, or any traditional algorithm.

**Input features extracted per 10ms frame (hop=160 @ 16 kHz):**
- Instantaneous frequency features: log power spectrum + phase time-differences at bins 0–63 (up to ~1828 Hz), giving `64×3 = 192` channels
- Autocorrelation difference function (YSDF/AMDF-style): periods 1–256 samples (62.5–16000 Hz range), giving 256 channels

These are projected and summed → `ConvNeXtStack` (9 blocks, 192 channels, 33-sample causal-lookahead kernel, 1-frame delay) → linear head → 448-bin softmax distribution.

**Pitch bin encoding:**  
- Bin 0 = unvoiced  
- Bins 1–447: log-spaced at 96 bins/octave, anchored at 55 Hz (A1):  
  `f0_hz = 55.0 × 2^(bin / 96)`  
  This covers ~55 Hz (A1) to ~2 kHz at 96 bins/octave = 8 bins/semitone resolution.

### 2.3 Inference — pitch conditioning and shift

Code path: `converter.py → forward()` (inference mode)

**Step 1 — PitchEstimator on input audio:**  
Runs on the full input window at 16 kHz. Returns `(pitch_logits [B, 448, T], energy [B, 1, T])`.

**Step 2 — sample_pitch():**  
Converts the softmax distribution to a hard quantized bin using a 4-bin sliding-window argmax (equivalent to a soft-to-hard quantization that is robust to neighboring-bin ambiguity). Returns `quantized_pitch` (Long tensor, 0 = unvoiced).

**Step 3 — User pitch shift as bin offset:**  
```python
quantized_pitch = clamp(quantized_pitch + round(semitones × 96/12), 1, 447)
# 96 bins/octave ÷ 12 semitones/octave = 8 bins/semitone
# Unvoiced frames (bin=0) are left as 0 — no shift applied to silence
```

**Step 4 — Pitch features fed to decoder:**  
`embed_quantized_pitch(quantized_pitch)` produces a sinusoidal positional encoding (sin/cos at harmonically spaced frequencies) that represents pitch as a rich embedding. Additional features — unvoiced probability, half-pitch probability (octave-error guard), double-pitch probability — are extracted from the softmax distribution and concatenated as `pitch_features [B, 4, T]` (energy + 3 pitch features).

**Step 5 — Decoder receives phone + pitch + energy:**  
Phone codes, pitch embedding, and pitch features are concatenated and fed to the formant-shifted decoder. The vocoder generates audio conditioned on all three — content, pitch, and energy — producing the converted voice at the target pitch.

---

## 3. Comparison: How Pitch Differs Between RVC and Beatrice 2

| Aspect | RVC | Beatrice 2 |
|---|---|---|
| F0 extractor | RMVPE (pretrained deep NN, loaded from `assets/rmvpe/`) | PitchEstimator (trained jointly with the voice conversion model) |
| Pitch representation | Mel-bin integer (1–255) + raw Hz float | Quantized log-bin (1–447, 96 bins/octave) + softmax distribution features |
| Shift mechanism | Multiply Hz by `2^(semitones/12)` before mel-binning | Add `round(semitones × 8)` to quantized bin index |
| Unvoiced handling | Bin 0 passed through, Hz=0 → NSF silences excitation | Bin 0 left at 0 — shift not applied to unvoiced frames |
| Pitch in vocoder | F0 Hz upsampled → harmonic sinusoid excitation (NSF) | Sinusoidal positional embedding fed to attention-based decoder |
| Pitch-content disentanglement | Implicit — HuBERT learns some invariance but pitch leaks into content codes | Explicit — pitch augmentation during training forces PhoneExtractor to be pitch-independent |
| Model learns target speaker pitch? | No | No |

Neither pipeline learns the target speaker's pitch distribution. Both pipelines pass the source speaker's F0 (with user shift applied) directly to the vocoder as hard conditioning. The user's semitone slider is the **only** pitch mapping mechanism in both systems.

---

## 4. Auto Pitch-Shift (Offline Inference)

Auto pitch-shift computes the semitone offset needed to align the source speaker's central pitch to the target speaker's central pitch. Available for both RVC and Beatrice 2 profiles.

### 4.1 Profile mean F0 (precomputed, backend)

**Endpoint:** `POST /api/offline/speaker_f0/{profile_id}/compute`  
**Algorithm:** pyworld DIO on all training audio files

DIO (Distributed Inline-filter Operation) is a multi-band pitch estimator. It runs the input through a bank of bandpass filters centered at candidate F0 frequencies, computes a combined instantaneous frequency estimate, and applies a low-pass postfilter to reduce octave errors. It is more robust than autocorrelation-based methods because the multi-band approach provides redundant evidence and partially suppresses the half/double-pitch errors common in autocorrelation.

For each training audio file:
```python
mono64 = audio.astype(np.float64)          # pyworld requires float64
f0, _ = pyworld.dio(mono64, sr=16000)       # F0 per 5ms frame, 0 = unvoiced
voiced = f0[f0 > 0]
```

Across all voiced frames from all files, the **geometric mean** is computed:
```python
mean_f0 = exp( sum(log(f0_i)) / n_voiced_frames )
```

Geometric mean is used because pitch is perceptually logarithmic. A voice at 100 Hz and 400 Hz should average to 200 Hz (one octave each way), not 250 Hz. Arithmetic mean would bias toward higher-pitched audio.

Result saved as `{profile_dir}/speaker_f0.json` (RVC) or `{profile_dir}/beatrice2_out/speaker_f0.json` (Beatrice 2). Computed once after training; displayed in the UI as the profile's reference pitch.

### 4.2 Input file mean F0 (per-inference, backend)

**Endpoint:** `POST /api/offline/input_f0` (accepts uploaded audio file)  
**Algorithm:** pyworld DIO, identical to profile computation

The input file is decoded to 16 kHz mono float64 and DIO is run identically:
```python
f0, _ = pyworld.dio(mono64, sr=16000, f0_floor=50.0, f0_ceil=800.0)
voiced = f0[f0 > 0]
mean_f0 = exp(mean(log(voiced)))
```

Using the same algorithm for both sides of the equation means systematic biases cancel out — if DIO consistently estimates a certain voice slightly high, it will do so for both the training data and the input file, and the resulting ratio will still be correct.

Returns `{"mean_f0": float, "method": "dio", "n_frames": int}`.

> **Previous approach (replaced):** Input F0 was computed in the browser via a simple frame-by-frame autocorrelation over the decoded PCM. This was fast and required no server round-trip but was prone to octave errors (the autocorrelation can lock onto a harmonic instead of the fundamental, halving the detected F0). Using DIO on the backend for both sides gives consistent, more reliable estimates.

### 4.3 Semitone shift derivation

```
semitones = 12 × log₂(profile_F0 / input_F0)
```

This is the exact signed interval in equal-tempered semitones between the two geometric mean pitches.

- Positive result: profile is higher than input → shift up (e.g., male→female)
- Negative result: profile is lower than input → shift down (e.g., female→male)

For display and use, the value is rounded to the nearest 0.5 semitone (Beatrice 2) or nearest integer semitone (RVC, whose pitch parameter is integer-only).

### 4.4 Pipeline routing

The computed shift is written into the correct parameter depending on the profile's pipeline:

| Pipeline | Parameter written | Form field | How it's used |
|---|---|---|---|
| RVC | `pitch` (int semitones) | `pitch` | `f0_up_key` in `rtrvc.py → _get_f0()` |
| Beatrice 2 | `pitchShift` (float semitones) | `pitch_shift_semitones` | Bin offset in `converter.py → forward()` |

### 4.5 Limitations

Auto pitch-shift maps the **central tendency** of the source pitch range to the target pitch range. It applies a **global, constant offset** — the pitch contour shape (intonation, prosody) is preserved exactly as in the source. It does not:
- Normalize pitch variance (a speaker with a wide pitch range will still have wide variation after shifting)
- Handle speakers whose pitch ranges overlap poorly with the target (e.g., very deep bass → very high soprano may require manual fine-tuning)
- Adapt dynamically during inference — the offset is computed once per file before conversion starts

For best results, the training data should cover the target speaker's full pitch range naturally. Profiles trained on very short audio clips or clips with limited pitch variation will produce less reliable `speaker_f0.json` estimates.

---

## 5. Mean-Std F0 Normalization

### 5.1 Motivation

The simple semitone-shift approach (Section 4.3) maps the source speaker's mean F0 to the target's mean, but leaves the **pitch range width unchanged**. If the source has a narrow-range monotone delivery (σ_s small) and the target is an expressive speaker with wide intonation swings (σ_t large), the output will sound flat relative to the target's natural style. Conversely, a wide-range source mapped to a narrow-range target will sound exaggerated.

Mean-std normalization fixes both problems by rescaling the F0 variance as well.

### 5.2 Formula

All operations are in **log-Hz space**, which is linear in semitones (perceptually uniform):

```
log_f0_out = log(μ_t) + (σ_t / σ_s) × (log_f0_in − log(μ_s))
```

Equivalently in Hz:

```
f0_out = μ_t × (f0_in / μ_s)^(σ_t / σ_s)
```

Where:
- `μ_s`, `σ_s` = geometric mean and log-space std of **source speaker's** voiced F0 (from input file)
- `μ_t`, `σ_t` = geometric mean and log-space std of **target speaker's** voiced F0 (from training audio)
- σ (log-space std) is the natural log standard deviation of voiced F0 frames — equivalent to the log of the geometric standard deviation

The formula has two effects:
1. **Mean shift**: `log(μ_t) - (σ_t/σ_s) × log(μ_s)` — a constant offset aligning the central tendency
2. **Variance rescaling**: the `(σ_t / σ_s)` factor stretches or compresses the pitch contour around the mean

**Unvoiced frames (F0 = 0) are not modified** — the transformation is applied only where F0 is detected.

### 5.3 What σ (log-space std) measures

Log-space std is computed as:

```python
log_f0 = np.log(voiced_f0_hz)   # voiced frames only
σ = log_f0.std()
```

A value of `σ = 0.20` means roughly a ±20% multiplicative spread around the mean — approximately ±3.5 semitones at 1-sigma. Typical values:
- Monotone/flat speaker: σ ≈ 0.08–0.12
- Normal conversational speaker: σ ≈ 0.15–0.25
- Highly expressive/singing: σ ≈ 0.30+

When `σ_t / σ_s > 1`, the output pitch contour is stretched (more expressive). When `< 1`, it is compressed (flatter).

### 5.4 Implementation

**Backend — `compute_speaker_f0` (`routers/offline.py`)**

```python
log_f0 = np.log(voiced_f0)
mean_f0 = math.exp(log_f0.mean())
std_f0  = log_f0.std()   # log-space std
```

Both `mean_f0` and `std_f0` are now saved in `speaker_f0.json`. Existing profiles with only `mean_f0` continue to work — normalization degrades gracefully to mean-only shift when `std_f0` is absent.

**Backend — `POST /api/offline/input_f0`**

Returns `{"mean_f0": float, "std_f0": float, "method": "dio", "n_frames": int}`.

**Backend — `_run_offline_inference` (RVC, `routers/offline.py`)**

When `f0_norm_params` is provided, the `_patched_get_f0` closure applies the normalization frame-by-frame after RMVPE extraction:

```python
log_f = np.log(voiced_frames_hz)
log_f_norm = log(μ_t) + (σ_t / σ_s) × (log_f − log(μ_s))
f0_normalized = np.exp(log_f_norm)
```

The normalized Hz values are then re-quantized to RVC's mel-bin representation (1–255, log2-scale anchored at 32.7 Hz) for the content encoder and passed as raw Hz to the NSF vocoder. RVC is initialized with `key=0` so no separate semitone shift is applied on top.

**Backend — Beatrice 2 (`BeatriceInferenceEngine`, `inference.py`)**

B2's `sample_pitch()` returns `quantized_pitch` — a Long tensor of log-spaced bin indices (0=unvoiced, 1–447=voiced, 96 bins/octave anchored at 55 Hz). Because bins are linear in log-Hz, the normalization formula is a pure linear rescaling in bin space:

```
bin_out = μ_t_bin + (σ_t/σ_s) × (bin_in − μ_s_bin)
where μ_bin = log2(μ_hz / 55) × pitch_bins_per_octave
```

This is implemented by monkey-patching `pitch_estimator.sample_pitch` inside `_apply_f0_norm_patch()`, which wraps the original method and transforms the returned `quantized_pitch` tensor before it reaches `converter.py`'s bin-offset logic. The patch is installed just before inference and removed in a `finally` block. `pitch_shift_semitone=0` is passed to the model so the built-in bin-offset is a no-op.

The normalization applies to **both** `convert()` (realtime chunks) and `convert_offline()` (full-file batch chunking), and preserves thread safety — the patch is installed/removed around each call, not held globally.

**Frontend — `offline/page.tsx`**

When auto pitch is enabled and all four stats are available:
- Form fields `f0_src_mean`, `f0_src_std`, `f0_tgt_mean`, `f0_tgt_std` are appended to the FormData
- The pitch sliders show `norm  μ +Nst  σ×X.XX` instead of `auto: +N st`
- The profile stats line shows `Hz  σ=X.XXX` when std is available

When auto pitch is on but std is missing from an old profile (no `std_f0` in json), the backend falls back to mean-only shift (`ratio = 1.0` guard in `_patched_get_f0`).

### 5.5 Comparison: mean-only shift vs mean-std normalization

| Scenario | Mean-only shift | Mean-std normalization |
|---|---|---|
| Same-type voice (e.g. male→male, similar range) | Good | Marginally better |
| Cross-gender (male→female) with similar expressiveness | Good | Similar |
| Monotone source → expressive target | Flat, unnatural output | Pitch contour stretched — more natural |
| Expressive source → monotone target | Overly dynamic output | Pitch contour compressed — more natural |
| Very short source file (few voiced frames) | Reliable | σ estimate unreliable → treat σ ratio with caution |

**When to re-compute profile F0**: after adding significantly more training audio. The original `speaker_f0.json` computed from limited data may have a skewed σ. The `/api/offline/speaker_f0/{id}/compute` endpoint overwrites the file cleanly.

### 5.6 Limitations

- **σ ratio instability with short files**: if the source file has fewer than ~500 voiced frames (≈5 seconds of speech), the log-std estimate is noisy. DIO's `n_frames` count in the response can be used to detect this.
- **Prosody is rescaled, not reconstructed**: normalization stretches/compresses the existing F0 contour — it does not resynthsize natural intonation in the target speaker's style. The output contour shape is still the source speaker's.
- **Extreme ratios clip badly**: if σ_t/σ_s > 2 or < 0.5, consider clamping the ratio before applying to avoid unnaturally extreme intonation. The current implementation applies the ratio without clamping.
