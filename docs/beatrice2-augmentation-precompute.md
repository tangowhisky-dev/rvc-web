# Beatrice 2: Pre-computed Formant Shift Augmentation

## Problem

At `batch_size=32`, training stalls every ~8 iterations because DataLoader
workers are bottlenecked on `random_formant_shift()` in `augment_audio()`.
This function calls `pyworld.dio` + `pyworld.stonemask` + `pyworld.cheaptrick`
(CPU-bound, serial per sample) plus STFT/iSTFT — it is the dominant cost in
`__getitem__`. Workers cannot refill the prefetch queue fast enough to keep
the GPU busy.

The proposed fix is to pre-compute formant-shifted copies during preprocessing
and disable the training-time pyworld augmentation. This document records what
needs to be validated before that change can be made safely.

---

## Current Augmentation Pipeline (training time)

Each `__getitem__` call applies two formant-related transforms:

### 1. Resampler-based pitch/tempo shift (always applied, fast)

```python
formant_shift_candidates = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
formant_shift = formant_shift_candidates[torch.randint(0, 9, ())]
resampler_fraction = Fraction(sample_rate / out_sample_rate * 2**(shift/12))
clean_wav = get_resampler(num, denom)(clean_wav)
```

This shifts pitch AND tempo together (it just resamples). It's fast — a
cached `torchaudio.transforms.Resample` applied to the waveform. It covers
±2 semitones in 0.5-semitone steps (9 discrete values).

### 2. Pyworld spectral envelope shift (50% probability, slow)

```python
if torch.rand(()) < formant_shift_probability:  # 0.5
    clean = random_formant_shift(clean, sample_rate,
                                 semitone_min=-3.0, semitone_max=3.0)
```

This shifts the **spectral envelope only** (formant frequencies) without
changing pitch or tempo. It uses pyworld to extract the spectral envelope,
interpolates it by the shift ratio, and reconstructs via STFT. This is
the expensive call — roughly 50-200ms per sample on CPU.

---

## Proposed Fix

Pre-compute N formant-shifted copies of each audio chunk during `preprocess.py`.
At training time, disable the pyworld augmentation
(`augmentation_formant_shift_probability=0`). Workers load pre-shifted files
and apply only the fast augmentations (noise, reverb, LPF, random filter).

---

## What Needs Validation

### 1. Discrete vs continuous shift distribution

**Current**: the training-time shift is drawn uniformly from `[-3, +3]`
semitones (continuous). Each sample sees a different random shift on every
epoch, and across enough iterations the distribution approximates uniform.

**Proposed**: 5 discrete copies at `[-2, -1, 0, +1, +2]` semitones.
The distribution is now discrete and the range is narrower (±2 vs ±3).

**What to check**:
- Does the model benefit from shifts beyond ±2 semitones? The original
  range is ±3. Shifts near ±3 may be rare at training time (continuous
  uniform) but they push the model toward more extreme generalisation.
- Does discrete sampling (the same 5 fixed shifts repeated every epoch)
  cause the model to overfit to specific shift values? With a large
  dataset this is unlikely, but with small single-speaker datasets
  (e.g. <30 minutes) it may matter.
- Statistically: continuous uniform on [-3,+3] has mean=0, std≈1.73.
  Five-point discrete on [-2,-1,0,+1,+2] has mean=0, std≈1.41. The
  claim that "5 copies gives equivalent diversity" is **not validated**
  — it is an approximation.

### 2. Interaction with the resampler shift

Both augmentations are applied together at training time. The resampler
shift moves pitch+tempo by ±2 semitones (9 steps). The pyworld shift
moves spectral envelope by ±3 semitones independently. They are
**independent random variables** applied in sequence.

Pre-computing the pyworld shift means it is no longer independent of
the resampler shift within a training iteration — instead the file has
a fixed pyworld shift baked in, and the resampler shift is applied on
top at training time. The combined distribution changes:

- Current: (resampler_shift, formant_shift) are independent; the joint
  distribution covers an 18-semitone range for pitch-like variation.
- Proposed: formant_shift is fixed per file, resampler_shift is still
  random. The joint distribution is a product of discrete × discrete
  rather than discrete × continuous.

**What to check**: whether the independence of the two shifts matters
for training quality, or whether the model just needs exposure to a
variety of (pitch_shift, formant_shift) combinations regardless of
how they are paired.

### 3. Effective dataset size vs augmentation diversity

With N=5 copies the dataset is 5× larger. The DataLoader sees each
copy as an independent file and shuffles them. This means:

- Each epoch, each copy appears exactly once → each file's N shifts
  are seen once per epoch.
- Current pipeline: each file appears once per epoch with one random
  shift. Over K epochs, a file is seen K times with K different shifts.
- Pre-computed: each file appears N times per epoch with N fixed shifts,
  repeated every epoch.

For a fixed training budget (e.g. 10,000 steps), the pre-computed
pipeline sees more unique (file, shift) combinations only if K < N
per file. For a single-speaker dataset with 200 files and batch_size=32,
10,000 steps ≈ 1,600 epochs — far more than N=5. The diversity claim
is **not equivalent** for small datasets with many epochs.

### 4. Quality impact on UTMOS

The claim that pre-computed augmentation gives "equivalent training
quality" has not been tested. Before adopting this approach in
production:

1. Train two models with identical hyperparameters on the same dataset:
   one with live pyworld augmentation (batch_size=8 to avoid the stall),
   one with pre-computed 5-copy augmentation (batch_size=32).
2. Compare final UTMOS scores and loss curves.
3. If UTMOS differs by >0.05 (within UTMOS noise floor), the approaches
   are not equivalent and further tuning of N or shift range is needed.

---

## Alternative Approaches Not Requiring Validation

These fixes address the bottleneck without changing augmentation semantics:

1. **Reduce `prefetch_factor` from 4 to 2** (one-line change) — smooths
   queue drain pattern. Does not fix root cause but reduces burst stalls.

2. **Increase `num_workers`** — more parallel pyworld calls. Effective
   if the machine has CPU cores available beyond `h.num_workers=16`.

3. **Move formant shift to a background thread within the worker** using
   `torch.utils.data.get_worker_info()` and async prefetch. Complex.

4. **Replace pyworld with a faster spectral envelope estimator** — e.g.
   LPC-based or WORLD's D4C directly. Would require validating that the
   augmentation effect is equivalent.

---

## Recommendation

Do not adopt pre-computed augmentation without running the UTMOS comparison
in §4 above. The performance benefit is real but the quality equivalence
claim is unvalidated. Start with `prefetch_factor=2` as a safe immediate
fix, and schedule a proper ablation if batch_size=32 remains a priority.
