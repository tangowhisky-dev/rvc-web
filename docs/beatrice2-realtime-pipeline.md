# Beatrice 2: Realtime Inference Pipeline Analysis

**Date**: 2026-04-17  
**Updated**: 2026-04-17 (implementation complete)  
**Status**: ✅ Implemented  
**File**: `docs/beatrice2-realtime-pipeline.md`

---

## Current State (post-implementation)

All four issues identified in the original analysis have been addressed.
The Beatrice 2 realtime path now uses a 0.5s context window instead of the
inherited 2.2s RVC window, fixes the `_EXTRA_48K` inconsistency, and
pre-builds resamplers outside the hot callback path.

---

## Pipeline Comparison

### RVC Realtime

```
audio callback (200ms block @ native SR)
  → RNNoise denoising (optional, 48kHz)
  → resample native→16k (inference) + native→48k (visualiser/save)
  → roll input_buffer_16k  (extra_16k=32000 + f0_extractor_frame ≈ 2.2s total)
  → rvc.infer(input_buffer_16k, skip_head_frames, return_length_frames)
      HuBERT feature extraction (global attention over full window)
      RMVPE F0 estimation (needs long window for pitch period stability)
      FAISS index search
      decoder + vocoder
      outputs (block_tgt + sola_buf + sola_search) samples
  → RMS matching (EWMA)
  → SOLA crossfade (phase vocoder blend + cross-correlation alignment)
  → emit 200ms block to output device
```

### Beatrice 2 Realtime (after fix)

```
audio callback (200ms block @ 48kHz)
  → RNNoise denoising (optional)
  → resample 48k→16k  (pre-built _resamp_48_16, reused every callback)
  → roll input_buffer_16k  (extra_16k=7840 + block_16k=3200 = ~11040 samples = 0.69s)
  → engine.convert(input_buffer_16k)
      PhoneExtractor (causal conv + local attention)
      PitchEstimator (local)
      ConverterNetwork vocoder
      output at 24kHz
  → resample 24k→48k  (pre-built _resamp_b2_48, reused every callback)
  → trim tail (block_48k + sola_buf samples)
  → RMS matching (EWMA)
  → simple sin² crossfade (no cross-correlation)
  → emit 200ms block
```

---

## Issues Fixed

### Issue 1: Context window 2.2s → 0.5s ✅

**Root cause**: `_EXTRA_TIME = 2.0` was shared between RVC and Beatrice 2.
RVC needs 2s for HuBERT global attention and RMVPE F0 stability. Beatrice 2
uses causal convolution and local attention — neither requires global context.

**Fix**: Added `_EXTRA_TIME_B2 = 0.5` constant. Passed as `extra_time` to
`_compute_block_params()` for the Beatrice 2 path only. RVC path unchanged.

**Result**:
- Input buffer: 35200 samples (2.2s) → ~11040 samples (0.69s)
- Per-callback compute: ~4× reduction
- Startup silence: 2.0s → 0.5s
- GPU memory per callback: proportionally reduced

#### Why 0.5s was chosen

0.5s gives the pitch estimator 40+ complete pitch periods for even a low male
voice (80Hz = 12.5ms/period). The phone extractor's convolutional receptive
field is well within 300ms. 0.5s provides a safe margin above both requirements.

#### Path to 0.3s

**0.3s is likely sufficient and worth testing.** At 0.3s:
- `extra_16k = 4640 samples`, total buffer = `4640 + 3200 = 7840 samples (0.49s)`
- Pitch estimator still sees 24+ pitch periods at 80Hz
- Per-callback compute: ~6× reduction vs original 2.2s

To try 0.3s: change `_EXTRA_TIME_B2 = 0.5` → `_EXTRA_TIME_B2 = 0.3` in
`_realtime_worker.py` and run a listening test:
- Sustained vowels across block boundaries — listen for pitch instability
- Consonants (plosives, fricatives) — listen for smearing or artifacts
- Low-pitched voice (male, <100Hz) — most sensitive to short windows

If no degradation is heard at 0.3s, adopt it permanently. If there are
artifacts on specific voices, 0.5s is the safe default.

---

### Issue 2: SOLA cross-correlation ✅ (was already correct)

Verified that the B2 SOLA block already uses simple `sin²` blend only — no
cross-correlation, no phase vocoder. The RVC path uses `_phase_vocoder()`;
the B2 path does not. No code change was needed.

This is correct because Beatrice 2's output ratio is exact
(`out_sr / in_sr = 24000 / 16000 = 1.5`). There is no alignment drift to
correct. Cross-correlation would find offset=0 on every block (or noise).

The simple crossfade remains useful — it smooths the waveform phase at block
boundaries without the cost of cross-correlation search.

---

### Issue 3: `_EXTRA_48K` used wrong constant ✅

`input_buffer_48k` (used for waveform visualisation) was sized using RVC's
`_EXTRA_TIME = 2.0` even after reducing the inference window:

```python
# Before (wrong):
_EXTRA_48K = int(_EXTRA_TIME * _SR_48K)   # 2.0s × 48000 = 96000 samples

# After (correct):
_EXTRA_48K = int(_EXTRA_TIME_B2 * _SR_48K)  # 0.5s × 48000 = 24000 samples
```

No quality impact — this buffer feeds the visualiser, not inference. But the
inconsistency was misleading and the oversized allocation was wasteful.

---

### Issue 4: Resamplers constructed per-callback ✅

Two `torchaudio.transforms.Resample` objects were constructed inside
`_process_block` on every 200ms callback:

```python
# Before (per-callback construction):
t_16k = _R(orig_freq=48000, new_freq=16000)(t_in).squeeze(0)
t_48k = _R(orig_freq=tgt_sr, new_freq=48000)(t_out).squeeze(0)
```

These are now pre-built once outside `_process_block` and reused:

```python
# After (pre-built, hoisted above _process_block):
_resamp_48_16 = _R(orig_freq=48000, new_freq=16000)
_resamp_b2_48 = _R(orig_freq=tgt_sr, new_freq=48000)
```

Also removed a redundant `import torch as _torch` inside the SOLA state
block — `_torch` is now imported once alongside the resampler construction.

---

## Constants Reference

```python
_EXTRA_TIME    = 2.0   # RVC context window — HuBERT + RMVPE require this
_EXTRA_TIME_B2 = 0.5   # Beatrice 2 context window — safe default
                       # Try 0.3 if inference is still too slow (see above)
```

---

## Related Documents

- `docs/beatrice2-augmentation-precompute.md` — DataLoader bottleneck and
  pre-computed augmentation analysis
- `backend/beatrice2/inference.py` — `convert()` (realtime) and
  `convert_offline()` (offline chunked)
- `backend/app/_realtime_worker.py` — `_run_beatrice2_realtime()`,
  `_EXTRA_TIME_B2`, `_compute_block_params()`
