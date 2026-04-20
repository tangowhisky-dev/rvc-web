# Beatrice 2: Realtime Inference Pipeline Analysis

**Date**: 2026-04-17  
**Status**: Analysis complete — implementation pending  
**File**: `docs/beatrice2-realtime-pipeline.md`

---

## Current State

The Beatrice 2 realtime path in `_realtime_worker.py` was built by copying the
RVC realtime infrastructure and swapping only the inference call. This means
several RVC-specific design decisions were inherited without validation for
Beatrice 2's architecture. Two of them are wrong: context window size and SOLA
cross-correlation.

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

### Beatrice 2 Realtime (current)

```
audio callback (200ms block @ 48kHz)
  → RNNoise denoising (optional)
  → resample 48k→16k
  → roll input_buffer_16k  (extra_16k=32000 + block_16k=3200 = 35200 samples = 2.2s)
  → engine.convert(input_buffer_16k)   ← single forward pass over full 2.2s
      PhoneExtractor (causal conv + local attention)
      PitchEstimator (local)
      ConverterNetwork vocoder
      output at 24kHz
  → resample 24k→48k
  → trim tail (block_48k + sola_buf samples)
  → RMS matching (EWMA)
  → SOLA crossfade (sin² fade blend)
  → emit 200ms block
```

---

## Problem 1: Context Window is 10× Too Large

### Why RVC needs 2.2s

RVC requires a 2-second history window for two reasons:

1. **HuBERT global self-attention** — HuBERT's features for frame N depend on
   all other frames in the window. A short window produces unstable features at
   the edges. 2s is the minimum for reliable feature quality.

2. **RMVPE F0 stability** — pitch tracking needs enough signal to observe
   multiple pitch periods. At 80Hz (low male voice), one period = 12.5ms. RMVPE
   needs ~500ms for robust tracking. The 2s window gives headroom.

RVC then uses `skip_head_frames` to discard the history-derived output and emit
only the tail 200ms.

### Why Beatrice 2 does NOT need 2.2s

Beatrice 2's `PhoneExtractor` is a causal convolutional network with local
attention — not HuBERT. Its receptive field is a few hundred milliseconds at
most. Its `PitchEstimator` is also local. Neither architecture requires global
context over 2 seconds.

The training `wav_length=96000` (6s) is the maximum window the model was
trained on, not a minimum context requirement.

### Consequences of the oversized window

1. **Wasted compute**: the model processes 2.2s of audio to produce 200ms of
   output — 11× more computation than necessary per callback. On a GPU this
   may still be fast, but it consumes memory bandwidth and increases inference
   time proportionally.

2. **Latency inflation**: the effective output latency is
   `2.2s + inference_time`, not `200ms + inference_time`. The first 2.2s after
   session start are filled with zeros (the rolling buffer initialises to zero)
   so the conversion sounds wrong for the first ~2 seconds of every session.

3. **Stale audio in the window**: at any callback, the model sees 2.0s of audio
   that was already processed in previous callbacks, plus 200ms of new audio.
   This is redundant — the model re-processes the same audio 10 times before
   it falls out of the window.

### Recommended fix

Reduce `_EXTRA_TIME` from `2.0s` to `~0.3s` for the Beatrice 2 path. This
gives the phone extractor and pitch estimator enough history (~300ms) to produce
stable outputs at the block boundary without the 10× overhead.

The Beatrice 2 path should have its own `_EXTRA_TIME_B2 = 0.3` constant rather
than sharing `_EXTRA_TIME = 2.0` with RVC.

Expected impact:
- Input buffer: 35200 samples (2.2s) → ~8000 samples (0.5s)
- Per-callback inference: ~5–8× faster
- Session startup silence: 2.0s → 0.3s
- Perceived latency: reduced by ~1.7s

The exact value (0.3s vs 0.5s) should be validated by listening tests — if
conversion quality degrades at 0.3s, increase to 0.5s.

---

## Problem 2: SOLA Cross-Correlation Is Unnecessary

### Why RVC needs SOLA cross-correlation

RVC's `rtrvc.infer()` does not guarantee that its output aligns perfectly with
the input block boundary. The F0 estimation and vocoder hop sizes introduce an
alignment drift of up to ±10ms per block. Without correction, this drift
accumulates and produces audible clicks and pitch artifacts.

SOLA solves this by:
1. Requesting slightly more output than needed (`block + sola_buf + sola_search`)
2. Cross-correlating the tail of the previous block (`sola_buffer`) against
   the head of the new output to find the best alignment offset within a 10ms
   search window
3. Applying a `sin²` fade blend at the found offset

### Why Beatrice 2 does NOT need cross-correlation

Beatrice 2's output ratio is exact: `out_sr / in_sr = 24000 / 16000 = 1.5`.
Every input sample maps to exactly 1.5 output samples with no rounding. There
is no alignment drift between consecutive blocks — the output is always
perfectly aligned with the input.

The cross-correlation step in the current Beatrice 2 SOLA implementation finds
nothing useful: the best offset is always 0 (or noise). It wastes compute and
adds ~10ms of unnecessary search overhead per callback.

### Does the fade blend still help?

Marginally. Even with perfect alignment, the vocoder produces a waveform that
may start from a slightly different phase at the head of a new block vs the
tail of the previous one. A crossfade blend (`sin²` fade) over the overlap
region smooths this.

However, if the context window is reduced to 0.3s (Problem 1 fix), consecutive
windows overlap by 0.1s (300ms context - 200ms block = 100ms overlap). The
model's output for the overlap region is nearly identical between consecutive
windows. Phase discontinuities become negligible without any crossfade.

### Recommended fix

Replace the full SOLA implementation (cross-correlation + blend) with a simple
`sin²` crossfade over the last `sola_buf` samples, with no cross-correlation
search. Or remove the crossfade entirely and validate by listening.

The Beatrice 2 SOLA path should be a separate, simpler implementation rather
than sharing the RVC SOLA code which includes cross-correlation machinery
Beatrice 2 doesn't need.

---

## Implementation Plan

### Step 1: Add `_EXTRA_TIME_B2` constant

```python
_EXTRA_TIME_B2 = 0.3   # Beatrice 2: causal conv, no global attention needed
```

Pass this to `_compute_block_params()` when building `block_params` for the
Beatrice 2 path, instead of `_EXTRA_TIME`.

### Step 2: Update `_run_beatrice2_realtime` buffer sizes

```python
extra_16k = (int(0.3 * _SR_16K) // _WINDOW) * _WINDOW  # ~4800 samples
input_buffer_16k = np.zeros(extra_16k + block_16k, dtype=np.float32)
```

The tail-trim logic (`out_full_48k[-needed_48k:]`) remains unchanged — just
the buffer being passed to `engine.convert()` gets shorter.

### Step 3: Remove cross-correlation from Beatrice 2 SOLA

Replace the current SOLA block in `_process_block` (B2 path) with:

```python
if sola_buf > 0:
    # Simple crossfade — no cross-correlation needed (exact output ratio)
    head = torch.from_numpy(out_np[:sola_buf])
    merged = sola_buffer * out_fade_window + head * in_fade_window
    sola_buffer = torch.from_numpy(out_np[return_length:].copy())
    out_full = np.concatenate([merged.numpy(), out_np[sola_buf:]])
else:
    out_full = out_np
```

This is already what the current code does — the RVC path is the one with
cross-correlation. Verify that the current B2 SOLA block does not call into
the phase vocoder or cross-correlation helpers.

### Step 4: Validate by listening

After reducing `_EXTRA_TIME_B2`:
- Listen for degraded conversion quality at block boundaries
- Listen for increased artifacts on sustained vowels across block boundaries
- If artifacts appear at 0.3s, increase to 0.5s and re-test
- Measure per-callback inference time before and after

---

## Related Documents

- `docs/beatrice2-augmentation-precompute.md` — DataLoader bottleneck and
  pre-computed augmentation analysis
- `backend/beatrice2/inference.py` — `convert()` (realtime) and
  `convert_offline()` (offline chunked)
- `backend/app/_realtime_worker.py` — `_run_beatrice2_realtime()` and
  `_run_rvc_realtime()` (RVC path, for comparison)
