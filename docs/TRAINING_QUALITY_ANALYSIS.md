# Training Quality Analysis

## Why Our Cloning Quality is Poor After 200–300 Epochs

Investigation of our training pipeline vs. reference projects (Retrieval-based-Voice-Conversion-WebUI, ultimate-rvc) and community best practices.

---

## Revalidated Claims

### ✅ CONFIRMED — Overtraining detector uses last-batch loss, not epoch average

**This remains the single most likely cause of poor quality and is the highest-priority fix.**

Our overtraining detector in `backend/app/training.py` (lines 1102–1115) tracks `loss_gen` — the **last batch's generator loss from the final log interval of each epoch**. With `log_interval=10` and ~200 steps/epoch (10 min audio, bs=8), this is step ~200 out of 208, not an average.

ultimate-rvc tracks `avg_global_gen_loss / len(dataset)` — the mean loss accumulated across all steps in the epoch.

**Why the difference matters with large files:**
With 10–15 min of audio and bs=8 there are ~200–310 steps/epoch. The last-batch `loss_gen` has significant variance — it can spike on a hard batch even when the epoch trend is improving. With `overtrain_threshold=10–20` (the guidance shown in the UI), the detector fires after 10–20 consecutive bad final batches — easily triggered by normal noise, not actual overtraining.

**Consequence:** A user training 200+ epochs with overtrain detection enabled at threshold=15 can stop at epoch 20–35 on a perfectly-converging run.

**Fix required:** Accumulate `loss_gen` across all log-interval lines within an epoch, divide by step count, track the epoch mean — not the final batch.

---

### ✅ CONFIRMED — Default epochs = 50 is too low

Even with 10–15 min of audio and manual runs of 200+, the UI default of 50 is a trap for new users. ultimate-rvc defaults to 500.

**Fix required:** Change default to 300. Add hint text: "200–300 for 10+ min of audio; 500+ for best quality."

---

### ❌ RETRACTED — Noise reduction claim

Our app **does** offer noise reduction — it is available as an opt-in per-file action in the profile builder (`POST /api/profiles/{id}/audio/{file_id}/clean`). It uses `noisereduce` spectral gating with `prop_decrease=0.80`, non-stationary mode. The UI user confirmed they run this routinely.

This is not a gap. The claim in the original document is wrong.

---

### ❌ RETRACTED — 32k sample rate claim

**The 32k constraint is real but the reasoning was wrong.** The embedder (spin-v2) is not the cause — all embedders operate on 16 kHz audio regardless of training sample rate (extracted to `1_16k_wavs/` in preprocessing). The 16 kHz embedder input is constant across 32k/40k/48k training.

The actual 32k constraint is:
1. **RefineGAN pretrained weights** only exist at 32k (`assets/refinegan/f0G32k.pth`, `f0D32k.pth`). 40k/48k RefineGAN weights do not exist.
2. **HiFi-GAN** weights exist at 32k, 40k, and 48k (`assets/pretrained_v2/f0G32k/40k/48k.pth`). The code never uses them.

**Is 32k sufficient for speech-only voice cloning?**

Yes, with caveats.

32 kHz has a 16 kHz Nyquist ceiling. For speech:
- Fundamental pitch (85–255 Hz): fully captured
- Vowel formants (F1–F4, up to ~3.5 kHz): fully captured
- Consonants and sibilance (4–10 kHz): fully captured
- Breathiness, air, subtle texture (10–16 kHz): fully captured up to 16 kHz

Everything perceptually important in speech is below 12 kHz. 32k captures it. The quality difference between 32k and 40k for **speech** is marginal — the gains from 40k matter more for singing (higher harmonics, vibrato texture) than for speech cloning.

**The 32k lock is not what is hurting quality.** Leave it.

---

### ✅ CONFIRMED — 10–15 min large files: positive impact

Large files are good. Preprocessing slices them:
- 10 min → ~222 slices of 3.0s → ~1665 training segments
- 15 min → ~333 slices of 3.0s → ~2497 training segments

More unique segments = more diverse gradient signals = better generalisation. The model never sees the raw large file; it sees hundreds of 0.4s training windows drawn from those slices.

**At 200 epochs, 10 min bs=8:** 41,600 gradient updates.
**At 300 epochs, 15 min bs=8:** 93,600 gradient updates.

This is a reasonable training budget for a fine-tune from pretrained weights. The lr_decay schedule (`0.999875/epoch`) barely moves the learning rate over this range (`1e-4` → `9.75e-05` at 200 epochs), so the model has stable learning throughout.

**There is no problem here.** Large files are the right approach.

---

### ✅ CONFIRMED (but context changes the action) — c_spk and resampler

The `F.interpolate(mode="linear")` resampler for the speaker loss computation is genuinely low quality (aliasing risk). However:

- The user has already tested `c_spk=0` vs `c_spk=2.0` and observed **no quality difference either way**.
- The reference projects have no speaker loss at all.
- This means the speaker loss is not contributing, positively or negatively, in the current setup.

**Revised conclusion:** `c_spk` is neither helping nor hurting. It adds compute per step for no benefit. The noisy resampler probably means the gradient signal is too weak/corrupted to have any effect. Default to `c_spk=0` and consider removing the UI control unless a proper resampler is implemented.

---

## The Actual Remaining Issues

Given the revalidation, the problem space is narrower than originally stated:

### Issue 1 — Overtraining detector (CRITICAL, still unaddressed)

**Symptom:** Training stops early on any run with detection enabled, even at generous thresholds, because the signal is a single noisy batch.

**Fix:** Track epoch-average `loss_gen` (mean over all log-interval lines within the epoch) instead of last-batch.

### Issue 2 — Segment duration: 3.0s vs 3.7s (LOW, trivial fix)

We pass `per=3.0` to `preprocess.py`. The reference project defaults to `3.7s`. With 10–15 min of audio this gives slightly less context per slice but with ~200+ slices it's unlikely to matter much.

With `per=3.7, overlap=0.3`:
- 10 min → ~182 slices (vs 222 at 3.0s) — fewer but longer slices
- Not a meaningful quality difference at these data volumes

Low priority. Could change to 3.7 without risk.

### Issue 3 — Batch size heuristic uses RAM not VRAM (MEDIUM)

`routers/training.py` line 124: `sweet_spot_batch_size = max(4, min(24, int(available_ram_gb / 1.5)))`.

On a high-RAM/low-VRAM machine (e.g. 64 GB RAM, 6 GB VRAM) this suggests batch=24 which OOMs on the first step.

**Fix:** Use `torch.cuda.get_device_properties().total_memory` for VRAM-based heuristic when CUDA is available.

### Issue 4 — No best-epoch checkpoint (MEDIUM)

We save only `G_latest.pth` at `save_every` intervals. If the model peaks at epoch 180 and slightly degrades to epoch 200, the only saved artifact is epoch 200.

ultimate-rvc saves `{model_name}_best.pth` whenever a new epoch-average loss minimum is hit.

For a user doing 200–300 epoch runs without overtraining detection, having a best-epoch checkpoint lets them compare and pick the right artifact.

**Fix:** Save `G_best.pth` alongside `G_latest.pth` whenever a new epoch-average loss minimum is seen, and copy it to `profile_dir/checkpoints/G_best.pth`.

---

## Root Cause Summary

With 10–15 min of clean (noise-reduced) audio, 200–300 epochs, and the current setup:

| Factor | Status | Impact on quality |
|---|---|---|
| Audio data quantity | ✅ 10–15 min is good | None — not the problem |
| Noise reduction | ✅ User runs it manually | None — not the problem |
| 32k sample rate | ✅ Sufficient for speech | None — not the problem |
| Overtraining detector | ⚠️ Fires on noisy signal | **HIGH** if enabled at low threshold |
| `c_spk` speaker loss | ⚠️ No effect either way | None measured |
| Default epochs (50) | ⚠️ User overrides to 200+ | None for this user |
| No best-epoch save | ⚠️ May miss peak quality | Medium |
| Batch size heuristic | ⚠️ RAM-based, not VRAM | Medium (OOM risk) |
| Segment duration | ✅ 3.0 vs 3.7s minor | Negligible |

**If overtraining detection is disabled and the user runs 200–300 epochs with 10–15 min of clean audio, the pipeline should produce a good model.** The remaining question is whether it actually does — if quality is still average with detection off, the issue is likely inference-side (index rate, pitch shift, protect settings) rather than training.

---

## Recommended Next Steps

1. **Fix the overtraining detector** (track epoch-average, not last-batch) — eliminates the one confirmed training bug.
2. **Add best-epoch checkpoint saving** — gives a recovery point without changing training behaviour.
3. **Fix batch size heuristic** — VRAM-based to prevent OOM on CUDA machines.
4. **If quality is still poor with detection disabled:** investigate inference settings (index_rate, protect, f0 method, pitch shift) — the problem may not be in training at all.
