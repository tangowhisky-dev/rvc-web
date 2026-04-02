# Training Strategy

This document explains what the model is optimising, why each loss term exists, how
the weights were chosen, and how the best-epoch checkpoint is selected.  It covers
both the HiFi-GAN and RefineGAN vocoder paths and documents all the decisions that
are specific to this codebase.

---

## Architecture recap

The model trained here is a **VITS-derived voice conversion network** (`SynthesizerTrnMsNSFsid`).
It has four principal components:

```
phone (HuBERT features) ──┐
pitch (F0)               ──┤→ TextEncoder (prior) → m_p, logs_p
                           │
spec (linear)            ──┤→ PosteriorEncoder     → z, m_q, logs_q
                           │
speaker ID               ──┘→ Embedding            → g  [B, 256, 1]
                                                              │
                          z ──→ NormFlow (forward) → z_p    │
                          z_p ← NormFlow (reverse) ← z_p    │ (inference)
                                                              │
                    z_slice ──→ Decoder (NSF/RefineGAN) ──→ y_hat
```

- **Prior** (`enc_p`): predicts a distribution over the latent `z` given only the HuBERT
  features and F0.  At inference this is what generates audio — no ground-truth spectrogram.
- **Posterior** (`enc_q`): encodes the ground-truth spectrogram `y` into `z`.  Only used
  during training to give the generator a "better-than-prior" latent to decode from.
- **Normalising flow**: maps `z` (posterior) → `z_p` and `z_p` → `z` (reverse at inference).
  Bridges the gap between the prior and posterior distributions.
- **Decoder**: upsample `z_slice` × 320 to 12,800 audio samples (0.4 s at 32 kHz).

During training the decoder sees the posterior-derived latent.  During inference it sees
the prior-derived latent.  The training objective is designed so the prior and posterior
are close enough that inference quality matches training quality.

---

## The loss terms

Five losses are summed to produce `loss_gen_all`, the quantity backpropagated through
the generator in every training step:

```
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_spk
```

There is also a separate `loss_disc` term backpropagated through the discriminator, which
does not affect generator weights.

### 1. Adversarial generator loss — `loss_gen`  (weight 1.0)

```python
def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:               # one per sub-discriminator
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)    # least-squares GAN
        loss += l
    return loss, gen_losses
```

**What it does:** For each sub-discriminator, compute the mean squared error between
the discriminator's output on the *fake* sample and the ideal output of `1` (real).
Summed across all sub-discriminators.

**Why this formulation:** Least-squares GAN (LSGAN) rather than the original
BCE formulation.  LSGAN gradients don't saturate when the discriminator is confident,
giving stable training throughout without mode collapse.

**Discriminator companion (`loss_disc`):**
```python
r_loss = mean((1 - D(real))**2)    # real should output 1
g_loss = mean((0 - D(fake))**2)    # fake should output 0
loss_disc = sum(r_loss + g_loss)
```
The discriminator is updated first each step with `y_hat.detach()` so its gradients
do not flow into the generator.  The generator is then updated second with `y_hat`
(no detach) so the discriminator's signal *does* flow back through it.

**Weight 1.0:** This is the baseline "fool the discriminator" signal.  It receives
weight 1.0 because it is a unitless competition signal normalised by the number of
discriminators.  Adding a weight here would just rescale the units of the other losses.

---

### 2. Feature matching loss — `loss_fm`  (weight 1.0, built-in ×2)

```python
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):        # per sub-discriminator
        for rl, gl in zip(dr, dg):            # per layer
            rl = rl.float().detach()           # real features, no grad
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))   # L1
    return loss * 2
```

**What it does:** For every internal feature map of every sub-discriminator, compute
the L1 distance between the real-audio feature maps and the fake-audio feature maps.
The discriminator is run again — this time *not* detached — so these feature map
gradients flow back to the generator.

**Why this exists:** Adversarial loss alone is a weak perceptual signal.  It only tells
the generator "the discriminator is not fooled" but not *in what direction* to improve.
Feature matching provides dense per-layer guidance: match not just the final score but
the full internal representation.  This significantly accelerates convergence and
stabilises audio quality.

**The ×2 multiplier:** Comes from the original HiFi-GAN paper's implementation.  It
empirically gives feature matching roughly the same order of magnitude as adversarial
loss without a separate tuned weight.  Weight is effectively ~2.0 relative to
`loss_gen`.

---

### 3. Mel spectrogram loss — `loss_mel`  (weight **c_mel = 45**)

**HiFi-GAN path (single-scale):**
```python
y_mel     = slice_on_last_dim(mel_of_real, ...)   # [B, 80, 40]
y_hat_mel = mel_spectrogram_torch(y_hat, ...)      # [B, 80, 40]
loss_mel  = F.l1_loss(y_mel, y_hat_mel) * c_mel    # c_mel = 45
```

**RefineGAN path (multi-scale):**
```python
loss_mel = MultiScaleMelSpectrogramLoss()(wave, y_hat) * c_mel / 3.0
```

The multi-scale loss computes 7 independent log-mel L1 losses across STFT scales:

| STFT window | Mel bins | Captures |
|---|---|---|
| 32 samples | 5 | Sub-millisecond transient energy |
| 64 | 10 | Consonant attacks |
| 128 | 20 | Formant structure |
| 256 | 40 | Mid-range spectral shape |
| 512 | 80 | Full-range spectrum |
| 1024 | 160 | Fine spectral resolution |
| 2048 | 320 | Long-context harmonic coherence |

The `/3.0` adjustment for RefineGAN keeps the summed multi-scale loss in the same
numeric range as the single-scale L1, so `c_mel = 45` remains the right magnitude.

**Why c_mel = 45 is so large:**

The raw L1 mel loss is a small number (order ~0.4–0.7 at convergence).  The adversarial
and feature matching losses are also small numbers (order ~1–5).  Without weighting,
the mel loss would be 10–15× smaller and the model would produce perceptually poor audio
that "fools the discriminator" without accurately reconstructing the spectral envelope.

`c_mel = 45` brings the mel loss to the same order of magnitude as the sum of the other
terms.  The practical effect is that roughly 85% of the generator gradient signal comes
from the mel term, ensuring the decoder learns to reproduce the correct spectral shape
before it worries about convincing the discriminator.  This is the dominant force for
audio fidelity.

The value 45 comes from the original VITS and HiFi-GAN papers and has been validated
empirically across many TTS/VC systems.  Our codebase uses it unchanged.

---

### 4. KL divergence loss — `loss_kl`  (weight **c_kl = 1.0**)

```python
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl  = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * exp(-2 * logs_p)
    kl  = sum(kl * z_mask) / sum(z_mask)
    return kl
```

**What it does:** Measures KL divergence between the posterior distribution `q(z|x)`
(encoded from the real spectrogram) and the prior distribution `p(z|text, pitch)` after
the normalising flow: KL( q(z_p) || p(z_p) ).

In plain terms: forces the distribution the prior produces to match the distribution
the posterior produces.  If these diverge, inference quality drops because at inference
time only the prior is available.

**Why weight 1.0:** The KL term is already normalised by the mask sum, giving it a
natural scale of ~1–3.  A weight of 1.0 keeps it in balance with `loss_gen` and
`loss_fm`.  Higher weights cause posterior collapse (the posterior becomes indistinguishable
from the prior, losing speaker-specific detail); lower weights allow the prior-posterior
gap to grow and degrade inference quality.  1.0 is the standard VITS value.

---

### 5. Speaker identity loss — `loss_spk`  (weight **c_spk**, default 2.0)

```python
def speaker_loss(gen_emb, ref_emb):
    sim = F.cosine_similarity(gen_emb, ref_emb).mean()
    return 1.0 - sim
```

**What it does:** Computes cosine similarity between:
- `gen_emb`: ECAPA-TDNN embedding of the generated 0.4s segment
- `ref_emb`: Pre-computed ECAPA-TDNN embedding of the *entire training corpus*
  (the "profile embedding", computed once before training starts)

The loss is `1 - cosine_similarity`, ranging [0, 2].  Zero means identical speaker
identity; 2 means maximally opposite.

**Why a profile-level reference (not per-segment):**

ECAPA-TDNN was trained on 2–4 second utterances.  The 0.4 s training segments are
too short for reliable speaker characterisation — embeddings would be noisy and
content-dependent rather than speaker-dependent.  Computing one embedding over all
profile audio (minutes, not milliseconds) gives a stable, content-agnostic speaker
identity target.  Each step compares the generated 0.4 s against that stable target.

**Why c_spk = 2.0 (not higher):**

The cosine loss value is in [0, 2] while mel loss (unnormalised) is also in the 0–2
range before the ×45 multiplier.  c_spk = 2.0 gives speaker loss roughly the same
order of magnitude as `loss_gen + loss_fm` (sum ~2–10), making it a meaningful
regulariser without drowning the reconstruction signal from mel loss.

Setting c_spk too high (> 5) causes the model to sacrifice spectral fidelity in order
to match speaker timbre — audible as a muffled or blurred output.  Setting it to zero
disables speaker conditioning during training (the model relies solely on the speaker
embedding table `emb_g`, which is a learned lookup vector per speaker ID, not a
perceptual similarity signal).

**c_spk = 0 is valid** for simple fine-tunes with a single speaker and clean audio.
The speaker embedding table `emb_g` still encodes speaker identity; c_spk > 0 adds an
explicit perceptual alignment signal on top of that.

---

## Discriminator architecture

Two configurations depending on vocoder:

### HiFi-GAN  (`disc_version = "v2"`)

```
DiscriminatorS           — full-resolution time-domain conv
DiscriminatorP(2)        — reshape waveform to period-2 matrix, conv
DiscriminatorP(3)
DiscriminatorP(5)
DiscriminatorP(7)
DiscriminatorP(11)
DiscriminatorP(17)
DiscriminatorP(23)
DiscriminatorP(37)
                         Total: 9 sub-discriminators
```

**DiscriminatorS** focuses on long-range temporal coherence (rumble, continuity).
**DiscriminatorP** with period N folds the waveform into an [T/N, N] matrix and
applies 2D convolutions.  Different periods capture different harmonic structures:
small primes (2, 3, 5) capture fine periodicity; large primes (17, 23, 37) capture
pitch-period-scale features without aliasing between discriminators.

### RefineGAN  (`disc_version = "v3"`)

```
DiscriminatorS           — same
DiscriminatorP(2)
DiscriminatorP(3)
DiscriminatorP(5)
DiscriminatorP(7)
DiscriminatorP(11)
DiscriminatorR([1024, 120, 600])   — resolution discriminator
DiscriminatorR([2048, 240, 1200])
DiscriminatorR([512, 50, 240])
                         Total: 9 sub-discriminators
```

RefineGAN replaces the large-period discriminators (17, 23, 37) with three
`DiscriminatorR` that operate on STFT spectrograms at different resolutions.
These provide explicit frequency-domain feedback, which pairs well with the
multi-scale mel loss — both the loss and the discriminator signal operate in
the frequency domain at multiple scales.

---

## Optimiser and learning rate schedule

```python
optim_g = AdamW(net_g.parameters(), lr=lr_scaled, betas=(0.8, 0.99), eps=1e-9)
optim_d = AdamW(net_d.parameters(), lr=lr_scaled, betas=(0.8, 0.99), eps=1e-9)
scheduler = ExponentialLR(gamma=0.999875, last_epoch=epoch_str - 2)
```

**AdamW:** standard for transformer-adjacent models.  `betas=(0.8, 0.99)` vs the
usual `(0.9, 0.999)` gives faster momentum decay (half-life ~5 steps vs ~10) and
faster second-moment decay.  This makes the optimiser more responsive to fresh gradient
directions — beneficial in fine-tuning where the loss landscape changes quickly.

**ExponentialLR decay:** After 100 epochs, LR × 0.999875^100 ≈ 0.988 LR.  After
1000 epochs, LR × 0.882.  After 3000 epochs, LR × 0.687.  The decay is slow enough
that it doesn't matter much for short runs (< 200 epochs) but prevents divergence
on long runs by reducing the effective step size as the model approaches a minimum.

**Batch-size learning rate scaling (sqrt rule):**

The pretrained weights were produced at `batch_size=32, LR=1e-4`.  This codebase
auto-scales LR at `_write_config` time:

```python
LR_eff = 1e-4 * sqrt(batch_size / 32)
```

| batch_size | LR |
|---|---|
| 4 | 3.54e-5 |
| 8 | 5.00e-5 |
| 16 | 7.07e-5 |
| 32 | 1.00e-4 |

The sqrt rule (rather than the linear rule used for SGD) is appropriate for AdamW
because the adaptive second-moment scaling partially compensates for gradient noise.
Without this correction, MPS training at bs=8 ran at 2× the correct effective LR,
producing a noisy loss curve and a higher loss floor than equivalent CUDA training.

---

## Best-epoch model selection

### What metric is tracked

The best epoch is identified by the **epoch-average generator adversarial loss** —
`avg_gen` — not the total loss `loss_gen_all`, and not `loss_mel`.

```python
# Inside _drain_stdout in training.py:
avg_gen = (sum of loss_gen values across all steps in the epoch) / step_count
```

This is the average across all training steps in the epoch of `loss_gen` —
the adversarial fooling-the-discriminator component only, without mel, KL, FM, or
speaker terms.

### Why not loss_mel?

`loss_mel` is the dominant term (×45) and converges monotonically for well-behaved
runs.  Using it as the best-model selector would produce the checkpoint from the last
epoch almost every time — unhelpfully, since the last epoch is already `G_latest.pth`.

`loss_gen` oscillates more.  It measures how well the generator currently fools the
discriminator, which is a more sensitive indicator of perceptual quality plateau
and overtraining.  When `loss_gen` stops improving while `loss_mel` continues improving,
the additional spectral refinement is no longer accompanied by perceptual gains.

### Why not loss_gen_all?

`loss_gen_all` is a weighted sum of five heterogeneous quantities.  Its absolute value
is dominated by `loss_mel × 45`.  Using it as a selection metric would be equivalent
to using `loss_mel` with noise added from the other terms.  `loss_gen` is a cleaner,
more interpretable signal.

### The minimum improvement threshold

```python
_min_delta = 0.004
is_best = avg_gen < current_best - _min_delta
```

Updates to `G_best.pth` are only written when `avg_gen` improves by at least 0.004.
This prevents spurious saves caused by numerical noise between consecutive epochs at
the same quality level.  0.004 is approximately 0.02% of a typical `loss_gen` value
of ~20 — a practically negligible difference that would not be audible.

### What happens to the best checkpoint

`G_best.pth` is written by `train.py` every time a new best epoch is detected.
After training ends (whether normally or via overtraining early stop), the artifact
copy phase in `_run_pipeline` does:

1. Runs `save_small_model(G_best.pth)` → writes `model_best.pth` in the profile
   directory (inference-ready fp16, ~54 MB, stripped of optimiser state)
2. Copies `G_best.pth` → `profile_dir/checkpoints/G_best.pth`
3. Updates `best_epoch` and `best_avg_gen_loss` in the DB

At inference time, the user can choose between `model_infer.pth` (from the final epoch,
most refined spectrally) and `model_best.pth` (from the best-generator epoch, potentially
more natural-sounding but less spectrally detailed).

---

## Overtraining detection

```python
_ot_consecutive_gen += 1 when avg_gen > current_best + _min_delta
terminate when _ot_consecutive_gen >= overtrain_threshold  (default 0 = disabled)
```

The logic mirrors ultimate-rvc's overtraining detection:

- `avg_gen` (epoch-average generator loss) is the primary signal.
- A separate `avg_disc` counter fires at `2 × overtrain_threshold` — the discriminator
  signal is noisier and less reliable, so it needs a longer confirmation window.
- When triggered, `train_proc.terminate()` is sent to stop training immediately at
  the current epoch boundary.
- The artifact phase then uses `G_best.pth` (written at the last best epoch before
  overtraining began) as the inference model.

This means the overtraining detector and the best-epoch selector share the same metric
and the same update logic — they are the same computation from different perspectives.

---

## Summary table

| Loss | Formula | Effective weight | Role |
|---|---|---|---|
| `loss_gen` | Σ MSE(1 − D(fake)) | 1.0 | Fool discriminator (perceptual) |
| `loss_fm` | 2 × Σ L1(fmap_real − fmap_fake) | ~2.0 | Match discriminator internals |
| `loss_mel` | L1(mel_real − mel_fake) | **45** | Spectral fidelity (dominant) |
| `loss_kl` | KL(posterior ‖ prior) | 1.0 | Latent alignment (inference quality) |
| `loss_spk` | 1 − cos(gen_emb, ref_emb) | 0–3 | Speaker identity |
| `loss_disc` | Σ MSE(D(real)−1) + MSE(D(fake)) | — | Discriminator only, no gen grad |

**Gradient origin of each generator update:**
- ~85% from `loss_mel` (spectral accuracy)
- ~8% from `loss_fm` (perceptual detail via discriminator internals)
- ~4% from `loss_gen` (adversarial realism)
- ~2% from `loss_kl` (latent regularisation)
- ~1% from `loss_spk` (speaker identity, if enabled)

These percentages are approximate and shift as training progresses — early training
is more adversarial, late training is dominated by mel refinement.

**Best-epoch metric:** epoch-average `loss_gen` (adversarial component only), with
minimum improvement threshold 0.004.  Written to `G_best.pth` by `train.py` on the
fly; converted to `model_best.pth` in the artifact phase.
