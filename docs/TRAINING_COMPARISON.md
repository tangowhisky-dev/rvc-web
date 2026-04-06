# Training Pipeline Comparison: rvc-web vs codename-rvc-fork-4

Concise technical analysis of both training approaches, with quality implications for voice cloning.

---

## 1. Architecture Overview

| Dimension | rvc-web | codename-rvc-fork-4 |
|---|---|---|
| Base model | VITS2 NSF | VITS (base, optional VITS2 mode) |
| Vocoder options | HiFi-GAN, RefineGAN | HiFi-GAN, RefineGAN, RingFormer v1/v2, APEX-GAN |
| Discriminators | MPD + MSD (v2) or MPD + MSD + MRD (v3/RefineGAN) | MPD + MSD + MRD, COMBD+SBD, UnivHD, custom per-vocoder |
| Embedder | SPIN-v2, SPIN, ContentVec, HuBERT | HuBERT (primary), ContentVec |
| Sample rate | 32k (all) | 40k or 48k |
| Speaker loss | ECAPA-TDNN cosine similarity (`combined` mode) | None |

---

## 2. Loss Functions

### 2.1 Spectral (Mel) Loss

**rvc-web:**
```
loss_mel = L1(mel(y_hat), mel(y)) × c_mel          # HiFi-GAN path
loss_mel = MultiScaleMelLoss(y_hat, y) × c_mel/3   # RefineGAN path
```
Single-scale L1 on the standard training mel (1024 FFT, 80 bins, fixed params). With RefineGAN, 7-scale mel loss is used.

**fork-4:**
Three configurable options:
- **L1 Mel Loss** — same as rvc-web HiFi-GAN path
- **Multi-Scale Mel Loss** — 7 STFT scales (window 32→2048), L1 on log-mel. Identical implementation to rvc-web's RefineGAN path.
- **Multi-Resolution STFT Loss** — `auraloss` library, mel-scaled perceptual weighting, 3 FFT sizes (1024/2048/4096). Operates in STFT magnitude space rather than mel filterbank space.

**Key difference:** fork-4's Multi-Res STFT loss adds phase-aware perceptual weighting that rvc-web lacks. However, the "mel similarity" metric fork-4 prints each epoch (`mel_spec_similarity = 100 - L1*100`, clamped 0–100) is purely a **diagnostic display** — it is not a loss term. It measures the L1 distance between single-scale mel slices at the last batch of each epoch. It is noisy (single batch, not epoch average) and not used to drive optimisation. The name suggests it optimises for mel similarity; it doesn't.

### 2.2 Adversarial Loss

**rvc-web:** LSGAN only — `mean((1-D)²)` for generator, `mean((1-D_r)²) + mean(D_g²)` for discriminator.

**fork-4:** Configurable:
- **LSGAN** — same as rvc-web
- **TPRLS** (Truncated Paired Relative Least Squares) — robust discriminator loss using median-centering to reduce mode collapse risk; more stable when discriminator dominates early
- **Hinge** — `max(0, 1-D)` / `max(0, 1+D)`, preferred for many GAN papers

**Quality implication:** TPRLS and hinge losses are empirically more stable for voice GAN training. LSGAN can oscillate when the discriminator learns too fast relative to the generator. rvc-web is locked to LSGAN.

### 2.3 KL Divergence

Identical formulation in both (`logs_p - logs_q - 0.5 + 0.5*(z_p-m_p)²*exp(-2*logs_p)`).

**fork-4 addition:** Optional **KL annealing** — `kl_beta = 0.5*(1 - cos(step/cycle * π))` cyclic schedule. Starts KL at ~0, ramps to 1 over `kl_annealing_cycle_duration` epochs, then repeats. Prevents posterior collapse early in training. rvc-web uses fixed `kl_beta = 1.0` throughout.

### 2.4 Feature Matching Loss

Both use `2 × Σ mean|fmap_r - fmap_g|`. Identical.

### 2.5 Speaker Identity Loss (rvc-web only)

rvc-web `combined` mode adds:
```
loss_spk = cosine_distance(ECAPA(y_hat_slice), profile_embedding) × c_spk
```
ECAPA-TDNN encodes 400ms segments; the reference embedding is pre-extracted from target audio. This directly penalises speaker divergence at the embedding level — something fork-4 has no equivalent of.

**Quality implication:** Speaker loss gives the generator an explicit gradient signal to preserve voice identity, not just waveform fidelity. Whether it helps at the fine-tuning scale (100–300 epochs) depends heavily on `c_spk` weight balance. Too high → constrains the generator against mel/KL objectives. Too low → negligible effect.

### 2.6 RingFormer Spectral Decomposition Loss (fork-4 only)

For RingFormer vocoder:
```
loss_sd = (L1(|STFT(y)|, mag_predicted) + phase_loss(y_stft, y_hat_stft)) × 0.7
```
`phase_loss = mean(1 - Re(x_norm · g_norm*))` — measures cosine similarity in complex STFT domain. Explicitly optimises phase coherence, which improves transient sharpness and reduces smearing artefacts. rvc-web has no phase loss.

### 2.7 Envelope Loss (fork-4, partial)

```
loss_env = MaxPool1d(y_real) L1 MaxPool1d(y_fake) + negative polarity
```
Matches audio amplitude peaks and troughs. Present in `losses.py` but not wired into the main training loop in the reviewed code. Experimental.

---

## 3. Optimiser and Learning Rate

**rvc-web:**
- AdamW, fixed `lr=1e-4` (or sqrt-scaled at non-32 batch sizes)
- ExponentialLR decay `γ=0.999` per epoch — slow, monotonic
- No warmup

**fork-4:**
- Configurable: AdamW, RAdam, DiffGrad, Ranger21, AdamSPD
- AdamSPD adds a "stiffness penalty" that projects weights back toward pre-trained values — explicitly limits drift from the pretrain, analogous to L2 regularisation in weight space
- Configurable LR schedulers: exponential (per-epoch or per-step), cosine annealing
- Optional warmup phase
- Optional gradient clipping schedule (tight clip early, released later)
- Separate LR for G and D (`custom_lr_g`, `custom_lr_d`)

**Quality implication:** AdamSPD's stiffness penalty is conceptually similar to LoRA's low-rank constraint — it prevents catastrophic forgetting of the pretrain's learned voice prior. For short fine-tuning runs (<200 epochs) this likely helps preserve naturalness. Cosine annealing tends to find flatter minima than monotonic exponential decay, which generally improves generalisation.

---

## 4. Checkpoint Strategy and Resume

**rvc-web:**
- `G_latest.pth` / `D_latest.pth` — explicit fixed names, no epoch disambiguation
- `G_best.pth` — saved on every new best epoch-average generator loss
- Resume: explicit path load, `epoch_str = iteration + 1`
- Best-epoch written directly to DB via `TrainingDBWriter`

**fork-4:**
- `G_{global_step}.pth` — step-encoded names, no ambiguity on resume glob
- Resume: regex glob on filename, derives epoch from step count
- No `G_best.pth` equivalent — best model identification left to the user via TensorBoard mel similarity metric

**Quality implication:** rvc-web's `G_best.pth` is a practical advantage — you can always export the best checkpoint regardless of when training is stopped. fork-4 requires manually identifying the best checkpoint from TensorBoard logs.

---

## 5. Two-Stage Training Protocol (fork-4 only)

fork-4 includes an experimental TSTP mode:
1. **Phase 1:** Train all modules normally until epoch-average KL < 0.1
2. **Phase 2:** Freeze encoders (enc_p, enc_q, flow, emb_g), train decoder only; accelerate LR decay

The intent: once the latent space is aligned (KL near zero), the remaining gradient should flow only to the vocoder/decoder to refine audio quality without disturbing the aligned representations.

rvc-web has no equivalent. With 200-epoch fine-tuning runs the KL typically drops to <0.1 by epoch 30–50, meaning Phase 2 could occupy the majority of training — potentially yielding cleaner decoder output.

---

## 6. Sample Rate and Frequency Coverage

**rvc-web:** 32k throughout — Nyquist 16kHz, covers all perceptually relevant speech frequencies. Lower VRAM, faster training.

**fork-4:** 40k or 48k — Nyquist 20/24kHz. Captures the 16–20kHz "air" range. Marginal for speech cloning; relevant for singing voice synthesis.

For speech-only use (rvc-web's stated target), 32k is sufficient and the reduced resolution speeds convergence.

---

## 7. What "mel similarity" Actually Is in fork-4

The metric printed as `Mel Spectrogram Similarity: XX.XX%` each epoch is:
```python
loss_mel = F.l1_loss(y_hat_mel, y_mel)          # last batch of epoch only
similarity = (1.0 - loss_mel).clamp(0, 1) * 100  # not epoch average
```
This is **not a loss function** — it's a display metric computed on the final batch slice, not epoch-averaged, and not used in backprop. It gives a rough intuition (higher = mel closer to target) but is too noisy to be authoritative. The actual spectral loss driving training is the separately computed `loss_mel` term (Multi-Scale Mel or L1 depending on config).

---

## 8. Verdict: Which Approach Produces Better Voice Cloning Quality

| Quality Factor | rvc-web | fork-4 | Winner |
|---|---|---|---|
| Spectral loss coverage | Single-scale L1 (HiFi-GAN) or 7-scale (RefineGAN) | 7-scale or multi-res STFT with perceptual weighting | **fork-4** (perceptual weighting) |
| Adversarial loss stability | LSGAN only | LSGAN / TPRLS / Hinge selectable | **fork-4** |
| Speaker identity signal | ECAPA cosine loss (combined mode) | None | **rvc-web** |
| Pretrain preservation | ExponentialLR + standard AdamW | AdamSPD stiffness + cosine annealing | **fork-4** |
| Phase coherence | None | Phase loss for RingFormer | **fork-4** (RingFormer) |
| Best epoch capture | G_best.pth automatic | Manual via TensorBoard | **rvc-web** |
| KL collapse prevention | None (β=1 fixed) | KL annealing available | **fork-4** |
| Embedding quality | SPIN-v2 (better content disentanglement) | HuBERT primarily | **rvc-web** |
| Operational simplicity | One config, sensible defaults | Many knobs, easy to misconfigure | **rvc-web** |

### Summary

**fork-4 has a richer loss surface** — multi-res STFT with perceptual weighting, TPRLS adversarial loss, KL annealing, and phase loss (RingFormer) collectively give the optimiser more signal about perceptual quality. AdamSPD's stiffness penalty and TSTP's encoder-freezing phase both help preserve the pretrained voice prior, which matters most in short fine-tuning runs.

**rvc-web has two advantages fork-4 lacks entirely**: SPIN-v2 embedder (better speaker-content disentanglement than HuBERT, meaning cleaner content features as decoder input) and explicit speaker identity loss via ECAPA-TDNN (directly penalises speaker drift during fine-tuning).

**Practical verdict for speech cloning at 100–300 epochs:**  
The most impactful delta is the adversarial loss (TPRLS/Hinge vs LSGAN) and KL annealing. LSGAN can stall when the discriminator overtakes the generator early — something we observe as `loss_gen` plateauing above 3.0 while `loss_mel` keeps improving. Adopting TPRLS or hinge loss in rvc-web, and adding cyclic KL annealing, would close the gap on fork-4's quality advantage without requiring the full TSTP machinery.

SPIN-v2 + ECAPA speaker loss is rvc-web's meaningful differentiator and should be preserved.
