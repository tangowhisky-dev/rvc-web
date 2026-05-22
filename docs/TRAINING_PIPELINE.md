# RVC Training Pipeline — Complete Analysis

## Overview

RVC (Retrieval-based Voice Conversion) fine-tunes a pre-trained voice conversion model on a target speaker's audio. The pipeline chains multiple stages: audio preprocessing → F0 extraction → feature extraction → model training → FAISS index building.

---

## 1. Audio Preprocessing

### Input
- **Source**: Raw audio files uploaded to a profile (WAV, MP3, FLAC, etc.)
- **Sample rate**: Converted to **32,000 Hz** during preprocessing
- **Duration**: Any length — files are automatically chunked

### Chunking Algorithm (`preprocess.py`)

Each audio file is processed in two stages:

**Stage 1 — Silence Slicing** (`Slicer` class):
- Splits audio at silence boundaries (RMS < -42dB)
- Prevents chunking across silent gaps
- Produces variable-length slices

**Stage 2 — Overlapping Segments**:
- Each slice is divided into overlapping segments
- **Segment duration**: 3.0 seconds (96,000 samples at 32kHz)
- **Overlap**: 30% (0.9 seconds)
- **Stride**: 2.1 seconds (3.0 × 0.7)
- **Final segment**: Up to 3.3 seconds (captures trailing content)

```
File: [────────────────────────────────────────] 10 seconds
       [3.0s]
            [2.1s stride → 3.0s]
                 [2.1s stride → 3.0s]
                      [2.1s stride → 3.0s]
                           [remaining → up to 3.3s]
```

**Chunks per file** (approximate):
```
chunks = ceil((file_duration - 3.3) / 2.1) + 1
```
Example: 10-second file → 5 chunks

### Per-Chunk Processing
1. **High-pass filter**: 5th-order Butterworth, cutoff 48Hz (removes rumble)
2. **Peak normalization**: `audio = (audio/max × 0.9 × 0.75) + (1-0.75) × audio`
3. **Quality filter**: Segments with peak amplitude > 2.5 are discarded
4. **Output files**:
   - `0_gt_wavs/{idx}.wav` — 32kHz float32 WAV (ground truth waveform)
   - `1_16k_wavs/{idx}.wav` — 16kHz WAV (for F0 extraction and feature extraction)

### Output Summary
| Directory | Content | Format |
|-----------|---------|--------|
| `0_gt_wavs/` | 32kHz ground truth segments | WAV (float32) |
| `1_16k_wavs/` | 16kHz resampled segments | WAV (float32) |

---

## 2. F0 Extraction (`extract_f0_print.py`)

### Input
- `1_16k_wavs/*.wav` — 16kHz audio segments

### Method
- **Algorithm**: RMVPE (Robust Mel-spectrogram based Voice Pitch Estimator)
- **F0 range**: 50 Hz to 1100 Hz
- **F0 bins**: 256
- **Hop size**: 160 samples at 16kHz = **10ms per frame**

### Output
| File | Content | Shape |
|------|---------|-------|
| `2a_f0/{name}.wav.npy` | Coarse F0 (256-bin discrete pitch, int64) | [T_f0] |
| `2b-f0nsf/{name}.wav.npy` | Continuous F0 (float32, for NSF conditioning) | [T_f0] |

For a 3.0s segment: `3.0 × 16000 / 160 = 300` F0 frames

---

## 3. Feature Extraction (`extract_feature_print.py`)

### Input
- `1_16k_wavs/*.wav` — 16kHz audio segments

### Embedder
- **Default**: HuBERT (spin-v2)
- **Alternatives**: contentvec, hubert_base
- **Output dimension**: 768 (v2) or 256 (v1)
- **Frame rate**: 320-sample hop at 16kHz = **20ms per frame**

### Processing
1. Load HuBERT model from `assets/hubert/hubert_base.pt`
2. Process each 16kHz segment through HuBERT
3. Extract per-frame 768-dimensional feature vectors
4. **Repeat 2×** along time axis (`np.repeat(phone, 2, axis=0)`) to align with F0 resolution
5. Cap at 900 frames

### Output
- `3_feature768/{name}.npy` — shape `[min(2×T_feat, 900), 768]`

For a 3.0s segment: `3.0 × 16000 / 320 = 150` HuBERT frames → repeated to 300 frames

---

## 4. Filelist Construction

Each line in `filelist.txt`:
```
{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{spk_id}
```

Two mute padding entries are appended. The list is shuffled.

---

## 5. Training Data Loading (`data_utils.py`)

### TextAudioLoaderMultiNSFsid

**Loads per entry**:
1. **Features**: `.npy` → shape `[T_feat, 768]`, repeated 2×, capped at 900
2. **F0 data**: `pitch` (int64, coarse) and `pitchf` (float32, continuous) → shape `[T_f0]`
3. **Audio**: WAV at 32kHz → spectrogram `[513, T_spec]` + waveform `[1, T_wave]`
4. **Alignment**: Truncates to `min(len_phone, len_spec)` if lengths differ

### DistributedBucketSampler

**Bucket boundaries**: `[100, 200, 300, 400, 500, 600, 700, 800, 900]`

Samples are grouped by approximate spectrogram length to minimize padding waste within batches.

**Batch size**: `hps.train.batch_size × n_gpus` (typically 4-8 on MPS)

### TextAudioCollateMultiNSFsid

Zero-pads all elements to maximum length in batch. Returns:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `phone` | `[B, max_phone_len, 768]` | HuBERT features |
| `phone_lengths` | `[B]` | Actual feature lengths |
| `pitch` | `[B, max_phone_len]` | Coarse F0 (int64) |
| `pitchf` | `[B, max_phone_len]` | Continuous F0 (float32) |
| `spec` | `[B, 513, max_spec_len]` | Linear spectrogram |
| `spec_lengths` | `[B]` | Actual spectrogram lengths |
| `wave` | `[B, 1, max_wave_len]` | Waveform (32kHz) |
| `wave_lengths` | `[B]` | Actual waveform lengths |
| `sid` | `[B]` | Speaker IDs |

---

## 6. What an Epoch Consists Of

### Training Loop

```python
for epoch in range(epoch_str, hps.train.epochs + 1):
    train_and_evaluate(rank, epoch, hps, ...)
    scheduler_g.step()
    scheduler_d.step()
```

**Steps per epoch**: `len(train_loader)` = total batches from `DistributedBucketSampler`

Example: 50 chunks + 2 mute padding = 52 entries, batch_size=8 → **~7 steps per epoch**

### Data Caching

When `if_cache_data_in_gpu=True`:
- **First epoch**: All batches loaded from disk → moved to device → stored in cache list
- **Subsequent epochs**: Cache shuffled in-place, no disk I/O

---

## 7. Each Training Step

### Step 1: Generator Forward Pass

```python
y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = \
    net_g(phone, phone_lengths, spec, spec_lengths, sid, pitch, pitchf)
```

**Generator architecture** (`SynthesizerTrnMsNSFsid`):
1. **Speaker embedding**: `g = emb_g(sid)` → `[B, 256]`
2. **Text encoder** (prior): `m_p, logs_p = enc_p(phone, pitch)` → `[B, 192, T_phone]`
3. **Posterior encoder**: `z, m_q, logs_q = enc_q(spec, g=g)` → `[B, 192, T_spec]`
4. **Normalizing flow**: `z_p = flow(z, g=g)` → `[B, 192, T_spec]`
5. **Random slice**: `z_slice = rand_slice(z, segment_size=40)` → `[B, 192, 40]`
6. **NSF Generator** (HiFi-GAN): `y_hat = dec(z_slice, pitchf, g=g)` → `[B, 1, 12800]`

**Upsample rates**: `[10, 8, 2, 2]` → product = 320
- 40 latent frames × 320 = 12,800 samples = 0.4s at 32kHz

### Step 2: Random Segment Slicing

The full-length `wave` and `y_hat` are sliced to `segment_size`:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `wave` | `[B, 1, 12800]` | Ground truth waveform segment |
| `y_hat` | `[B, 1, 12800]` | Generated waveform segment |
| `y_mel` | `[B, 80, 40]` | Mel spectrogram of ground truth |
| `y_hat_mel` | `[B, 80, 40]` | Mel spectrogram of generated |

### Step 3: Discriminator Training

```python
y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
# Backward pass for discriminator
```

**MultiPeriodDiscriminator** (v2):
- 1 `DiscriminatorS` (time-domain)
- 8 `DiscriminatorP` (periods: 2, 3, 5, 7, 11, 17, 23, 37)
- Total: **9 sub-discriminators**

### Step 4: Generator Training

```python
y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)  # NOT detached
loss_fm = feature_loss(fmap_r, fmap_g)
loss_mel = L1(y_mel, y_hat_mel) × c_mel          # c_mel = 45
loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) × c_kl  # c_kl = 1.0
loss_gen, _ = generator_loss(y_d_hat_g)
loss_spk = speaker_loss(gen_emb, ref_emb) × c_spk  # if c_spk > 0
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_spk
# Backward pass for generator
```

### Loss Functions

| Loss | Formula | Weight | Purpose |
|------|---------|--------|---------|
| **Generator** | Σ MSE(1 - D(fake)) | 1.0 | Fool discriminator |
| **Feature Matching** | 2 × Σ L1(fmap_real - fmap_fake) | 1.0 | Match intermediate features |
| **Mel** | L1(mel_real - mel_fake) | 45 | Audio quality |
| **KL** | KL(posterior || prior) | 1.0 | Latent regularization |
| **Speaker** | 1 - cosine_sim(gen_emb, ref_emb) | c_spk (0-3) | Speaker identity |

---

## 8. Config Hyperparameters (`v2/32k.json`)

### Training
| Parameter | Value | Notes |
|-----------|-------|-------|
| `batch_size` | 4 | Overridden by UI (default 8) |
| `segment_size` | 12800 | Waveform samples per segment (0.4s) |
| `epochs` | 20000 | Overridden by UI (default 50) |
| `learning_rate` | 1e-4 | AdamW, auto-scaled by sqrt(batch_size/32) |
| `lr_decay` | 0.999875 | Per-epoch exponential decay |
| `c_mel` | 45 | Mel loss weight |
| `c_kl` | 1.0 | KL loss weight |
| `c_spk` | 0-3 | Speaker loss weight (UI configurable) |
| `fp16_run` | true | Mixed precision (auto-disabled on MPS) |
| `adv_loss` | lsgan | Adversarial loss: lsgan or tprls |
| `kl_anneal` | false | Cyclic cosine KL annealing schedule |
| `kl_anneal_epochs` | 40 | Cycle duration for KL annealing |
| `optimizer` | adamw | Optimizer: adamw or adamspd |

### Data
| Parameter | Value | Notes |
|-----------|-------|-------|
| `sampling_rate` | 32000 | |
| `hop_length` | 320 | 10ms at 32kHz |
| `n_mel_channels` | 80 | |

### Model (v2)
| Parameter | Value |
|-----------|-------|
| `inter_channels` | 192 |
| `hidden_channels` | 192 |
| `filter_channels` | 768 |
| `upsample_rates` | [10, 8, 2, 2] (product = 320) |
| `gin_channels` | 256 |

---

## 9. End-to-End Numbers (Example)

**Input**: 10 audio files, each ~10 seconds, no silence breaks

| Stage | Count | Details |
|-------|-------|---------|
| Preprocessing | 50 chunks | 5 chunks/file × 10 files |
| F0 Extraction | 50 files | ~300 frames each |
| Feature Extraction | 50 files | ~300 × 768 each |
| filelist.txt | 52 entries | 50 + 2 mute padding |
| Batches/epoch | ~7 | 52 entries / batch_size=8 |
| Steps/epoch | ~7 | One generator + one discriminator update each |
| Total steps (50 epochs) | ~350 | |

---

## 10. Speaker Loss Integration (ECAPA-TDNN)

### Implementation (Profile-Level Embeddings)

Profile-level embeddings are used — computed once before training from all profile audio, saved as `profile_embedding.pt` in the experiment directory.

```python
# In generator training step:
_wave_16k = resample(wave.squeeze(1), 32kHz → 16kHz)    # [B, T]
_yhat_16k = resample(y_hat.squeeze(1), 32kHz → 16kHz)   # [B, T]

with torch.no_grad():
    ref_emb = profile_embedding    # [1, 192] — loaded from profile_embedding.pt
gen_emb = speaker_encoder(_yhat_16k)        # [B, 192]

loss_spk = (1 - cosine_sim(gen_emb, ref_emb)) × c_spk
```

- Profile embedding provides a stable, content-agnostic speaker identity target
- Each training step compares the generated 0.4s segment against this fixed reference
- `c_spk` weight is configurable via the training API (default 2.0)
- Eliminates per-segment noise that would occur if embedding was computed from 0.4s segments
