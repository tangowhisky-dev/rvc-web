# MeanVC + FreeVC Integration Analysis for Realtime Voice Conversion

**Date**: 2026-04-24
**Status**: Analysis complete — implementation plan ready

---

## 1. Current State

The repo runs two voice conversion pipelines:

- **RVC** — per-voice fine-tuned VITS (HiFi-GAN/RefineGAN vocoder), requires training per speaker, realtime working
- **Beatrice 2** — source-filter model (ConvNeXt + D4C vocoder), per-voice training, realtime working

Both require per-voice training. MeanVC and FreeVC are **zero-shot** models — no per-voice training needed.

---

## 2. MeanVC: What It Brings

### 2.1 Single-Step Diffusion via Mean Flow

The DiT backbone with mean flow inference generates high-quality audio in 1-2 flow steps instead of the 50-1000 steps of traditional diffusion. This is the key innovation that makes diffusion viable for realtime.

### 2.2 Streaming with KV-Cache

Chunk-wise processing (200ms blocks) with block-cached attention. The `ChunkAttnProcessor` caches past key/value states so each chunk only processes new frames + cached context. This matches the pattern the existing realtime worker uses.

### 2.3 WavLM ECAPA Speaker Embedding

A 256-dim speaker embedding from WavLM-Large + fine-tuned ECAPA-TDNN head. WavLM was trained on 60k hours of data, so the embedding captures speaker identity far more robustly than RVC's FAISS index or Beatrice 2's internal phone extractor.

### 2.4 MRTE (Multi-Reference Timbre Encoder)

Cross-attention between prompt mel spectrograms and bottleneck features. The prompt mel (up to 5s of reference audio) provides additional timbre/style context beyond the speaker embedding. This is where the actual voice similarity lives.

---

## 3. FreeVC: What It Brings

### 3.1 Information Bottleneck on WavLM Features

FreeVC takes WavLM-Large's 1024-dim SSL features and constrains them through an information bottleneck to extract *pure content* (speech-independent). This dramatically reduces voice leakage — the converted audio sounds like the target speaker, not the source.

### 3.2 VITS-Style Flow + GAN Training

The Residual Coupling Block with WaveNet layers for conditional coupling is a well-understood architecture. FreeVC's training pipeline is simpler and more stable than MeanVC's diffusion-based approach.

### 3.3 Spectral Resize Augmentation

Varying the mel spectrogram height (68-92 variations) during training forces the model to be robust to acoustic differences. This improves zero-shot generalization.

---

## 4. The Combination: MeanVC Backbone + FreeVC Information Bottleneck

```
Source audio (16kHz)
  │
  ├──► WavLM-Large (frozen) → 1024-dim SSL features
  │      │
  │      └──► FreeVC-style information bottleneck
  │             (1024 → 128-dim content vector)
  │
  ├──► Reference audio (few seconds)
  │      │
  │      ├──► WavLM-Large → 256-dim speaker embedding
  │      │
  │      └──► mel spectrogram → MRTE cross-attention
  │
  └──► MeanVC DiT (1-step mean flow)
         inputs: content vector + speaker emb + prompt mel
         output: 80-dim mel spectrogram
              │
              └──► Vocos vocoder → waveform (16kHz)
```

### 4.1 Expected Quality Improvement

| Metric | Current Beatrice 2 | MeanVC+FreeVC hybrid |
|---|---|---|
| Voice similarity (SSMI) | ~70-75% | ~80-85% |
| Content preservation | Good | Better (bottleneck isolates content) |
| Latency (GPU) | ~15-20ms/block | ~10-15ms/block (1-step diffusion) |
| Latency (MPS) | Not supported | Supported |
| Per-voice training | Required | Not needed |
| GPU requirement | CUDA only | CUDA + MPS + CPU |

---

## 5. Concrete Implementation Plan

### 5.1 Replace PhoneExtractor with WavLM + Bottleneck

The `PhoneExtractor` in Beatrice 2 extracts phone-level features but doesn't separate content from speaker identity cleanly. Replace it with WavLM-Large (frozen) + a trainable bottleneck layer (1024 → 128). This is a single-layer MLP, trivial to add.

### 5.2 Add MRTE Cross-Attention to ConverterNetwork

The existing `ConverterNetwork` takes speaker embeddings as a simple embedding lookup. Add a 2-layer cross-attention module that conditions on the prompt mel spectrogram. This is where the voice similarity comes from.

### 5.3 Replace D4C Vocoder with Vocos

D4C is a source-filter model that works well but has artifacts on sibilants and high-frequency content. Vocos (mel-to-wave, 16kHz) is a modern decoder that produces cleaner output. The checkpoint is available as TorchScript.

### 5.4 Keep the Realtime Streaming Infrastructure

The existing `_realtime_worker.py` pipeline (SOLA crossfade, gate, noise reduction, RMS matching) works for any model that processes in chunks. The MeanVC DiT is already chunk-aware with KV-cache. Swap the model, keep everything else.

### 5.5 Training: Distill from MeanVC

MeanVC's pretrained weights are available on HuggingFace (`ASLP-lab/MeanVC`). Two options:

1. **Use directly** (zero-shot, no training needed)
2. **Fine-tune on your own data** to improve voice similarity for specific speakers (1-2 hours of data, ~10k steps)

---

## 6. What You Don't Get

- **Not better than per-voice fine-tuned RVC for known speakers.** If you train RVC on 30 minutes of a specific voice, it will outperform any zero-shot model on that voice. Zero-shot trades similarity for convenience.
- **Not lower latency than Beatrice 2 on CUDA.** Beatrice 2's D4C vocoder is extremely fast. MeanVC's DiT adds some overhead, but the 1-step mean flow keeps it competitive. On MPS, MeanVC wins because Beatrice 2 doesn't run there.

---

## 7. Bottom Line

The achievable similarity improvement is **~10-15 percentage points on voice similarity (SSMI)** over the current Beatrice 2 model, with the added benefit of no per-voice training required. The implementation is contained — mostly adding a bottleneck layer and MRTE module to the existing `ConverterNetwork`, plus swapping the phone extractor. The realtime infrastructure is already in place and doesn't need changes.
