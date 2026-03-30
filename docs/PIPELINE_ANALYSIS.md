# RVC-Web Pipeline Analysis & Optimization Guide

This document analyzes the current training and inference pipelines in rvc-web and provides recommendations for improving voice quality while minimizing latency for realtime voice conversion.

---

## Table of Contents

1. [Training Pipeline Analysis](#training-pipeline-analysis)
2. [Inference/Realtime Pipeline Analysis](#inference-realtime-pipeline-analysis)
3. [Quality Issues: Diagnosis & Solutions](#quality-issues-diagnosis--solutions)
4. [Model Validation Before Inference](#model-validation-before-inference)
5. [Fine-tuning Best Practices](#fine-tuning-best-practices)
6. [Recommendations for Voice Quality](#recommendations-for-voice-quality)
7. [Recommendations for Low-Latency Realtime](#recommendations-for-low-latency-realtime)
8. [Blackhole Integration](#blackhole-integration)
9. [Summary of Changes](#summary-of-changes)

---

## Training Pipeline Analysis

### Current Pipeline Phases

The current training pipeline (`backend/app/training.py`) executes these phases:

1. **Preprocess** - Audio resampling, noise removal, RMS normalization, silence trimming
2. **F0 Extraction** - Pitch contour extraction using RMVPE
3. **Feature Extraction** - HuBERT semantic feature extraction
4. **Training** - GAN-based VITS model training
5. **FAISS Index** - Build semantic search index for voice retrieval

### Current Implementation Observations

| Component | Current Implementation | Assessment |
|-----------|----------------------|------------|
| F0 Extraction | RMVPE (default) | Good - RMVPE is recommended for quality |
| Feature Extraction | HuBERT (v2, 768-dim) | Good - v2 provides better quality |
| Index | IVF256, Flat | Basic - can be optimized |
| Batch Size | Default 8 | Reasonable for most datasets |
| Epochs | User-configurable | Needs guidance |
| Pretrain | f0G48k.pth / f0D48k.pth | Good default pretrains |

### FAISS Index Current State

The current index worker (`backend/app/_index_worker.py:66`) uses:
```python
index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
```

**Issue**: Using simple Flat index is not optimal for speed/quality tradeoff.

---

## Inference/Realtime Pipeline Analysis

### Current Architecture

The realtime pipeline (`backend/app/_realtime_worker.py`) uses:

1. **Input**: 44.1kHz microphone input via sounddevice
2. **Resampling**: 44.1kHz → 16kHz (for model) and 44.1kHz → 48kHz (for output)
3. **Model Inference**: RVC with HuBERT + RMVPE + VITS
4. **Post-processing**: 
   - SOLA (Synchronous Overlap-Add) algorithm for block smoothing
   - Phase vocoder crossfade
   - Silence gate with hold/release
   - RMS envelope matching
5. **Output**: 48kHz to speaker/Blackhole

### Current Block Parameters

```python
_SR_48K = 48000
_BLOCK_48K = 5760           # 120ms output block at 48kHz
_EXTRA_TIME = 2.0           # 2 seconds historical context
_ZC = _SR_48K // 100        # 480 samples = 10ms (zero-crossing unit)
_SOLA_BUF_48K = 2 * _ZC     # 960 samples = 20ms (crossfade buffer)
_SOLA_SEARCH_48K = _ZC      # 480 samples = 10ms (search window)
```

### SOLA Crossfade Analysis

The SOLA (Synchronous Overlap-Add) algorithm eliminates block boundary artifacts caused by async inference drift. Key parameters:

| Parameter | Samples | Time @48kHz | Purpose |
|-----------|---------|-------------|---------|
| Block | 5760 | 120ms | Main inference output |
| SOLA Buffer | 960 | 20ms | Crossfade window |
| SOLA Search | 480 | 10ms | Alignment search range |
| **Total Output** | **7200** | **150ms** | Block + SOLA overhead |

**SOLA Overhead is FIXED at 30ms** - independent of block size:

```
Block Size    Total Output    SOLA Overhead    SOLA % of Total
-----------   -------------   -------------    --------------
120ms         150ms           30ms              20%
80ms          110ms           30ms              27%
60ms          90ms            30ms              33%
```

**Current SOLA buffer (20ms)** provides 4 cycles @ 200Hz - sufficient for voice crossfade.

### Latency Analysis

| Component | Current Value | Notes |
|-----------|---------------|-------|
| Block Size | 120ms | Can be reduced |
| SOLA Buffer + Search | 30ms | Fixed overhead |
| Historical Context | 2000ms | Needed for pitch extraction |
| Inference Time | ~50-100ms (MPS) | Varies by hardware |
| **Total Output Latency** | **150ms** | Block + SOLA |
| **End-to-End Latency** | **~200-250ms** | Includes audio device buffer |

---

## Quality Issues: Diagnosis & Solutions

### Common Quality Problems & Causes

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| **Robotic/Glitchy Voice** | Dataset has noise, reverb, or overlapping voices | Clean dataset with UVR |
| **Not Cloning Correctly** | Insufficient training data (<10 min), undertrained | Use 15-30 min clean data |
| **Excessive Noise** | Noisy source audio, no denoising step | Run UVR De-Noise |
| **Muffled Output** | Dataset sample rate mismatch | Match training sr to dataset sr |
| **Sibilance (harsh S sounds)** | Dataset has harsh sibilants or overfitted | De-ess dataset, try earlier epoch |
| **Breath Artifacts** | Protect parameter too low | Increase protect to 0.5-0.7 |

### Root Cause Analysis by Pipeline Phase

#### 1. Preprocessing Issues

**Current Implementation** (`preprocess.py`):
- Slicer threshold: -42 dB (may keep too much silence)
- Min length: 1500 samples (~93ms at 16kHz)
- High-pass filter: 48Hz (removes bass but may affect character)

**Problems**:
- Silence not trimmed properly → noisy training
- No explicit denoising step → noise baked into model
- No reverb removal → reverb learned as voice characteristic

#### 2. F0 Extraction Issues

- **RMVPE** is good but may fail on noisy audio
- Consider **Crepe** for cleaner extraction if compute allows

#### 3. Feature Extraction Issues

- Uses **HuBERT** (768-dim for v2) - good quality
- If audio is degraded → features are degraded

#### 4. Training Issues

| Issue | Impact | Fix |
|-------|--------|-----|
| Too few epochs | Undertrained, weak cloning | Train 100-300 epochs |
| Too many epochs | Overfitted, robotic | Stop at quality plateau |
| Wrong batch size | Unstable training | 4 for <30min, 8 for 30+min |
| Noisy loss graphs | Training instability | Reduce batch size |

#### 5. FAISS Index Issues

- **Current**: Simple IVF+Flat - may not retrieve optimal matches
- **Fix**: Use PQ+RFlat for better retrieval quality

---

## Model Validation Before Inference

### Before Using a Model in Production

#### 1. Check Training Logs

```bash
# View training loss curves
tensorboard --logdir logs/<experiment_name>
```

**What to look for**:
- `g/total` loss decreasing (not exploding)
- `kl_loss` positive and stable (not negative)
- No NaN values in any loss

#### 2. Audio Validation Tests

Test each saved epoch with:

```bash
# Batch inference test
python infer/modules/vc/pipeline.py \
  --input <test_audio.wav> \
  --model logs/<exp>/G_<epoch>.pth \
  --index logs/<exp>/added_<index>.index
```

**Listen for**:
- Clarity of cloned voice
- Naturalness (not robotic)
- Proper pitch matching
- No artifacts or noise

#### 3. Automated Quality Metrics

| Metric | Good Range | Bad Sign |
|--------|------------|----------|
| SNR (Signal-to-Noise) | >20 dB | <10 dB |
| Mel-Spectrogram Similarity | >0.7 | <0.5 |
| Pitch Correlation | >0.8 | <0.6 |

#### 4. Inference Parameter Sweep

Test different parameter combinations:

| Parameter | Range to Test | Notes |
|-----------|---------------|-------|
| index_rate | 0.0, 0.25, 0.5, 0.75, 1.0 | Higher = more voice retrieval |
| protect | 0.0, 0.25, 0.5, 0.75, 1.0 | Lower = less breath, may sound artificial |
| pitch | -12 to +12 | Match to target voice pitch |

---

## Fine-tuning Best Practices

### Dataset Preparation (Critical)

Based on latest AI Hub documentation:

#### 1. Audio Quality Requirements

| Requirement | Specification | Why |
|-------------|---------------|-----|
| Format | WAV/FLAC (lossless) | Preserve original quality |
| Sample Rate | 32k-48k (match target) | Prevents resampling artifacts |
| Channels | Mono | RVC trains on mono |
| Bit Depth | 16-bit minimum | Dynamic range preservation |

#### 2. Dataset Length

| Duration | Quality | Notes |
|----------|---------|-------|
| <10 min | Poor | Insufficient for cloning |
| 15-30 min | Good | Recommended for fine-tuning |
| 40+ min | Best | More natural results |

#### 3. Audio Cleaning Pipeline

**Recommended** (from AI Hub):

```
1. Remove instrumental (UVR - Inst V1 Plus)
2. Remove reverb (UVR - De-Reverb v2)  
3. Extract vocals (UVR - Vocals FV4)
4. Remove noise (UVR - Mel Denoiser v2)
5. De-ess harsh sibilants (RX11 De-ess)
6. Remove DC offset (Audacity/Audacity)
7. Normalize to -23 LUFS (Audacity)
8. Truncate silence (max 3s)
```

#### 4. Common Dataset Mistakes

| Mistake | Result | Fix |
|---------|--------|-----|
| Using MP3 | Artifacts from compression | Convert to WAV/FLAC |
| Including reverb | Muffled output | Remove with UVR |
| Multiple speakers | Voice bleeding | Single speaker only |
| Background music | Music learned as voice | Remove with UVR |
| Long silences | Training inefficiency | Truncate to <3s |
| Inconsistent volume | Unstable training | Normalize -23 LUFS |

### Training Parameters

#### Optimal Settings for Fine-tuning

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| **Epochs** | 100-300 | Test each, stop at plateau |
| **Batch Size** | 4 (<30min), 8 (30+min) | Balance speed/stability |
| **Save Every** | 10 epochs | Allow epoch selection |
| **Learning Rate** | Default (from pretrain) | Don't change for fine-tuning |
| **f0** | 1 (enabled) | Pitch-aware model |

#### Epoch Selection Strategy

1. Save every 10 epochs
2. Test each epoch with same reference audio
3. Listen for:
   - Voice clarity
   - Naturalness
   - Pitch accuracy
   - No robotic artifacts
4. **Best epoch is NOT necessarily the last one**

### FAISS Index Training

**Current rvc-web uses**: Simple IVF+Flat

**Recommended for quality**:
```python
# In _index_worker.py
index = faiss.index_factory(768, "IVF512,PQ128x4fs,RFlat")
```

This provides:
- **Faster search** (PQ compression)
- **Better retrieval** (RFlat recalculation)
- **Higher quality** voice cloning

---

## Recommendations for Voice Quality

### 1. Training Dataset Guidelines

Based on latest community best practices:

- **Dataset Duration**: 10-30 minutes minimum for good quality
- **Audio Quality**: Clean, mono, 44.1kHz+ recommended
- **Silence Handling**: Remove long silences (>5s)
- **Noise Reduction**: Apply denoising before training
- **Speaker Consistency**: Single speaker preferred for fine-tuning

### 2. Optimal Training Parameters

| Parameter | Recommended Value | Rationale |
|-----------|------------------|-----------|
| Epochs | 100-300 | Test each epoch; stop when quality plateaus |
| Batch Size | 8 (30+ min data), 4 (<30 min) | Smaller for limited data |
| Save Every | 10 epochs | Balance between saves and speed |
| log_interval | 10 | Reduce MPS CPU sync overhead |

### 3. FAISS Index Optimization

**Current**: `IVF{n_ivf},Flat`

**Recommended** (from official RVC wiki):
```python
# Use PQ for faster search with minimal quality loss
index = faiss.index_factory(768, "IVF512,PQ128x4fs,RFlat")
```

This uses:
- **IVF512**: 512 Voronoi cells (clusters)
- **PQ128x4fs**: 128-byte product quantization with FastScan
- **RFlat**: Recompute exact distances for top-k results

### 4. F0 Extraction Methods

Current pipeline uses RMVPE (default). Based on inference settings research:

| Method | Quality | Speed | Best For |
|--------|---------|-------|----------|
| RMVPE | Good | Fast | Default, harmonic-rich voices |
| Crepe | Best | Slow | Clean audio, soft/whispery voices |
| FCPE | Very Good | Fast | Realtime, newer algorithm |

**Recommendation**: Keep RMVPE for realtime (fast), but consider Crepe for batch inference when quality is paramount.

### 5. Embedder Model Options

Current uses standard HuBERT. Newer options:

| Model | Quality | Notes |
|-------|---------|-------|
| HuBERT (default) | Good | Standard, reliable |
| ContentVec | Better | Better at separating content from timbre |
| SPIN | Best (newer) | Speaker-invariant, good for realtime |
| SPIN-v2 | Best | Latest, improved separation |

---

## Recommendations for Low-Latency Realtime

### 1. Reduce Block Size (SOLA-Validated)

Current: 120ms blocks + 30ms SOLA = 150ms total output
```python
_BLOCK_48K = 5760  # 120ms
```

**SOLA-Validated Recommendations**:

| Block Size | Total Output | SOLA % | Verdict |
|------------|--------------|--------|---------|
| 120ms | 150ms | 20% | Current - safe baseline |
| 80ms | 110ms | **27%** | **Recommended - best balance** |
| 60ms | 90ms | 33% | Aggressive - may hear artifacts |

```python
# Recommended for quality + realtime
_BLOCK_48K = 3840  # 80ms block + 30ms SOLA = 110ms total
_EXTRA_TIME = 1.5  # 1.5s context (reduced from 2s)
```

**Rationale for 80ms recommendation**:
- SOLA overhead (27%) still acceptable for voice
- 110ms output latency + ~50ms inference = ~160ms end-to-end
- Good balance between latency and quality

### 2. SOLA Buffer Sizing: Fixed vs Block-Linked

**Question**: Should SOLA buffer be fixed or linked to block size?

**Answer**: **Fixed SOLA is better** - supported by research and production implementations:

1. **Academic Research** (Springer - Parameters evaluation of SOLA algorithm):
   - Crossfade window should cover 2-6 pitch periods
   - At voice frequencies (100-200Hz): 1 period = 5-10ms
   - **Optimal: 10-30ms** (current: 20ms ✓)

2. **Production Implementations**:
   - **w-okada voice-changer**: Uses extraConvertSize (4096 samples) to generate SOLA buffer
   - **Current rvc-web**: Fixed 30ms (20ms buffer + 10ms search)
   - Both approaches are valid; fixed is simpler

3. **Why Fixed SOLA Wins**:
   - **Predictable latency**: SOLA overhead always 30ms
   - **Pitch-anchored**: 20ms = ~2-4 pitch cycles @ 100-200Hz
   - **Less complexity**: Fewer parameters to tune
   - **No quality gain** from block-linking: Research shows crossfade quality depends on pitch periods, not block size

**Conclusion**: Keep SOLA fixed at 30ms (20ms buffer + 10ms search). This is optimal for voice frequencies.

### 3. Optimize FAISS Search

Current index is loaded fully. For realtime:

```python
# In rtrvc.py or realtime worker
index.nprobe = 1  # Only search 1 cluster (fastest)
```

**Note**: Current implementation already sets `nprobe = 1` (`_index_worker.py:68`).

### 4. Use Half-Precision (FP16)

Current: `is_half=False` in RVC initialization

**Recommendation**: Enable FP16 for inference:
```python
rvc = RVC(
    ...
    is_half=True,  # Enable FP16
)
```

**Note**: May cause quality loss on some hardware. Test first.

### 5. Model Quantization

Consider using quantized models:
- **INT8 quantization** can reduce inference time by 30-50%
- Available in some RVC forks

### 6. WebGPU / CUDA Acceleration

Current uses MPS (Apple Silicon) or CPU. For lower latency:
- Use CUDA on NVIDIA GPUs (if available)
- Consider ONNX runtime for cross-platform acceleration

### 7. Reduce Historical Context

Current: 2000ms context
```python
_EXTRA_TIME = 2.0  # 2000ms
```

**Recommendation**: Try reducing:
```python
_EXTRA_TIME = 1.0  # 1000ms - faster but may affect pitch accuracy
```

### 8. Use FCPE Instead of RMVPE

FCPE (Fast Context-base Pitch Estimator) is faster than RMVPE:
```python
# In realtime worker, change:
"rmvpe"  # Current
-> "fcpe"  # Faster alternative
```

---

## Blackhole Integration

For realtime voice conversion with Blackhole:

### Setup

1. **Install Blackhole**: Download from https://github.com/ExistentialAudio/Blackhole
2. **Create Aggregate Device**:
   - Open Audio MIDI Setup
   - Create Multi-Output Device combining:
     - Built-in Output (speakers)
     - Blackhole 2ch (virtual)
3. **In rvc-web Realtime**:
   - Set input device to your microphone
   - Set output device to the Multi-Output Device

### Latency Optimization for Blackhole

| Setting | Recommended | Notes |
|---------|-------------|-------|
| Sample Rate | 48kHz | Match output |
| Buffer Size | 256-512 | Lower = less latency |
| **Block Size** | **80ms** | SOLA-validated optimal |
| Total Output | 110ms | Block + SOLA (30ms fixed) |

**Note**: With 80ms block + 30ms SOLA overhead, total output is 110ms. Combined with inference (~50-80ms on MPS), expect **~160-190ms end-to-end latency**.

---

## Summary of Changes

### High-Impact Changes

1. **FAISS Index**: Upgrade from Flat to PQ+RFlat for faster search
2. **Block Size**: Reduce from 120ms to **80ms** (SOLA-validated: 110ms total output)
3. **FP16 Inference**: Enable `is_half=True` if stable
4. **FCPE**: Consider switching from RMVPE to FCPE for speed

### Medium-Impact Changes

1. **Historical Context**: Reduce from 2s to **1.5s** (test quality)
2. **Embedder**: Consider SPIN for better timbre separation
3. **Early Stopping**: Train until quality plateaus, not fixed epochs

### Quality-Focused Changes

1. **Dataset**: Ensure 10-30 minutes of clean audio
2. **Epoch Testing**: Listen to each saved epoch, don't just use defaults
3. **RMVPE for Inference**: Keep RMVPE for batch, consider Crepe for quality

---

## Recommended Configuration for Best Quality + Realtime

For achieving the best voice quality while maintaining realtime performance with Blackhole:

```python
# In _realtime_worker.py

# Block parameters
_BLOCK_48K = 3840           # 80ms (was 120ms)
_EXTRA_TIME = 1.5           # 1.5s context (was 2s)

# SOLA stays unchanged (30ms fixed overhead)
# Total output: 80ms + 30ms = 110ms

# Inference settings
pitch = 0                   # Pitch shift (semitones)
index_rate = 0.75          # 0.75 for quality balance (0.5-1.0)
protect = 0.5              # 0.5 = normal, lower = less breath

# Keep RMVPE (not FCPE) for quality
# Keep is_half=False for quality (test FP16 first)
```

**Expected Performance**:
- Output latency: ~110ms (vs 150ms current)
- Inference time: ~50-80ms (MPS)
- End-to-end latency: **~160-190ms**
- SOLA overhead: 27% - acceptable for voice

---

## Appendix: Key Files Reference

| File | Purpose |
|------|---------|
| `backend/app/training.py` | Training orchestration |
| `backend/app/_index_worker.py` | FAISS index building |
| `backend/app/realtime.py` | Realtime session manager |
| `backend/app/_realtime_worker.py` | Realtime audio processing |
| `Retrieval-based-Voice-Conversion-WebUI/infer/lib/rtrvc.py` | Core RVC inference |
| `Retrieval-based-Voice-Conversion-WebUI/infer/modules/vc/pipeline.py` | Batch VC pipeline |

---

## References

- [RVC Project Wiki - Training Tips](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Instructions-and-tips-for-RVC-training)
- [RVC Project Wiki - FAISS Tuning](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/faiss-tuning-TIPS)
- [AI Hub - RVC Training Guide](https://docs.aihub.gg/rvc/resources/training/)
- [AI Hub - Inference Settings](https://docs.aihub.gg/rvc/resources/inference-settings/)
- [StreamVC: Real-Time Low-Latency Voice Conversion](https://arxiv.org/html/2401.03078v1)
- [Low-latency Real-time Voice Conversion on CPU](https://arxiv.org/abs/2311.00873)
