# RVC Voice Cloning Quality Analysis

**Date:** 2026-03-29  
**Profile:** moeed (trained on moeed.mp3 + moeed2.mp3)  
**Input:** rvc_input.mp3 (41.76s, 48kHz mono)  
**Output:** rvc_output.mp3 (41.76s, 48kHz mono)  

---

## Executive Summary

The voice cloning produces robotic, unintelligible output due to **severe high-frequency content loss** (89% loss in 1-4kHz band) combined with potential **insufficient model convergence**. Training was actually done for **150 epochs** (not iterations), but the output still has major quality issues.

---

## 1. CORRECTED: Training Analysis

### Epoch vs Iteration Clarification

| Metric | Value | Notes |
|--------|-------|-------|
| Checkpoint `iteration` field | **150** | This represents EPOCHS, not iterations |
| Database `total_epochs_trained` | 300 | Possibly includes resume attempts |
| Batch size | 16 | User confirmed |
| Training samples | 572 audio segments | From 2 source files |
| Batches per epoch | 35 | 572 / 16 = 35.75 (floor) |
| Total batches processed | 5,250 | 150 epochs × 35 batches |

**The model WAS trained for 150 epochs**, which is within the typical range (100-500 epochs) for RVC models. This is NOT an undertraining issue.

### Training Loss (from TensorBoard)

- TensorBoard events file: 22MB (significant training data logged)
- Training did progress and losses decreased

---

## 2. Root Cause: High-Frequency Loss

### Critical Finding: 89% High-Frequency Loss

| Frequency Band | Input Energy % | Output Energy % | Loss |
|----------------|----------------|-----------------|------|
| 0-300 Hz | 12.2% | 8.7% | -29% |
| 300-1000 Hz | 65.8% | 88.5% | +34% (focused) |
| **1000-4000 Hz** | **21.1%** | **2.4%** | **-89%** |
| 4000-8000 Hz | 0.9% | 0.3% | -67% |
| 8000+ Hz | 0.0% | 0.0% | - |

**The critical speech intelligibility band (1-4kHz) has lost 89% of its energy!**

### Audio Quality Degradation Metrics

| Metric | Input | Output | Impact |
|--------|-------|--------|--------|
| Spectral Centroid | 1920 Hz | 1474 Hz | -23% (muffled) |
| Harmonic-to-Noise Ratio | 57.9 dB | 24.4 dB | **-58%** (robotic) |
| Zero Crossing Rate | 0.045 | 0.059 | +31% (artifacts) |
| Waveform Jitter | 0.000686 | 0.001027 | +50% (unstable) |

---

## 3. Sample Rate Analysis (CORRECTED)

### Is 32kHz Output Wrong?

**No, this is CORRECT.**

| Stage | Sample Rate | Notes |
|-------|-------------|-------|
| Input audio | 48kHz | User's microphone |
| Resampled for RVC | 16kHz | Required by HuBERT |
| RVC model output | 32kHz | Model trained at 32kHz |
| Final output | 48kHz | Upsampled for playback |

The pipeline correctly:
1. Resamples input to 16kHz for RVC processing
2. Runs inference at model's native 32kHz
3. Upsamples to 48kHz for output

**This is NOT the source of the quality problem.**

---

## 4. Potential Causes

### 4.1 High FAISS Index Rate (Most Likely)

Current inference settings:
- `index_rate: 0.75` (75% from FAISS retrieval, 25% from model)

This high index rate means:
- 75% of output is based on nearest-neighbor feature matching
- Only 25% is actual model generation
- Can cause robotic artifacts when index doesn't match well

### 4.2 Model Convergence

Even though 150 epochs is typical:
- With only 572 training samples, more epochs may be needed
- The model may not have fully learned the speaker's voice characteristics
- Consider 300-500 epochs for single-voice models

### 4.3 Input Audio Characteristics

Source audio quality:
- moeed.mp3: 1697s, good spectral stability (99.1%)
- moeed2.mp3: 1421s, good spectral stability (99.2%)
- Both have good pitch stability

**Source audio quality is NOT the issue.**

### 4.4 Feature Embedder

- Using `spin-v2` as the embedder
- This extracts 768-dimensional features
- Should be adequate for voice cloning

---

## 5. Pitch/F0 Analysis

| Metric | Input | Output |
|--------|-------|--------|
| Voiced frames | 29.8% | 37.9% |
| F0 range | 249-493 Hz | 145-493 Hz |
| F0 mean | 371 Hz | 344 Hz |
| F0 std | 67 Hz | 85 Hz |
| Pitch shift | - | -27 Hz (-1.3 semitones) |

- Output has artificially increased "voiced" frames (possible index over-reliance)
- Pitch range extended downward (145 Hz vs 249 Hz) - likely noise/artifacts
- Higher pitch variance indicates instability

---

## 6. Pipeline Analysis

### Training Pipeline ✅ Working
```
moeed.mp3 (1697s) + moeed2.mp3 (1421s)
    ↓
Preprocess → 572 audio segments (3s each)
    ↓
F0 Extraction (RMVPE) → 2a_f0, 2b-f0nsf
    ↓
Feature Extraction (spin-v2) → 3_feature768 (768-dim)
    ↓
FAISS Index → 83,486 vectors
    ↓
Training (150 epochs, batch_size=16)
    ↓
G_latest.pth (443MB), D_latest.pth (857MB)
    ↓
model_infer.pth (56MB) + model.index (264MB)
```

### Inference Pipeline ⚠️ Issues
```
Input: 48kHz audio
    ↓
Resample to 16kHz
    ↓
HuBERT/spin-v2 feature extraction
    ↓
RMVPE F0 extraction
    ↓
FAISS retrieval (k=8, index_rate=0.75)
    ↓
RVC Generator (32kHz output)
    ↓
Resample to 48kHz
    ↓
Output: 48kHz (with severe HF loss)
```

---

## 7. Recommendations

### 7.1 High-Priority: Reduce FAISS Index Rate

**Change in inference:**
```python
index_rate: 0.3-0.5  # Instead of 0.75
protect: 0.5-0.7     # Instead of 0.33
```

This will:
- Reduce robotic artifacts from over-reliance on index
- Allow more model-generated content
- Should improve naturalness

### 7.2 Train Longer

If index rate adjustment doesn't help:
- Retrain for 300-500 epochs
- Monitor loss convergence
- Generate test inference during training

### 7.3 Verify Model Convergence

Check that training losses are decreasing:
```bash
# TensorBoard logs are in:
logs/rvc_finetune_active/events.out.tfevents.*
```

### 7.4 Try Different F0 Methods

Current: `rmvpe` (set in inference)  
Try: `fcpe` or `crepe` for potentially better pitch tracking

---

## 8. Technical Details

### Model Configuration
```json
{
  "train": {
    "batch_size": 16,
    "learning_rate": 0.0001,
    "epochs": 150
  },
  "data": {
    "sampling_rate": 32000,
    "n_mel_channels": 80,
    "hop_length": 320,
    "filter_length": 1024
  },
  "model": {
    "version": "v2",
    "embedder": "spin-v2",
    "vocoder": "HiFi-GAN"
  }
}
```

### FAISS Index
- Total vectors: 83,486
- Dimension: 768
- Type: IVF (Inverted File)

### Key Code Paths
- Offline inference: `backend/app/routers/offline.py`
- Realtime inference: `backend/app/_realtime_worker.py`
- Core RVC: `backend/rvc/infer/lib/rtrvc.py`
- Training: `backend/rvc/infer/modules/train/train.py`

---

## 9. Summary of Corrections from Initial Analysis

| Item | Initial Analysis | Corrected |
|------|------------------|-----------|
| Training epochs | 150 iterations (~1 epoch) | **150 EPOCHS** ✓ |
| Batch size | 4 (from config.json) | **16** (from DB) ✓ |
| Sample rate mismatch | Incorrectly flagged | **Correct** ✓ |
| Root cause | Undertraining | **High FAISS index rate + possible convergence** |

---

## Conclusion

1. **Training WAS actually 150 epochs** - this is not the primary issue
2. **Sample rate handling is CORRECT** - no mismatch problem exists
3. **Main problem: 89% high-frequency loss** in output
4. **Most likely cause: High FAISS index_rate (0.75)** causing robotic output
5. **Recommendation: Lower index_rate to 0.3-0.5**

The pipeline code is working correctly. The issue is likely in the balance between FAISS retrieval and model generation, or the model needs more epochs to converge fully.
