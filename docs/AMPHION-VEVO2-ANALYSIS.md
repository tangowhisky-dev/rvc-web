# Amphion / Vevo2 — Analysis for Real-Time Speech-to-Speech Conversion

**Repo:** `~/code/Amphion`  
**Focus:** `models/svc/vevo2/` (Vevo2, ACL/IEEE TASLP 2026)  
**Question:** Can it be used for real-time S2S voice conversion?

---

## 1. What Vevo2 Actually Is

Vevo2 is a **unified generation framework** that handles TTS, singing voice synthesis, voice conversion, singing voice conversion, voice editing, and melody control — all from a single model stack. It is **not** designed for real-time use. It is a high-quality offline generation system built around a multi-stage LLM + diffusion pipeline.

The "VC" task it supports is **zero-shot style-preserved voice conversion**: given a source audio and a timbre reference audio (a few seconds of the target speaker), it converts the source speaker's voice to match the target's timbre. Pretrained weights exist and work out of the box.

---

## 2. Architecture: Five Components in Sequence

```
Source Audio (24kHz)
    │
    ├──► [1] Whisper Medium        → 1024-dim content features @50Hz
    │         (frozen, 307M)
    │
    ├──► [2] Chromagram extractor  → 24-dim pitch/harmony features @50Hz
    │         (DSP, no params)
    │
    ├──► [3] Content-Style Tokenizer (CocoContentStyle)
    │         VQ-VAE: Whisper + Chroma → discrete tokens
    │         vocab=16384, frame_rate=12.5Hz (175 bps)
    │         encoder: VocosBackbone (12-layer ConvNeXt, 384-dim)
    │
    │         [Optional AR path for TTS/editing/style control]
    │         [4a] AR Model (Qwen2.5-0.5B, ~500M)
    │               generates content-style tokens autoregressively
    │               given text + prosody + style reference tokens
    │
    ├──► [4b] Flow-Matching Transformer (DiffLlama, 350M)
    │          content-style tokens → mel spectrogram
    │          16-layer LLaMA-style transformer with adaptive RMSNorm
    │          32 ODE steps (Euler) to solve
    │          input: src tokens + timbre_ref tokens + timbre_ref mel (as prompt)
    │          output: target mel @128 bins, 24kHz, hop=480
    │
    └──► [5] Vocos Vocoder (250M)
              mel spectrogram → 24kHz audio
              VocosBackbone + ISTFT head (non-autoregressive)
```

**Total loaded at inference (FM path only):**  
Whisper-medium (307M) + CocoContentStyle tokenizer (~50M) + FMT (350M) + Vocos (250M) ≈ **~960M parameters**

**With AR path add:**  
Qwen2.5-0.5B AR model (~500M) + Prosody tokenizer (~10M) ≈ **~1.47B parameters total**

---

## 3. The FM (Style-Preserved VC) Path in Detail

This is the relevant path for speech-to-speech conversion. No AR model needed.

```python
# Step 1: Extract Whisper features from source (16kHz input required)
whisper_feats = whisper_medium.embed_audio(mel)  # [1, T, 1024] @50Hz

# Step 2: Extract chromagram from source (24kHz)
chroma = librosa.chroma_stft(...)  # [T, 24] @50Hz

# Step 3: VQ-encode to content-style tokens
codecs = content_style_tokenizer.quantize(whisper_feats, chroma)  # [1, T] @12.5Hz

# Step 4: Optionally pitch-shift source chromagram to match target pitch range
# (get_shifted_steps: compute median f0 ratio between source and target)

# Step 5: VQ-encode TIMBRE reference the same way → timbre_ref_codecs

# Step 6: Concatenate [timbre_ref_codecs | src_codecs] → diffusion input
# Step 7: Extract timbre_ref mel as the "prompt" for the diffusion model

# Step 8: Flow-matching reverse diffusion (32 Euler steps)
for i in range(32):
    flow_pred = diff_llama(xt, t, cond, mask)
    xt = xt + flow_pred * h  # Euler step

# Step 9: Vocos → 24kHz audio
audio = vocos(predict_mel)
```

The key thing: **steps 1–7 must complete for the entire input utterance before step 8 begins**. Step 8 runs 32 full-sequence forward passes through a 350M transformer. Step 8 alone is the dominant cost.

---

## 4. Why It Cannot Do Real-Time

### 4.1 The FMT is not causal

`DiffLlama` inherits from HuggingFace `LlamaModel`. LLaMA attention is bidirectional (no causal mask applied in the NAR diffusion setting) — it attends over the full sequence in both directions. **There is no streaming, chunking, or left-context-only mode.** The entire mel sequence must be materialized at once.

### 4.2 Whisper has a hard 30-second window

`whisper.pad_or_trim(wavs)` is called unconditionally in `extract_whisper_features`. Whisper's encoder processes a fixed 30-second mel spectrogram window. For short utterances this means padding; for long utterances it truncates. Either way, **you cannot feed it a running stream of audio chunks** without significant rearchitecting.

### 4.3 32 ODE steps × full sequence = expensive

`reverse_diffusion` runs a loop of `n_timesteps=32` (default). Each iteration calls the full 350M DiffLlama forward pass on the entire sequence. For a 3-second utterance at hop=480, 24kHz → ~150 mel frames:

- 32 steps × 1 forward of 350M transformer over 150-frame sequence
- On an A100, rough estimate: ~0.8–1.5s per utterance (3s audio)
- RTF ≈ 0.3–0.5 on GPU, **much worse on CPU**

Even at the optimistic end, this is an offline batch processor, not a streaming one.

### 4.4 The chromagram requires the full waveform

`librosa.chroma_stft` is computed over the full source audio. Its output feeds the VQ tokenizer. No incremental/causal version exists in the codebase.

### 4.5 Total pipeline latency breakdown (estimate, A100)

| Step | Cost | Notes |
|---|---|---|
| Whisper embed | ~100ms | 30s padded STFT + 24-layer encoder |
| Chromagram | ~10ms | CPU, DSP |
| VQ tokenize (CocoContentStyle) | ~30ms | 12-layer ConvNeXt encoder |
| FMT reverse diffusion (32 steps) | ~800–1500ms | Dominant cost |
| Vocos decode | ~50ms | Non-autoregressive, fast |
| **Total** | **~1–2s** | **For 3–5s of input audio** |

This means even on an A100, Vevo2 FM runs at roughly **RTF 0.3–0.5** — fast enough for offline batch conversion, nowhere near real-time.

### 4.6 The AR path is even slower

`ar_model.generate()` is autoregressive: it generates content-style tokens one-by-one with `max_new_tokens=500`. At 12.5Hz, 500 tokens = 40 seconds of audio. Token-by-token generation on a 500M LLM adds several seconds of latency before the FMT even starts. The AR path is for TTS and style editing, not VC — but the point stands.

---

## 5. What Vevo2 Does Well (for rvc-web's context)

| Capability | Vevo2 | RVC |
|---|---|---|
| Zero-shot (no fine-tuning per speaker) | ✅ Yes | ❌ No |
| Pretrained weights available | ✅ Yes (HuggingFace) | ✅ Yes |
| Speech quality | Very high (24kHz, Vocos) | High (32–48kHz, NSF-HiFi-GAN) |
| Singing voice conversion | ✅ Yes, first-class | Limited |
| Style conversion (not just timbre) | ✅ Yes | ❌ No |
| Multilingual | ✅ Yes | Limited |
| Real-time capable | ❌ No | ❌ Marginal (~200ms) |
| Per-speaker quality ceiling | Lower (zero-shot) | Higher (fine-tuned) |
| Training required for new speaker | No | Yes |
| VRAM at inference | ~8GB (FP16) | ~2–4GB |

---

## 6. Could It Be Made Real-Time?

Theoretically, but the engineering cost is substantial:

**What would need to change:**

1. **Replace Whisper with a causal encoder.** Whisper's fixed-window encoder is the first non-causal blocker. Could be replaced with a streaming SSL model (e.g. wav2vec2-conformer with causal attention), but this would require retraining the content-style tokenizer since its encoder was trained on Whisper features.

2. **Make DiffLlama causal.** LLaMA attention would need a causal mask in NAR diffusion, which fundamentally changes the model's behavior. The flow-matching objective expects to denoise across the full sequence jointly — restricting to causal attention would degrade quality significantly, similar to going from bidirectional BERT to unidirectional GPT for a task that needs context.

3. **Reduce ODE steps.** Consistency models or rectified flow with 1–4 steps could reduce the 32-step bottleneck. The codebase supports arbitrary `n_timesteps` but audio quality degrades below ~8 steps for this model size.

4. **Chunk the chromagram.** Straightforward, but the content-style tokenizer's ConvNeXt encoder has non-causal receptive field.

None of these are trivial. They would require retraining significant parts of the pipeline and would likely degrade quality noticeably.

---

## 7. Comparison to Alternatives for Real-Time Use

| System | RTF (CPU) | RTF (GPU) | Zero-Shot | Weights Available | RT Architecture |
|---|---|---|---|---|---|
| **RVC (current)** | ~2–5 | ~0.1–0.2 | No | Yes | No (overlap-add) |
| **RT-VC** | 0.33 | ~0.05 | Yes | **No** | Yes (causal) |
| **Vevo2 FM** | >>1.0 | ~0.3–0.5 | Yes | Yes | **No** |
| **StreamVC** | ~0.5 | ~0.1 | Yes | No | Yes (causal HuBERT) |

Vevo2 sits in a quality tier above the others but is the furthest from real-time.

---

## 8. Bottom Line

**Vevo2 cannot be used for real-time speech-to-speech conversion as-is.** Every stage of the pipeline — Whisper encoding, chromagram extraction, VQ tokenization, and especially the 32-step flow-matching diffusion over a 350M transformer — requires the full utterance upfront and runs in batch mode. There is no chunking, no causal architecture, and no streaming path anywhere in the codebase.

**What it is good for in rvc-web's context:**

- **Batch/offline VC with zero fine-tuning** — useful if you want to add a "quick preview" mode: convert a sample clip to a target voice without training a full RVC profile. Latency of 1–2s per utterance on GPU is acceptable for this.
- **Singing voice conversion** — Vevo2 handles this natively and far better than RVC.
- **Style conversion** (not just timbre matching) — unique capability not available in RVC at all.

**The pretrained weights are available on HuggingFace (`RMSnow/Vevo2`)**, so integration for offline use requires no training. For real-time use, continue with the RVC path and look at RT-VC or StreamVC architectures if latency reduction becomes a priority.
