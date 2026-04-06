# Project Beatrice — Analysis for Real-Time Speech-to-Speech Conversion

**URL:** https://prj-beatrice.com/  
**Trainer:** https://huggingface.co/fierce-cats/beatrice-trainer (MIT License)  
**VST source:** https://github.com/prj-beatrice/beatrice-vst (open source)  
**Current version:** 2.0.0-rc.1 (release candidate as of 2025)  
**Author:** Project Beatrice (single developer, Japan)

---

## 1. What It Is

Beatrice 2 is a **real-time voice conversion** VST plugin and inference library. It is explicitly designed and optimised for low-latency streaming conversion on consumer CPUs, with no GPU required at inference. It is the closest thing to a purpose-built streaming VC engine in the open-source space.

Key headline numbers from the developer (measured externally, not just claimed):
- **~38ms algorithmic latency** in the official VST (38ms is the algorithmic delay from the chunk processing design; external sources measured 50ms end-to-end at the microphone/speaker boundary)
- **RTF < 0.2** on a mid-range Intel laptop CPU (Core i7-1165G7) in single-threaded mode
- **~30MB model size** (entire paraphernalia directory)
- **CPU-only at inference** — no GPU required; inference runs in a single thread

These are not aspirational benchmarks — there are third-party measurements confirming the ~50ms figure, and VCClient runs it on Raspberry Pi 5.

---

## 2. Architecture: Three Causal Modules

Beatrice 2's architecture directly follows the **StreamVC** (Google, ICASSP 2024) scheme. The developer explicitly says: *"大まかな枠組みは Google の StreamVC と同じ"* ("The general framework is the same as Google's StreamVC").

```
Source Audio (streaming input)
    │
    ├──► [1] PhoneExtractor (causal)
    │         Extracts "what is being said" (soft phone units)
    │         Inspired by Soft-VC; uses wav2vec2-style feature extraction
    │         Architecture updated in rc.0: self-attention added, GRU removed
    │         Trained on ReazonSpeech (Japanese) + VocalSet (singing)
    │         Outputs: phone features + optional kNN-VC quantization
    │
    ├──► [2] PitchEstimator (causal)
    │         Extracts F0 from the source (voiced/unvoiced detection removed in rc.0)
    │         Whitened F0 (pitch without timbre leakage) fed to generator
    │         Pitch range: up to ~F6 (rc.0 raised from A5)
    │
    └──► [3] WaveformGenerator (causal, per-speaker fine-tuned)
              Takes phone features + pitch → target speaker's waveform
              Speaker conditioning via cross-attention (added in rc.0)
              Uses: FIRNet-style post-filter + HiFi-GAN/SoundStream discriminators
              Loss: multi-scale mel + D4C aperiodicity + loudness
              Per-speaker fine-tune: ~40 minutes on RTX 4090
```

**All three modules are causal** — they process input chunk by chunk with bounded look-ahead, enabling streaming operation.

The fine-tuning model is per-speaker (like RVC), not zero-shot. The pretrained PhoneExtractor and PitchEstimator are universal; only WaveformGenerator is per-target-speaker.

---

## 3. Integration Surface

This is where Beatrice diverges from being a clean Python library:

### 3.1 The inference library is a closed binary
The official position from the developer:
> *"公式推論ライブラリを利用する場合は、Project Beatrice にお問い合わせください。"*
> ("If you wish to use the official inference library, please contact Project Beatrice.")

The VST plugin, beatrice-client, and VCClient all use a **proprietary binary inference library** that is not openly distributed. The trainer (`beatrice-trainer`) is MIT-licensed Python, but the thing that actually runs inference is a closed native library.

### 3.2 What is open

| Component | Status | License |
|---|---|---|
| beatrice-trainer (training code) | Open, HuggingFace | MIT |
| beatrice-vst (VST plugin C++ wrapper) | Open, GitHub | See repo |
| Trained models (paraphernalia files) | Open, distributed freely | Per-model terms |
| Inference library (the actual NN runtime) | **Closed binary** | Proprietary |
| PhoneExtractor/PitchEstimator training code | Not published ("面倒") | N/A |

### 3.3 VCClient path (practical workaround)
[VCClient](https://github.com/w-okada/voice-changer) by w-okada bundles the Beatrice inference library and provides a local HTTP/WebSocket API. It is the only practical way to use Beatrice on non-Windows platforms (macOS Apple Silicon is specifically mentioned as supported via VCClient's modified library). This is how you'd actually integrate it into rvc-web — as a sidecar process.

### 3.4 WebAssembly / browser
Wasm inference is confirmed working by the developer for Beatrice 1. The smartphone app "Voime" (picon Inc.) uses Beatrice via Wasm. For server-side Python integration, Wasm is not the right path — VCClient is.

---

## 4. Training a New Speaker Model

The trainer is fully open (MIT), documented, and straightforward:

```bash
# Setup
git lfs install
git clone https://huggingface.co/fierce-cats/beatrice-trainer
uv sync --extra cu128

# Train
python3 beatrice_trainer -d <speaker_wavs_dir> -o <output_dir>
```

Requirements:
- VRAM: ~9GB default (can reduce batch size)
- Training time: ~40 minutes on RTX 4090 (your A100 would be similar)
- Data: no transcription needed; 5–15s clips of target voice
- Output: a `paraphernalia_<name>_<step>/` directory loadable by the inference clients

This is meaningfully simpler than RVC training — no preprocessing steps, no f0 extraction pipeline, no separate feature extraction stage. Just point at a directory of wavs and run.

---

## 5. Comparison to RVC for rvc-web's Context

| Dimension | RVC (current) | Beatrice 2 |
|---|---|---|
| Real-time latency | ~150–500ms with VCClient overlap-add | ~38ms algorithmic, 50ms measured |
| RTF (CPU, laptop) | >1.0 (needs GPU) | <0.2 (single thread) |
| RTF (GPU) | ~0.1–0.2 | Not measured (not needed) |
| Model size | ~60–200MB | ~30MB |
| Training data needed | 10–30 min typical | Works with shorter clips |
| Training complexity | Multi-stage (preprocess, f0 extract, features, train) | Single command |
| Training time (A100) | 10–100+ epochs (~hours) | ~40 min on 4090 |
| Sample rate | 32k / 48kHz | 24kHz (trainer output) |
| Quality | High | High (comparable, different character) |
| Zero-shot (no training) | No | No |
| Singing | Limited | Yes (VocalSet in PhoneExtractor training) |
| Platform at inference | Windows/Linux/Mac (via VCClient) | Windows native; Mac/Linux via VCClient |
| Python integration | Direct (PyTorch) | Via VCClient sidecar or closed binary |
| Inference source | Open (PyTorch) | Closed binary |
| Training source | Open (MIT) | Open (MIT) |

---

## 6. The Critical Limitation for rvc-web Integration

**The inference library is proprietary.** This is a real constraint for rvc-web:

- You cannot call Beatrice inference from Python directly like you call RVC's PyTorch model.
- The two practical integration paths are:
  1. **VCClient sidecar** — run VCClient as a local subprocess, communicate over its WebSocket/HTTP API. This is how third-party apps already integrate it. Latency overhead of the IPC is small relative to Beatrice's algorithmic delay.
  2. **Reverse-engineer/reimplement** — given the trainer is MIT and the architecture is described (StreamVC-like, 3 modules), the architecture is reproducible in PyTorch. The developer chose not to publish training code for PhoneExtractor/PitchEstimator but would share if asked. The WaveformGenerator training is fully in the trainer.

The developer has also confirmed that anyone can implement software that uses Beatrice 2 models:
> *"モデルアーキテクチャを公開しているため、Beatrice 2 のモデルで声の変換を行うソフトウェアを誰でも作成できます"*
> ("Since the model architecture is public, anyone can create software that performs voice conversion using Beatrice 2 models.")

So the architecture is documented and models are open. The closed part is specifically the optimised inference runtime (SIMD-optimised C++ library targeting Windows x86).

---

## 7. Bottom Line

**Beatrice 2 is the most compelling real-time VC system in the open-source space for rvc-web's requirements.** It is purpose-built for what rvc-web's real-time path needs:

- 38ms algorithmic latency vs RVC's ~150–500ms
- CPU-only at inference (no GPU required for live VC)
- Simple training pipeline (40 min, single command)
- Per-speaker fine-tuning (same paradigm as rvc-web's existing profile system)
- Proven: deployed in production in apps like Voime, supported in VCClient alongside RVC

The barriers are:
1. **No native Python inference** — requires VCClient sidecar or a PyTorch reimplementation
2. **Windows-first** — Mac inference requires VCClient (which w-okada has adapted for Apple Silicon)
3. **24kHz output** — lower sample rate than RVC's 32/48kHz options (audible difference for music)

**Recommended path if rvc-web wants a real-time streaming mode:** add Beatrice as an alternative inference backend, accessed via VCClient's API, with Beatrice-format models trained using the open `beatrice-trainer`. Training integration is straightforward since the trainer is a simple Python CLI with the same "point at wav files" input pattern as rvc-web's existing pipeline. The 24kHz ceiling is the main quality tradeoff vs RVC's 48kHz for music use cases.
