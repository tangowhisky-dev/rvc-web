# NSF-HiFi-GAN vs DDSP Vocoder — Technical Comparison

Benchmarks run on Apple M2/M3 CPU (MPS not used — both measured on CPU for fair comparison).  
RVC config: v2 48k (`upsample_rates=[12,10,2,2]`, `upsample_initial_channel=512`).  
DDSP config: RT-VC defaults (`hidden_dim=256`, `nharmonics=80`, `nbands=257`, `fs=16000`).

---

## 1. What each one is

### NSF-HiFi-GAN (what RVC uses)

"NSF" = Neural Source-Filter. It's HiFi-GAN with an explicit acoustic excitation signal injected at each upsampling stage.

The flow is:
```
Latent z [192-dim, ~100Hz]
    │
    ├──► conv_pre (192→512)
    │        + speaker conditioning g via 1×1 conv
    │
    ├── Upsample stage 1 (ConvTranspose ×12) ─── inject sine excitation via noise_conv
    │       └── MRF: 3 ResBlock1 (kernels 3,7,11 × dilations 1,3,5) summed/averaged
    │
    ├── Upsample stage 2 (ConvTranspose ×10) ─── inject sine excitation via noise_conv
    │       └── MRF: 3 ResBlock1 summed/averaged
    │
    ├── Upsample stage 3 (ConvTranspose ×2)  ─── inject sine excitation via noise_conv
    │       └── MRF: 3 ResBlock1 summed/averaged
    │
    ├── Upsample stage 4 (ConvTranspose ×2)  ─── inject sine excitation (1×1 conv)
    │       └── MRF: 3 ResBlock1 summed/averaged
    │
    └── conv_post (7×1) → tanh → waveform
```

Total upsample factor: 12×10×2×2 = **480** — matches hop_length exactly (480 at 48kHz).

**The sine source** (`SineGenerator`):
- Generates harmonics at f0, 2f0, 3f0... from the pitch signal
- A tiny Linear(1, 1) + Tanh merges them into one excitation signal
- This excitation is injected at each upsampling stage via a strided `Conv1d` that downsamples it to match the current resolution
- **No learned parameters in the sine itself** — it's pure signal processing (2 learnable params total)

The MRF (Multi-Receptive Field Fusion) inside each ResBlock1:
- For each dilation d in [1, 3, 5]: Conv1d(k, dil=d) → LReLU → Conv1d(k, dil=1) → add residual
- Three stacks per upsampling level, summed and averaged

This is a **fully convolutional, autoregressive-free** waveform model. The entire tensor passes through at once — there's no sample-by-sample generation.

### DDSP Vocoder (what RT-VC uses)

DDSP = Differentiable Digital Signal Processing. Instead of learning to hallucinate waveforms, it uses known physics of how voices work and learns only the *parameters* of those physical processes.

The flow is:
```
[f0, loudness, periodicity, EMA] + speaker_emb [128-dim]
    │
    ├──► CausalConvNet (15 → 256 → 256, 3 blocks dilated 1/3/9)
    │        [conditioned via FiLM on speaker_emb]
    │
    ├──► head_amp → harmonic amplitudes (sin + cos) for 80 harmonics
    │        ──► HarmonicOscillator:
    │               • build phase: cumsum(f0 * n / fs) for n=1..80
    │               • upsample amplitudes via Hann-window interpolation
    │               • anti-alias mask harmonics > Nyquist
    │               • output: Σ (aₙ_sin·cₙ_sin·sin(φₙ)) + Σ (aₙ_cos·cₙ_cos·cos(φₙ))
    │
    ├──► head_H → 257-band spectral envelope H
    │        ──► FilteredNoiseGenerator:
    │               • irfft(H) → zero-phase FIR kernel
    │               • generate white noise, filter by time-varying FIR
    │               • overlap-add to produce aperiodic component
    │
    ├──► speech = harmonics + noise
    │
    └──► reverb_mlp(speaker_emb) → 1025-tap IR → fftconvolve → output
```

The synthesis is **interpretable**: 
- Harmonic branch = voiced sounds (vowels, voiced consonants)
- Noise branch = unvoiced sounds (s, f, sh, breath noise)
- Reverb = room/mic characteristics baked into the speaker embedding
- These map directly to how the vocal tract actually works

---

## 2. Parameter counts (measured)

| Component | NSF-HiFi-GAN | DDSP Vocoder |
|---|---|---|
| Core generator | 15,648,256 | — |
| Sine source (NSF add-on) | 2 | — |
| Noise convs (NSF add-on) | 22,272 | — |
| Encoder (causal conv) | — | 5,568,931 |
| Harmonic oscillator | — | 0 (no params) |
| Filtered noise | — | 0 (no params) |
| Reverb MLP | — | 2,235,525 |
| **Total** | **15,670,530** | **7,804,456** |

**DDSP is half the size.** The saving comes from replacing the massive ConvTranspose + MRF upsampling stack (which has to learn how to synthesize audio from scratch) with physics-based oscillators and filters that have zero learnable parameters. The network only needs to learn *what controls to set*, not *how to make sound*.

---

## 3. Inference speed (measured, CPU, Apple M-series)

| Chunk size | NSF-HiFi-GAN | DDSP Vocoder | DDSP speedup |
|---|---|---|---|
| 30ms (NSF) / 15ms (DDSP) | 27.6ms → **RTF 0.92** | 4.9ms → **RTF 0.33** | **5.6×** |
| 100ms | 51.2ms → RTF 0.51 | 6.7ms → RTF 0.07 | **7.6×** |
| 500ms | 126.6ms → RTF 0.25 | 16.3ms → RTF 0.03 | **7.8×** |

RTF = Real-Time Factor. RTF < 1.0 means faster than real time. RTF = 0.92 at 30ms means NSF-HiFi-GAN is *barely* real-time at small chunk sizes on CPU.

Key observations:
- **DDSP is consistently ~7-8× faster** across chunk sizes
- NSF-HiFi-GAN's RTF degrades badly at small chunks (overhead dominates) — it nearly hits 1.0 at 30ms, meaning it would fall behind on a weaker CPU
- DDSP maintains good RTF even at tiny chunks because the physics-based synthesis is O(T) with very low constant factor; the HarmonicOscillator and FilteredNoiseGenerator are essentially free
- NSF-HiFi-GAN output is **48kHz** (higher quality); DDSP output is **16kHz** (lower)

Note: RVC runs GPU inference on A100 where the gap narrows significantly — both are fast enough. The difference matters most on CPU or resource-constrained inference.

---

## 4. How each handles pitch (f0)

This is the most important design difference for voice conversion.

**NSF-HiFi-GAN:**  
The sine source generates a precise harmonic series at the target f0 and injects it as a physical excitation into the upsampling chain. The network learns to use this excitation as a "scaffold" — it shapes timbre around the provided pitch rather than having to hallucinate pitch from the latent features. This is why RVC pitch conversion works as well as it does: pitch is not inferred, it's forced. The network still has to learn the timbral response around that pitch, which takes per-speaker training.

**DDSP Vocoder:**  
Pitch enters as an explicit input (f0 tensor). The HarmonicOscillator constructs 80 harmonics at exact integer multiples of f0 using cumulative-sum phase integration. This is provably correct — there's no learned pitch embedding or pitch following. The network only learns how to weight those harmonics and shape the spectral envelope around them. Phase continuity is tracked explicitly across chunks (no discontinuities at chunk boundaries).

Both handle pitch explicitly rather than implicitly — this is a shared strength that distinguishes them from pure black-box vocoders.

---

## 5. How each handles speaker identity

**NSF-HiFi-GAN:**  
Speaker identity enters as a 256-dim embedding `g` which is projected to the initial channel width and **added to the feature map before upsampling**. It's a single injection point. The speaker embedding is learned via a lookup table (`nn.Embedding`) indexed by speaker ID — this is why RVC requires per-speaker fine-tuning. The embedding captures the full timbral character that the upsampling stack needs to produce.

**DDSP Vocoder:**  
Speaker identity enters as a 128-dim embedding injected via **FiLM (Feature-wise Linear Modulation)** at every layer of the causal conv encoder. FiLM applies a per-channel affine transform: `output = scale(spkr_emb) * features + shift(spkr_emb)`. This is a richer conditioning — the speaker modulates every intermediate representation, not just the input. Additionally, the reverb kernel is entirely derived from the speaker embedding (a separate MLP), so room characteristics and mic response are also speaker-conditioned. This stronger conditioning is what enables zero-shot performance — the speaker encoder only needs to capture a fixed-size embedding from a target clip.

---

## 6. Training differences

**NSF-HiFi-GAN (RVC's training):**
- Loss: mel-spectrogram L1 + multi-period discriminator + multi-scale discriminator (full GAN)
- Requires 5–30 minutes of clean target-speaker audio for fine-tuning
- Base pretrained weights exist (from large multi-speaker training)
- Fine-tuning takes ~1000–5000 epochs to converge for a new speaker

**DDSP Vocoder (RT-VC's training):**
- Loss: multi-scale spectral loss + multi-resolution adversarial loss (smaller discriminator than HiFi-GAN's MPD+MSD)
- Trained on 555h LibriTTS-R, all 2311 speakers seen jointly — single training run
- No per-speaker fine-tuning at all
- Speaker conditioning generalizes to unseen speakers via the speaker encoder
- **No pretrained weights available publicly**

---

## 7. Audio quality comparison

| Dimension | NSF-HiFi-GAN | DDSP Vocoder |
|---|---|---|
| Sample rate | 32/40/48 kHz | 16 kHz only |
| Naturalness (fine-tuned) | High — approaches ground truth for target speaker | Good — UTMOS 3.81, MOS 3.87 (zero-shot) |
| Speaker similarity | Very high (fine-tuned) | ~76.6% Resemblyzer (zero-shot) |
| Artifacts on unvoiced | Low (discriminators suppressed them) | Low (explicit noise generation models unvoiced correctly) |
| Tonal stability | Very high — sine excitation anchors pitch | Very high — cumulative-sum phase guarantees continuity |
| Noise robustness | Moderate — GAN may hallucinate features from noise | Strong — content features (EMA) degrade gracefully |
| Singing | Good | Works (demonstrated in demo samples) |
| 16kHz vs 48kHz | NSF is higher fidelity above 8kHz | DDSP is bandwidth-limited; loses high-frequency air/sibilance |

The 16kHz ceiling is DDSP's practical quality limit. For phone calls or voice chat (which is often 8–16kHz anyway) this is fine. For recording or music applications, 48kHz NSF-HiFi-GAN is clearly better.

---

## 8. Streaming architecture

**NSF-HiFi-GAN:**  
Not causal. All Conv1d layers use symmetric padding — they look at future context. Streaming requires introducing latency equal to the receptive field (~200ms+) or aggressive windowing with audible boundary artifacts. RVC currently uses overlap-add with a crossfade, which adds latency. There is no `realtime_forward` in the code — chunks are just processed with overlap.

**DDSP Vocoder:**  
Designed from the ground up for streaming. All convolutions are causal (left-padding only). Three explicit streaming states are maintained:
- `prev_phase`: carries harmonic phase across chunk boundaries (no tonal clicks)
- `prev_out` in FilteredNoiseGenerator: overlap-add state for noise continuity
- `prev_out` in Vocoder: reverb tail from previous chunk
- Ring buffers for left context per module

The 32ms lookahead is an unavoidable STFT artifact (not architectural). Otherwise the vocoder itself adds zero latency beyond the chunk size.

---

## 9. Summary table

| Dimension | NSF-HiFi-GAN (RVC) | DDSP Vocoder (RT-VC) |
|---|---|---|
| **Params** | 15.7M | 7.8M |
| **RTF (100ms chunk, CPU)** | 0.51 | 0.07 |
| **RTF (30ms chunk, CPU)** | 0.92 | 0.33 |
| **Sample rate** | 48kHz | 16kHz |
| **Streaming native** | No | Yes |
| **Latency (streaming)** | ~200ms+ | ~47ms (chunk + lookahead) |
| **Per-speaker training** | Required | Not required |
| **Pretrained weights available** | Yes | No |
| **Quality (fine-tuned speaker)** | Excellent | N/A (zero-shot only) |
| **Quality (zero-shot)** | Poor (not designed for it) | Good |
| **Interpretable internals** | No | Yes |
| **Noise robustness** | Moderate | Good |
| **Max output quality** | High (48kHz, GAN-refined) | Moderate (16kHz, spectral loss) |

---

## 10. What this means for rvc-web

**Keep NSF-HiFi-GAN** for the core per-speaker fine-tuning use case. The 48kHz output, GAN-trained quality, and large ecosystem of pretrained weights are genuine advantages. The streaming latency is workable at 200ms for most voice chat applications.

**The DDSP vocoder would be worth integrating as an alternative synthesis backend** specifically if:
1. A low-latency CPU path is needed (e.g. the user doesn't have a GPU and 200ms is too much)
2. A zero-shot "preview" mode is desired — quick listen to a target voice without training

The hybrid path would be: keep RVC's feature extraction pipeline (HuBERT, f0, speaker embedding lookup), replace only the synthesis stage (`self.dec`) with a DDSP vocoder. The DDSP vocoder's input format is compatible — it takes f0, periodicity, loudness, and a content representation. The main adaptation needed is substituting HuBERT soft units for EMA (or training the DDSP vocoder on HuBERT features directly), since EMA labels require the SPARC model to generate.
