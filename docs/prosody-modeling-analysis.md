# Prosody Modeling — Analysis and Roadmap

**Date:** 2026-04-27  
**Status:** Research / Pre-implementation  
**Relates to:** `f0-prior-analysis.md`, `WAY-FORWARD.md`, `AMPHION-VEVO2-ANALYSIS.md`

---

## What Prosody Actually Is

Pitch is one signal. Prosody is the full system of suprasegmental cues that carry meaning above the phoneme level. The distinction matters because you can perfectly match a target speaker's pitch distribution and still produce output that sounds robotic, foreign, or wrong — because the *patterns* of pitch, rhythm, and emphasis haven't been transferred.

The five prosody dimensions:

| Dimension | Signal carrier | What a listener hears |
|---|---|---|
| **Pitch contour** | F0 trajectory | Intonation, questions vs statements, word stress |
| **Rhythm** | Duration + pause patterns | Tempo, natural phrasing, syllable timing |
| **Stress** | Energy + F0 + duration jointly | Word emphasis, sentence focus, contrast |
| **Pauses** | Voiced/unvoiced transitions | Breath groups, thinking pauses, sentence breaks |
| **Speaking style** | Global distribution of all above | Calm vs animated, formal vs casual, deliberate vs fast |

Current rvc-web implements pitch distribution matching (Layers 1–4 of `f0-prior-analysis.md`). The other four dimensions are completely unaddressed. A converted voice therefore sounds like the target's timbre and average pitch with the *source speaker's* rhythm, stress, pause patterns, and speaking energy.

---

## Why the Gap Is Hard

Pitch can be modified post-hoc because it is a separate signal channel in both RVC (NSF vocoder conditioned on F0 directly) and Beatrice 2 (PitchEstimator output patched before synthesis). Rhythm, duration, and stress are not separate channels — they are baked into the content features and the synthesis schedule. Changing them post-hoc would require either:

1. Time-stretching (changes duration but creates artifacts and destroys naturalness), or  
2. A model that was trained to disentangle them from content, or  
3. An external prosody signal injected as conditioning.

Options 2 and 3 are the path. Option 1 is already available (phase vocoder / WSOLA) but sounds mechanical and degrades quality more than it gains. Not worth pursuing.

---

## The Four Approaches

### Approach A: Global Style Tokens (GST)

**What it is:**  
Wang et al. (2018), "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis." A set of learnable embedding vectors (the style tokens) are trained end-to-end alongside the TTS/VC model. During training, an attention mechanism over the reference audio waveform produces a weighted combination of style tokens — a *style embedding* — that the decoder conditions on. At inference, the style embedding is extracted from a reference recording of the target speaker and injected into synthesis.

**What it captures:**  
GST learns to partition the style space into interpretable axes: speaking rate, energy, pitch range, formality. In practice, individual tokens often specialize — one token activates for "excited/fast," another for "calm/slow," another for "high-pitched." The separation is emergent, not supervised.

**Relevance to rvc-web:**

- Both RVC (VITS decoder) and Beatrice 2 (WaveformGenerator) have conditioning injection points. GST would add a second conditioning channel alongside the speaker embedding.
- For VC specifically: extract the style embedding from the *target speaker's training audio* (their habitual style) and inject it at inference. This transfers their speaking energy, tempo, and register without touching content.
- **The critical limitation:** GST is a global (utterance-level) vector. It captures average style for the whole utterance but not *local* prosodic variation — a rising question intonation at the end of a sentence, or a sudden emphasis on one word. For VC this is partly acceptable since the goal is "sounds like target" rather than "faithfully reproduces source prosody."

**Implementation path:**  
Add a reference encoder (3–4 conv layers + GRU, ~2M params) and 10 style token vectors (~40K params) to the RVC decoder. Fine-tune the conditioning injection. The reference encoder is applied to the *target's reference audio* (a short sample, not the full training set). This does not require retraining from scratch — the VITS decoder's existing conditioning structure can be extended with an additive style term.

**Training cost:** Moderate. Requires adding the reference encoder and style tokens to the training graph, and providing reference audio per training sample (which we already have: the profile audio files).

**Inference cost:** Near-zero (one reference audio → one style embedding → cached for the session). No per-frame compute.

---

### Approach B: Variational Autoencoder (VAE) Prosody Encoder

**What it is:**  
An encoder-decoder architecture where the encoder maps a segment of speech to a distribution in a latent prosody space (µ and σ), training proceeds with a KL divergence penalty to keep the latent space structured, and the decoder learns to reconstruct speech conditioned on a sample from that distribution. At inference, prosody is sampled from the *target speaker's* latent distribution.

**What it captures:**  
Unlike GST (which is a soft attention over discrete tokens), a VAE prosody encoder produces a continuous latent code. With sufficient encoder depth, the latent space learns to separate: speaking rate, energy envelope, pitch declination pattern, phrase boundary placement, and local emphasis. The KL term forces the latent space to be smooth — nearby latent codes produce perceptually similar prosody. This means you can interpolate, average, and sample from target-speaker prosody without discrete artifacts.

**Two modes of use for VC:**

1. **Target-style anchoring:** Extract prosody from *target speaker's reference audio* → sample from resulting distribution → inject into synthesis. Same use case as GST but with softer, more interpolatable representation.

2. **Prosody transfer:** Extract prosody from *source audio* → re-encode through target speaker's prosody prior → inject into synthesis. This preserves the *trajectory* of source prosody (questions stay rising, emphasis stays on the right syllable) while constraining it to the target's stylistic range. Requires training the VAE with paired source/target prosody data or a domain adaptation step.

**Relevance to rvc-web:**  
The Beatrice 2 architecture is already VAE-adjacent — the WaveformGenerator uses a conditioning structure that could be extended. For RVC, the VITS posterior encoder is a VAE that currently encodes *waveform*; a separate prosody VAE would encode *suprasegmental features* (energy contour, F0 contour, duration features) into a latent code that conditions the decoder.

**Implementation path:**  
1. Extract frame-level prosody features: log-energy, F0 (normalized), frame-level duration (speaking rate proxy). These are already computed during training.
2. Train a lightweight VAE encoder (3 conv layers + bidirectional GRU + projection, ~5M params) on these features.
3. Inject the latent code `z` into the decoder via FiLM conditioning (scale + shift per layer).
4. At inference: encode the target speaker's reference audio through the VAE → sample `z_tgt` → condition synthesis on it.

**Training cost:** Moderate-to-high. The VAE and decoder conditioning must be trained jointly. A pre-trained base model can be fine-tuned, but the conditioning pathway must be initialized from scratch.

**Inference cost:** Low — one reference audio pass per session, latent cached.

---

### Approach C: Reference Encoder (Tacotron-style)

**What it is:**  
Skerry-Ryan et al. (2018), "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis." A convolutional + recurrent encoder that maps a reference mel spectrogram to a fixed-dimensional prosody vector, then conditions the synthesis decoder on this vector.

**What it captures:**  
The reference encoder is a discriminative model — it compresses the entire prosody of a reference utterance into a single embedding. Unlike GST or VAE, it has no explicit disentanglement mechanism, but with sufficient training it learns to extract style-relevant features and ignore content. Works best when source and target speak similar content (controlled studio conditions) — for VC this is often satisfied since we only care about target style, not prosody transfer.

**Why it's simpler than GST or VAE:**  
No attention over token library, no KL term, no sampling. The encoder is a fixed deterministic function. For VC (target-style-only use case), this is often enough.

**Reference encoder architecture:**

```
reference mel spectrogram [T × 80]
    → Conv2D stack (6 layers, batch-norm, ReLU)
    → [T/64 × 128]
    → GRU (128-dim, unidirectional)
    → final hidden state [128-dim]
    → Linear → [style_dim]  (e.g. 256-dim)
```

The style vector is broadcast across all decoder timesteps as an additive conditioning signal.

**Difference from GST:**  
GST adds the token attention layer on top of the reference encoder. Reference encoder alone is the simpler, lower-capacity version. For fine-grained style transfer (editorial control of individual style dimensions) you need GST; for automatic target-style matching, reference encoder alone is sufficient and much easier to train.

**Practical recommendation:**  
Implement reference encoder first. Add GST token layer later if explicit style interpolation becomes a requirement (e.g. "50% formal, 50% casual").

---

### Approach D: Diffusion Prosody Model

**What it is:**  
A denoising diffusion model trained to sample prosody sequences (F0 contours, duration, energy) conditioned on phoneme or content feature inputs. At inference, the model generates a target-speaker prosody sequence consistent with both the input content and the target speaker's learned prosody distribution.

This is the highest-fidelity approach and is how systems like NaturalSpeech 2, VoiceBox, and VALL-E X achieve state-of-the-art naturalness — not by copying source prosody or extracting a style vector, but by *generating* prosody from a learned prior over target speaker trajectories.

**What it captures:**  
Everything. A well-trained diffusion prosody model implicitly captures phrase-level declination, micro-prosodic stress patterns, speaking tempo, pause placement, and intonation language (the speaker's characteristic pitch shapes for questions, lists, emphasis, hedging). It generates a *new* prosody trajectory that is consistent with the target speaker having said the same content — not a transformation of the source, but a generation.

**The VC pipeline with diffusion prosody:**

```
Input audio
  → Content features (HuBERT / PhoneExtractor)
  → [Diffusion prosody model] → target-speaker prosody (F0 + duration + energy)
  → [Voice decoder] (conditioned on content + prosody + speaker embedding)
  → Output audio
```

**Why it's hard:**

1. **Latency:** Diffusion requires N denoising steps (typically 20–100 for prosody). For a 5-second utterance at 100 steps: 100 forward passes through the prosody model. Even a small prosody model (50M params) adds 200–500ms of latency. Not viable for sub-200ms realtime targets.

2. **Training data:** The prosody model must be trained on the *target speaker's* prosody sequences. For rvc-web's typical use case (40–90 min training audio), this is likely sufficient to learn a speaker prior, but the model needs more careful design than the current training pipeline.

3. **Architecture integration:** The diffusion prosody output (F0 + duration) replaces the source speaker's prosody rather than modifying it. This requires a synthesis model that can accept arbitrary prosody sequences — VITS/RVC's NSF vocoder already accepts F0 directly, but duration control (which determines output length) requires a length regulator that is not currently in the RVC pipeline.

4. **Offline-only initially:** The latency and architecture requirements make this offline-only for the foreseeable future.

**Reduced version (diffusion F0 only, no duration):**  
Skip duration modeling. Train a small diffusion model on F0 sequences only (1D signal, much smaller than full prosody). Replace the post-extraction F0 curve with a sample from the model conditioned on content features and speaker. This is more tractable: 50K–100K steps training, ~20M params, 50ms inference latency for a 5-second utterance. Still offline-only.

---

## What Each Approach Adds

| Dimension | GST | VAE Prosody | Reference Encoder | Diffusion Prosody |
|---|---|---|---|---|
| Global speaking style (tempo, energy, formality) | ✓ (discrete tokens) | ✓ (continuous) | ✓ (single vector) | ✓ (generated) |
| Speaking rate / rhythm | Partial (global) | Partial (global) | Partial | ✓ (duration modeled) |
| Intonation shape (questions, lists, emphasis) | Partial | Partial | ✗ | ✓ (trajectory generated) |
| Pause placement | ✗ | ✗ | ✗ | ✓ (duration modeled) |
| Local word stress | ✗ | ✗ | ✗ | ✓ |
| Realtime viable | ✓ | ✓ | ✓ | ✗ (offline only) |
| Retraining required | Yes (full fine-tune) | Yes (joint training) | Yes (fine-tune) | Yes (new model) |
| Training cost | Medium | Medium-High | Low-Medium | High |
| Implementation complexity | Medium | Medium | Low | High |

---

## What is NOT Buildable Post-hoc

Three prosody dimensions cannot be addressed without retraining:

**1. Duration / rhythm / pauses:**  
Duration is determined by how the content features are decoded into synthesis frames. In VITS (RVC), a duration predictor maps phoneme features to frame counts. The target speaker's duration patterns would require fine-tuning the duration predictor on their speech. In Beatrice 2, duration is implicit in the causal synthesis schedule — it cannot be modified post-hoc.

**2. Local word stress (within-sentence):**  
Stress involves a joint change in F0, energy, and duration at the syllable level. F0 we can modify (we do this already). Energy and duration cannot be selectively changed for individual syllables without a syllable-level alignment — which requires either a forced aligner (e.g. MFA/Montreal Forced Aligner) or a model trained with attention alignment.

**3. Prosody language specificity:**  
A speaker of British English and an American English speaker can have identical average F0 and energy, but use qualitatively different intonation patterns (rising declaratives in British English, different question intonation contours). This is a distributional pattern that can only be captured by a model trained on the specific speaker's data — not by any post-processing transform.

---

## Practical Roadmap for rvc-web

Given the current architecture (RVC + Beatrice 2, single-speaker fine-tuning, 40–90 min training audio), this is the recommended sequence:

### Phase 1: Reference Encoder (buildable now, medium effort)

**Scope:** Add a reference encoder to the Beatrice 2 training pipeline. The WaveformGenerator already has a speaker conditioning pathway; extend it with an additive style vector extracted from a short reference audio clip.

**What it gives:** Automatic style matching to the target speaker's habitual tempo, energy, and register. Users record 5–10 seconds of target speech → the reference encoder extracts a style vector → inference conditions on it. Single vector per inference session.

**Changes required:**
- New reference encoder module (~5M params, conv + GRU)
- Extend WaveformGenerator's conditioning to accept style vector
- Extend training to sample a different training clip as reference (different from the training sample itself)
- Extend inference API to accept a reference audio input

**Estimated training overhead:** +10–15% (reference encoder is applied to one reference clip per batch sample, not the whole batch).

---

### Phase 2: Global Style Tokens (buildable, medium-high effort)

**Scope:** Add a GST layer on top of the reference encoder. Trains a set of ~10 style token embeddings. Enables explicit style dimensions that could later be exposed in the UI ("formal ↔ casual", "slow ↔ fast").

**What it adds over Phase 1:** Interpretable style axes. Ability to blend styles (e.g. "60% calm, 40% expressive"). Ability to set style without a reference audio (directly set token weights). More generalizable across speakers.

**Changes required:** GST attention layer on top of reference encoder. Minor changes to training.

**Dependency:** Phase 1 must be complete (reference encoder already trained and integrated).

---

### Phase 3: VAE Prosody Encoder (research track)

**Scope:** Replace or augment the GST with a VAE encoder that maps energy + F0 + voicing features to a structured latent prosody space. Train with a KL term.

**What it adds over GST:** Smooth, interpolatable prosody latent space. Enables prosody *transfer* (not just style matching) by encoding source prosody and re-conditioning through the target speaker's prior. Enables prosody editing in latent space.

**Dependency:** Phase 1–2 complete. Requires access to paired training examples (same content, different prosody) — difficult to guarantee with in-the-wild training audio. May require a data augmentation strategy (pitch + tempo perturbation as proxy).

---

### Phase 4: Diffusion F0 Prior (research/long-term)

**Scope:** Train a small diffusion model (~20M params) on target-speaker F0 sequences conditioned on content features. At offline inference, replace the post-extraction F0 curve with a sample from this model.

**What it adds:** Generated intonation that follows the target speaker's natural pitch trajectory — not a transformation of source F0 but a new speaker-consistent trajectory. Highest possible quality for pitch alone, short of full prosody generation.

**Dependency:** Content features extracted during RVC preprocessing (HuBERT features) are available as conditioning. Training data is the existing voice samples — same pass as current training. ~100K training steps additional.

**Limitation:** Offline only. Realtime requires further work (consistency models, distillation to 1–4 steps).

---

## What the Full Stack Produces

When all four phases are complete, the conversion pipeline looks like:

```
Input audio
  ├── Content features (HuBERT / PhoneExtractor) ──────────────────┐
  ├── F0 extraction (RMVPE / PitchEstimator)                        │
  │    └── [Diffusion F0 prior] → generated target F0 ─────────────┤
  └── Reference audio (target speaker, 5–10s) ─────────────────────┤
       └── [Reference Encoder + GST] → style vector ───────────────┤
                                                                    ▼
                              [Voice Decoder]
                              conditioned on:
                              - content features
                              - target F0 (generated)
                              - speaker embedding (from training)
                              - style vector (from reference encoder)
                                        │
                                        ▼
                              Output: target speaker's voice,
                              target speaker's pitch character,
                              target speaker's speaking style
```

This is the architecture of production systems like NaturalSpeech 2 and SoundStorm, applied at small scale (single-speaker fine-tuning with 40–90 min data rather than multi-speaker training at scale).

---

## What Existing Research to Draw From

**GST / Reference Encoder:**  
- Skerry-Ryan et al. (2018) — "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis" (ICML). Reference encoder architecture.
- Wang et al. (2018) — "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis" (ICML). GST.
- Li et al. (2022) — "StyleTTS" — reference encoder combined with style diffusion. The StyleTTS2 codebase is open, MIT-licensed, and contains a clean reference encoder + style GAN that could be adapted.

**VAE Prosody:**  
- Zhang et al. (2019) — "Learning Latent Representations for Style Control and Transfer in End-to-end Speech Synthesis" (ICASSP). VAE prosody encoder.
- Akuzawa et al. (2018) — "Expressive Speech Synthesis via Modeling Expressions with Variational Autoencoder." Explicit prosody VAE in TTS.

**Diffusion Prosody:**  
- Shen et al. (2023) — "NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers." Full prosody diffusion pipeline.
- Le et al. (2023) — "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale." Flow-matching over mel with prosody conditioning.
- Vevo2 / Amphion (see `AMPHION-VEVO2-ANALYSIS.md`) — reference implementation of flow-matching VC with content-style tokens. Architecture is public, weights available, but offline-only.

**Speaking Rate / Duration:**  
- Kim et al. (2020) — "Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search." Duration predictor architecture that could be adapted.
- Ren et al. (2021) — "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech." Length regulator + duration predictor that is the standard reference implementation.

---

## Key Design Tensions

**1. Style transfer vs style preservation:**  
For VC, do you want to transfer the *source speaker's prosody* to the target voice (preserve the source's intonation pattern, just in the target's voice) or impose the *target speaker's typical prosody* on whatever the source said? These are different products. The first is useful for performances/impressions. The second is useful for making the output sound consistently like the target. Both are useful; they require different conditioning strategies.

**2. Global vs local:**  
GST and reference encoder produce a single vector per utterance. This matches global speaking style but cannot produce local emphasis or intonation shape changes. For a question rising at the end, the F0 contour itself carries the signal — our existing F0 pipeline handles this. For local word emphasis (louder, longer, higher), local prosody control is required. The diffusion approach is the only one here that handles local prosody naturally.

**3. Controllability vs naturalness:**  
Increasing the capacity and expressiveness of the prosody model (more tokens, higher-dim VAE) improves naturalness but reduces interpretability and controllability. A 10-token GST is easy to interpret; a 256-dim VAE latent is not. For a user-facing product, interpretable dimensions are valuable. Design toward 4–8 interpretable axes rather than maximizing capacity.

**4. Training data requirements:**  
Reference encoder and GST train well with as little as 20 minutes of a single speaker's data — they learn to extract a style summary, not model the full distribution. VAE prosody needs ~30–60 minutes for a stable latent space. Diffusion models need 40–90 minutes minimum and benefit from augmentation. All of these are within reach for the current rvc-web training pipeline (60–90 min target).

---

## Summary

Pitch distribution matching (the `f0-prior-analysis.md` stack) is the right first step and is now implemented. It handles one of the five prosody dimensions well.

The next meaningful step toward premium-quality conversion is **Phase 1: reference encoder**. It is the most impactful, lowest-complexity intervention, requires no new training infrastructure beyond extending the WaveformGenerator conditioning, and produces a perceptible quality jump — the converted voice starts to *move* like the target speaker, not just sound like them.

GST (Phase 2) and VAE prosody (Phase 3) are progressive refinements that add controllability and transfer fidelity respectively. Diffusion prosody (Phase 4) is the ceiling — the architecture used by production systems at scale — and should be deferred until the simpler layers are proven on real training data.

The systems that sound premium (ElevenLabs, Play.ai, Resemble) all implement at minimum a reference encoder with either GST or VAE prosody. That is the implementation gap between rvc-web and production quality, beyond timbre and pitch.
