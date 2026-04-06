Analysis: Beatrice 2 vs StreamVC

 First, framing what's actually wrong with rvc-web's current situation. Then a precise assessment of each option.

 ────────────────────────────────────────────────────────────────────────────────

 What's Actually Wrong With RVC at 400 Epochs

 Before evaluating alternatives, it's worth being precise about why RVC quality has a ceiling. The architecture has three known structural limitations:

 1. HuBERT content leakage. HuBERT layer 9/12 features are not purely phonetic — they carry residual speaker identity and prosody. The model partially converts
 the voice but some source speaker characteristics bleed through. This is architectural, not fixable with more epochs.

 2. VITS posterior encoder quality bottleneck. The pitch and phoneme features are compressed through a normalising flow bottleneck designed for synthesis quality,
 not speaker disentanglement. The flow learns to use pitch contour as a proxy for timbre, which is why pitch shift and voice timbre are partially coupled in RVC —
 you hear source timbre artefacts on sustained notes.

 3. Vocoder synthesis quality vs latency tradeoff. NSF-HiFi-GAN was designed for offline synthesis quality at 44.1kHz. Forcing it into 200ms streaming blocks with
 SOLA stitching introduces phase discontinuities that the vocoder can't resolve because it has no inter-block state for the residual connections. Each block
 starts cold. RefineGAN has the same problem.

 4. Realtime buffer stitching. The RVC realtime pipeline (including rvc-web's) processes independent overlapping blocks and joins them with SOLA. SOLA is a
 signal-level repair, not model-level coherence. The model doesn't know it's in the middle of a sentence — each block is inferred independently from a rolling
 buffer. The quality floor for realtime is structurally lower than offline.

 These aren't training data or epoch count problems. 400 epochs already converges HiFi-GAN's mel reconstruction loss — going to 600 or 800 epochs won't close the
 gap meaningfully. The gap is in the architecture.

 ────────────────────────────────────────────────────────────────────────────────

 Option 1: Project Beatrice 2

 What it actually is: A complete, production-deployed, purpose-built streaming voice conversion system. Not a research paper — it is actively running in
 production VST and mobile apps right now.

 Architecture (from the rc.0 release notes + my prior analysis):

 ```
   PhoneExtractor (causal conv + self-attention, no GRU)
     → soft phone units + kNN-VC-style vector quantisation
     → explicitly blocks content-to-timbre leakage

   PitchEstimator (causal, range up to F6)
     → whitened F0 (pitch without timbre)
     → avoids source-timbre contamination through pitch channel

   WaveformGenerator (cross-attention speaker conditioning, FIRNet post-filter)
     → per-speaker fine-tuned on ~40min training data
     → D4C aperiodicity loss + multi-scale mel + loudness loss
 ```

 Key architectural advantages over RVC:

 1. kNN-VC quantisation on PhoneExtractor output — this is the same trick kNN-VC uses: replace each content frame with its nearest neighbour from the target
 speaker's content space. It's a hard speaker disentanglement step, not just a soft bottleneck. This is why Beatrice 2 rc.0's changelog specifically lists
 improved speaker similarity — the kNN step directly attacks the content leakage problem.
 2. Whitened F0 (from StreamVC) — F0 is normalised to utterance-level mean/std during training, so the decoder can't use pitch contour as a timbre proxy. Pitch
 and timbre are cleanly separated. RVC doesn't do this.
 3. Causal architecture throughout — every convolution is causal, with a designed 38ms algorithmic delay. The model knows it is streaming. The SOLA patching
 problem doesn't exist because the model is trained and designed for chunk-by-chunk operation from the start.
 4. Cross-attention speaker conditioning (added in rc.0) — the speaker embedding is injected via cross-attention at every decoder layer, not just via a single
 global FiLM conditioning. The target speaker's voice characteristics saturate the whole generation process.
 5. Training in rc.0 added formant shift data augmentation — meaning the model generalises across vocal tract lengths, which improves speaker similarity for
 source/target pairs with very different physiology.

 The inference library problem — honest assessment:

 The beatrice.lib binary is closed. beatrice-client (Rust GUI), VCClient, and the official VST all use it under a revocable licence from the developer. However:

 - The developer has explicitly stated: "Since the model architecture is public, anyone can create software that performs voice conversion using Beatrice 2
 models."
 - The beatrice-trainer (MIT) publishes the full WaveformGenerator training code. The trainer output is pure PyTorch checkpoint files in the paraphernalia_*/
 directory.
 - The beatrice-vst source is on GitHub — the C++ wrapper calls into beatricelib/ which is the only closed part.
 - The model weights (PhoneExtractor, PitchEstimator, WaveformGenerator) are all open and loadable in PyTorch.

 So the closed binary is an optimised C++ runtime, not a gated model format. You can reimplement inference in pure PyTorch — it's slower than the SIMD-optimised
 binary, but it works on any platform including Linux CUDA servers. The architecture is published in the trainer source code.

 Training pipeline for rvc-web integration:

 ```bash
   git lfs install
   git clone https://huggingface.co/fierce-cats/beatrice-trainer
   uv sync --extra cu128
   python3 beatrice_trainer -d data/profiles/<id>/audio/ -o data/profiles/<id>/beatrice/
 ```

 That's the entire pipeline. No preprocessing steps, no f0 extraction, no feature caching step separate from training. The trainer handles everything. Output is a
 paraphernalia_*/ directory. Your A100 would finish in ~30 minutes vs 40 minutes on a 4090.

 Quality ceiling: Beatrice 2 is used for singing voice conversion in production apps. The quality is demonstrably higher than RVC for the use cases where content
 leakage and timbre separation matter most. The FIRNet post-filter specifically addresses the vocoder-noise-floor issue rvc-web currently tries to patch with
 post-NR.

 The 24kHz ceiling: This is a real limitation. The WaveformGenerator is trained at 24kHz. For speech conversion it's fine — 24kHz captures all phonetic content
 cleanly. For singing over music, you'd notice the reduced high-frequency detail. RVC at 48kHz is better for music.

 ────────────────────────────────────────────────────────────────────────────────

 Option 2: StreamVC

 What it actually is: A Google ICASSP 2024 research paper, not a deployable system. The open-source repos are:

 - hrnoh24/stream-vc — an unofficial PyTorch reimplementation, marked [WIP], uses Lightning + Hydra boilerplate, batch size 8, trains on audio.yaml dataset. No
 pretrained weights. No inference example.
 - SteamVC/StreamVC_model — a Japanese fork, also no pretrained weights, Colab-oriented training setup.
 - PriyankaLogasubramanian/StreamVc_Project, Vijayardhan/streamVC — student course projects. Not usable.

 What StreamVC the paper actually proposes:

 - 16kHz output (not 24kHz, not 48kHz — 16kHz)
 - SoundStream encoder/decoder (C=64 for content, C=40 for decoder, D=64 embedding)
 - Content encoder: causal SoundStream encoder trained on HuBERT pseudo-labels
 - Speaker encoder: separate SoundStream encoder with learnable pooling (utterance-level global embedding)
 - Whitened F0 from YIN algorithm, 9-value conditioning (3 thresholds × 3 YIN outputs)
 - 60ms architectural latency (2-frame lookahead), 10.8ms compute per 20ms chunk on Pixel 7 via XNNPACK
 - Speaker similarity score: 80.34% after VCTK fine-tune (second best in their evaluation)
 - Zero-shot (any-to-any): no per-speaker fine-tuning required

 The zero-shot distinction is key. StreamVC uses a global speaker embedding from a reference clip — no model retraining needed per speaker. This is
 architecturally different from both RVC and Beatrice, which are per-speaker fine-tuned. Zero-shot convenience comes at the cost of quality ceiling: in their own
 evaluation, StreamVC with fine-tuning was second best; without fine-tuning it was weaker on speaker similarity.

 The implementation reality:

 None of the open-source repos have pretrained weights. To use StreamVC you would need to:
 1. Pretrain the content encoder on HuBERT pseudo-labels (~1.3M steps at batch 128 = weeks of training on a single A100)
 2. Pretrain the speaker encoder
 3. Train the decoder/vocoder
 4. Implement streaming inference from scratch (no reference code exists for the streaming path)

 The unofficial implementations (hrnoh24/stream-vc) don't even have the model file completed — model.py returns 404. The config shows batch_size=8, max_epochs=500
 — a research stub that has never been trained to completion and published.

 16kHz output is significantly worse than RVC (32kHz) or Beatrice (24kHz) for voice cloning purposes.

 ────────────────────────────────────────────────────────────────────────────────

 Direct Comparison

 ┌─────────────────────────┬─────────────────────────────────┬─────────────────────────────────────────────┬─────────────────────────────────────────────┐
 │ Dimension               │ RVC (current)                   │ Beatrice 2                                  │ StreamVC                                    │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ Deployment readiness    │ Production                      │ Production                                  │ Research paper only                         │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ Pretrained base models  │ Yes                             │ Yes                                         │ None publicly available                     │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ Per-speaker training    │ Yes, multi-stage                │ Yes, 1 command                              │ Would require full pretraining from scratch │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ Training time (A100)    │ Hours–days                      │ ~30 min                                     │ Weeks (pretrain) + days (fine-tune)         │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ Algorithmic latency     │ ~150–500ms (patched with SOLA)  │ 38ms (designed)                             │ 60ms (paper claim, unverified in practice)  │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ Content leakage         │ Moderate (HuBERT bleeds timbre) │ Low (kNN-VC quantisation)                   │ Low (soft units, no fine-tuning leakage)    │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ Timbre/pitch separation │ Weak                            │ Strong (whitened F0)                        │ Strong (whitened F0, same design)           │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ Output sample rate      │ 32/48kHz                        │ 24kHz                                       │ 16kHz                                       │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ CPU-only inference      │ No (needs GPU for realtime)     │ Yes (RTF < 0.2)                             │ Not implemented                             │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ Inference source        │ Open PyTorch                    │ Closed C++ binary (PyTorch reimplementable) │ Open but incomplete                         │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ Quality ceiling         │ Mid (SOLA stitching artefacts)  │ High (designed for streaming)               │ Unknown (no working implementation)         │
 ├─────────────────────────┼─────────────────────────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┤
 │ Integration path        │ Native                          │ VCClient sidecar or PyTorch reimpl          │ Must build from scratch                     │
 └─────────────────────────┴─────────────────────────────────┴─────────────────────────────────────────────┴─────────────────────────────────────────────┘

 ────────────────────────────────────────────────────────────────────────────────

 Verdict

 StreamVC is not an option in any practical sense. There are no pretrained weights, no complete inference code, and the 16kHz output rate is a step backwards. The
 paper is genuinely influential — Beatrice 2 is essentially "StreamVC done right with per-speaker fine-tuning" — but none of the GitHub implementations are
 usable. You would be looking at months of fundamental research engineering, not integration work.

 Beatrice 2 is the right direction. The quality gap you're seeing at 400 epochs isn't fixable by training longer — it's structural, and Beatrice 2 addresses the
 exact structural problems (content leakage, timbre/pitch coupling, streaming block discontinuities) with specific architectural solutions. It's in production,
 the trainer is MIT and straightforward, and you have an A100 which trains it in ~30 minutes per speaker.

 Recommended path:

 1. Short term (days): Train a Beatrice 2 model for your target speaker using the open beatrice-trainer. Compare quality directly against your best RVC model.
 This is 30 minutes of training and zero engineering.
 2. Medium term (weeks): Add Beatrice as a second inference backend in rvc-web. Two integration options in increasing order of control:
     - VCClient sidecar — run VCClient as a subprocess, communicate over its local WebSocket API. Lowest engineering effort, works today.
     - PyTorch reimplementation — load the paraphernalia_*/ model files directly (they're PyTorch checkpoints), implement the 3-module inference pipeline in
 Python. The architecture is documented in the trainer source. Gives you full control, platform independence, direct integration with rvc-web's audio pipeline.
 3. Keep RVC for offline/music use cases — 32/48kHz output and per-speaker fine-tuning at high epoch counts still win for non-realtime quality and music content
 where 24kHz matters. The two backends serve different use cases.

 The 24kHz ceiling is the only honest argument for keeping RVC as the primary realtime path. If your target use case is speech-to-speech conversion (not music),
 Beatrice 2 will sound noticeably better at 38ms latency than RVC at 200ms, and the quality difference at 400 epochs of RVC vs ~30 minutes of Beatrice training
 will likely favour Beatrice.