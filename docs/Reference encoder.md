What a Reference Encoder Is
│
│  A reference encoder is a small neural network — typically 3–6 convolutional layers followed by a GRU — that takes a raw audio waveform or mel spectrogram as
│  input and outputs a single fixed-size vector summarising "how this person speaks": their tempo, energy level, pitch range, and the overall style character of
│  that specific recording.
│
│  It is trained end-to-end alongside the main model. Because it has to compress speech into a compact vector that the synthesis decoder can use to reproduce that
│  style, it is forced to learn what is stylistically relevant and discard content (the words said). After enough training it reliably extracts things like: "this
│  person speaks slowly and quietly with narrow pitch range" versus "this person speaks quickly with large expressive pitch swings."
│
│  At training time the reference encoder receives a different clip of the same speaker as the training sample — same voice, different words. The model learns "the
│  style vector from this clip should make my output sound like this speaker's style."
│
│  At inference time you feed it 5–10 seconds of the target speaker's audio. It returns a style vector. That vector is injected into the synthesis decoder. The
│  output now has the target speaker's habitual tempo and energy character in addition to their voice timbre.
│
│  ────────────────────────────────────────────────────────────────────────────────
│
│  How It Fits Into Each Pipeline
│
│  ### Beatrice 2
│
│  What the current conditioning looks like:
│
│  The ConverterNetwork decoder (converter.py) builds its input tensor x by summing four embeddings:
│
│  ```python
│    x = (
│        embed_phone(phone)                          # content features
│      + embed_quantized_pitch(quantized_pitch)      # pitch bin embedding (sinusoidal)
│      + embed_pitch_features(pitch_features)        # energy + pitch continuous features
│      + embed_speaker(target_speaker_id)            # [256-dim] global speaker bias
│      + embed_formant_shift(formant_shift_indices)  # formant shift bias
│    )
│  ```
│
│  That x (shape [batch, hidden_channels, length]) is fed to Vocoder.forward(x, pitch, speaker_embedding).
│
│  Inside Vocoder, the prenet is a ConvNeXtStack with cross_attention=True and kv_channels=128. It does genuine cross-attention — the x sequence attends over a
│  speaker_embedding key-value matrix of shape [batch, 384, 128]. That speaker embedding comes from key_value_speaker_embedding(target_speaker_id), a large learned
│  lookup table [n_speakers, 384 × 128].
│
│  So Beatrice 2 already has two speaker conditioning pathways:
│  1. embed_speaker — a 256-dim additive bias in the main feature space (cheap, global)
│  2. key_value_speaker_embedding — a 384 × 128 key-value matrix used in the vocoder's cross-attention (expensive, detailed)
│
│  Where the reference encoder slots in:
│
│  The reference encoder produces a vector r (say 256-dim) that captures speaking style. The cleanest integration point is as an additive contribution alongside
│  embed_speaker in the main feature sum:
│
│  ```python
│    # Existing:
│    x = embed_phone(phone) + embed_quantized_pitch(qp) + embed_pitch_features(pf)
│      + embed_speaker(target_speaker_id) + embed_formant_shift(formant_shift_indices)
│
│    # With reference encoder:
│    style_vec = reference_encoder(reference_audio)     # [batch, 256, 1]
│    x = embed_phone(phone) + embed_quantized_pitch(qp) + embed_pitch_features(pf)
│      + embed_speaker(target_speaker_id) + embed_formant_shift(formant_shift_indices)
│      + style_projection(style_vec)                    # additive — no structural change
│  ```
│
│  style_projection is a single Conv1d(256, hidden_channels, 1) that maps the reference encoder output into the same space as everything else. The style vector is
│  broadcast over the length dimension ([batch, hidden_channels, 1] → broadcasts to [batch, hidden_channels, length]).
│
│  There is no structural change to the Vocoder, no new conditioning pathway to thread, no weight reshaping. The decoder sees a slightly different x that now also
│  encodes speaking style. Everything else is identical.
│
│  Reference encoder architecture (concrete):
│
│  ```python
│    class ReferenceEncoder(nn.Module):
│        def __init__(self, out_channels: int = 256):
│            super().__init__()
│            # mel → [batch, 128, T/64] feature pyramid
│            self.convs = nn.Sequential(
│                nn.Conv2d(1,  32, 3, stride=(2,2), padding=1),  nn.ReLU(),
│                nn.Conv2d(32, 32, 3, stride=(2,2), padding=1),  nn.ReLU(),
│                nn.Conv2d(32, 64, 3, stride=(2,2), padding=1),  nn.ReLU(),
│                nn.Conv2d(64, 64, 3, stride=(2,2), padding=1),  nn.ReLU(),
│                nn.Conv2d(64,128, 3, stride=(2,2), padding=1),  nn.ReLU(),
│                nn.Conv2d(128,128,3, stride=(2,2), padding=1),  nn.ReLU(),
│            )
│            # GRU over time, outputs final hidden state
│            self.gru = nn.GRU(128 * (n_mels // 64), 128, batch_first=True)
│            self.proj = nn.Linear(128, out_channels)
│
│        def forward(self, mel: torch.Tensor) -> torch.Tensor:
│            # mel: [batch, 1, n_mels, T]
│            x = self.convs(mel)                      # [batch, 128, n_mels//64, T//64]
│            x = x.flatten(1, 2).permute(0, 2, 1)    # [batch, T//64, 128*n_mels//64]
│            _, h = self.gru(x)                       # h: [1, batch, 128]
│            return self.proj(h.squeeze(0))           # [batch, 256]
│  ```
│
│  Approximately 3M parameters. The GRU sees the entire reference clip and its final hidden state summarises the global style.
│
│  Training procedure change:
│
│  Currently each training batch has (x_audio, target_speaker_id). With reference encoder, it becomes (x_audio, ref_audio, target_speaker_id) where ref_audio is a
│  different randomly sampled clip from the same speaker's training files (not the same clip being trained on — that would let the model shortcut by copying from
│  the reference instead of learning to generate).
│
│  The reference encoder is trained from random initialisation, jointly with the decoder. No pretraining or separate reference encoder training run is needed.
│
│  Training overhead:
│  - One additional mel spectrogram computation per batch sample (the reference clip). The mel is already computed for the training sample; computing a second one
│  is cheap.
│  - One additional forward pass through the ~3M-param reference encoder. Compared to the 30M+ param ConverterNetwork forward pass, this is ~10% overhead.
│  - No change to the loss function — the existing multi-scale mel + D4C aperiodicity + loudness + adversarial losses are unchanged. The model is penalised on
│  reconstruction quality; learning to use the reference encoder is fully emergent.
│
│  Key safeguard:
│  During training, when ref_audio == x_audio (same clip), the reference encoder can cheat — it can encode the target audio directly and the model learns "just
│  copy what the reference told you." This is called reference encoder leakage. It is prevented by always using a different clip as the reference. The model only
│  sees the reference for a different utterance from the same speaker, so it genuinely has to learn speaker style rather than content.
│
│  ────────────────────────────────────────────────────────────────────────────────
│
│  ### RVC
│
│  What the current conditioning looks like:
│
│  SynthesizerTrnMsNSFsid has one speaker conditioning vector g:
│
│  ```python
│    g = self.emb_g(sid).unsqueeze(-1)   # [batch, 256, 1]
│  ```
│
│  This g is passed to three places:
│  1. enc_q (posterior encoder, training only): z, m_q, logs_q = enc_q(y, y_lengths, g=g)
│  2. flow (normalising flow): z_p = flow(z, y_mask, g=g)
│  3. dec (NSF vocoder decoder): o = dec(z * x_mask, pitchf, g=g)
│
│  Inside the decoder (NSFGenerator / Generator), the injection is:
│
│  ```python
│    x = self.conv_pre(x)
│    if g is not None:
│        x = x + self.cond(g)    # self.cond = Conv1d(gin_channels=256, upsample_initial_channel, 1)
│  ```
│
│  A single additive conditioning at the very first layer of the vocoder. Everything downstream sees a g-shifted feature map. The same g is also threaded into the
│  ResidualCouplingBlock layers of the normalising flow, conditioning the bijective transforms.
│
│  Where the reference encoder slots in:
│
│  The g vector is the single point of control. The reference encoder output r (256-dim) is combined with g before injection:
│
│  ```python
│    # Training + inference:
│    g = self.emb_g(sid).unsqueeze(-1)                    # [batch, 256, 1] — original
│    r = self.reference_encoder(ref_mel).unsqueeze(-1)    # [batch, 256, 1] — new
│    g_combined = g + self.style_mix(r)                   # [batch, 256, 1] — additive
│    # then pass g_combined where g was passed before
│  ```
│
│  style_mix is a Conv1d(256, 256, 1) (or even just an identity — the simplest version just adds r directly). The combined g_combined replaces g throughout — in
│  enc_q, flow, and dec. No structural changes to those modules.
│
│  This is the minimum viable approach. Because g is already the universal conditioning signal throughout RVC's architecture and every module already has its cond
│  layer expecting a 256-dim signal, extending g with style information requires zero changes inside any sub-module.
│
│  Training procedure change:
│
│  Currently RVC's train.py feeds (phone, pitch, pitchf, spec, wave, sid) per step. A reference audio mel ref_mel would need to be added to this tuple. The
│  reference mel comes from a randomly sampled different utterance of the same speaker (same profile, different audio file).
│
│  This is already feasible with the current data loading in data_utils.py — the dataset already knows which audio files belong to which speaker. Adding "sample a
│  second clip from the same speaker" is a small change.
│
│  RVC has a real complication here though:
│
│  RVC is typically trained on a single speaker — there is often only one speaker in the profile's training data, and sometimes as few as 5–10 audio files. The
│  reference encoder needs to see within-speaker variability at training time to learn to extract style rather than identity. With very few files, every "different
│  clip" reference will be from the same small pool, and the model may overfit the style space to those specific clips.
│
│  Beatrice 2 is better positioned for reference encoder integration because:
│  1. It uses a larger training set (typically 40–90 min across more files)
│  2. Its architecture already has a dedicated cross-attention speaker embedding pathway that is architecturally ready for a richer conditioning signal
│  3. The VQ quantisation step provides a hard disentanglement of content from style, giving the reference encoder a cleaner signal to encode
│
│  For RVC, the reference encoder is still implementable and useful, but expect it to be more sensitive to training data quantity. Profiles trained on < 30 files
│  may see diminishing returns.
│
│  ────────────────────────────────────────────────────────────────────────────────
│
│  Offline Inference
│
│  Beatrice 2 offline (convert_offline in inference.py):
│
│  ```
│    1. User sends: input_file + profile_id + [optional] reference_audio
│    2. Backend loads the ConverterNetwork (already loaded for the profile)
│    3. If reference_audio provided:
│         ref_mel = compute_mel(reference_audio)
│         style_vec = converter.reference_encoder(ref_mel)   # [1, 256]
│       Else:
│         style_vec = zeros  OR  profile's pre-computed default style vec
│    4. convert_offline() runs block loop, injecting style_vec at every block
│    5. style_vec is constant across all blocks — no per-block recomputation
│  ```
│
│  The style vector is computed once at the start and broadcast to every synthesis block. The offline block loop already processes chunks sequentially; injecting a
│  fixed style_vec additive term costs nothing additional.
│
│  Default style (no reference audio supplied):
│
│  If the user does not supply a reference clip, the reference encoder can still contribute: during training, you compute the average style vector across all of
│  the profile's training audio and save it to the checkpoint alongside the model weights. At inference without a reference clip, this pre-computed "profile mean
│  style" is loaded and used automatically. The user experiences automatic speaking style matching to the target speaker's habitual style with no extra input
│  required.
│
│  API change:
│
│  ```python
│    # /api/offline/convert — new optional parameter
│    reference_audio: UploadFile | None = File(default=None)
│  ```
│
│  If present, the reference encoder runs on it. If absent, the pre-computed profile mean style is used. The rest of the pipeline is identical — the frontend does
│  not need to know whether a reference was supplied; the backend always produces a style vector.
│
│  ────────────────────────────────────────────────────────────────────────────────
│
│  Realtime Inference
│
│  The key insight for realtime:
│
│  The reference encoder processes the reference clip once at session start. The resulting style vector r is a constant — it does not change frame-by-frame during
│  the session. It is injected into every synthesis block identically. There is no per-block overhead after the initial computation.
│
│  This means realtime cost is:
│  - At session start: one reference encoder forward pass (~5ms on CPU for a 5-second clip) — one-time cost
│  - Per block: one broadcast addition of r to x — essentially free
│
│  Beatrice 2 realtime (_run_beatrice2_realtime in _realtime_worker.py):
│
│  ```
│    Session start:
│      1. Reference clip specified in start_session params (or use profile default)
│      2. Compute ref_mel once
│      3. style_vec = engine.reference_encoder(ref_mel)   # cached in engine
│      4. engine.style_vec = style_vec                    # stored for the session lifetime
│
│    Per-block inference loop:
│      1. engine.convert(block, f0_norm_params)
│      2. Inside convert(): injects engine.style_vec into x before vocoder forward pass
│      3. No change to the 200ms block budget — style_vec addition is nanoseconds
│  ```
│
│  The session start API (/api/realtime/start) would accept an optional reference_audio_path or reference_audio_b64 parameter. The worker caches the style vector
│  for the session lifetime. It can be updated mid-session via the update_params WebSocket handler without restarting — just recompute style_vec from a new
│  reference clip and replace it.
│
│  RVC realtime (_realtime_worker.py, _patched_get_f0 section):
│
│  Same pattern. g_combined is computed once from the reference clip before the block loop starts, then used identically to the existing g in every rvc.infer()
│  call. No per-block overhead.
│
│  ────────────────────────────────────────────────────────────────────────────────
│
│  The Full Picture
│
│  ```
│    Training (Beatrice 2):
│      ┌─────────────────────────────────────────────────────┐
│      │ For each batch:                                     │
│      │   x_audio ──► PhoneExtractor ──► phone             │
│      │   x_audio ──► PitchEstimator ──► pitch             │
│      │                                                     │
│      │   ref_audio (different clip, same speaker)          │
│      │       ──► mel spectrogram                           │
│      │       ──► ReferenceEncoder (~3M params) ──► r[256]  │
│      │                                                     │
│      │   x = embed_phone + embed_pitch + embed_speaker     │
│      │     + style_projection(r)   ◄── NEW                 │
│      │   y_hat = Vocoder(x, pitch, kv_speaker_embedding)   │
│      │   loss = mel_loss + D4C_loss + loudness_loss + adv  │
│      └─────────────────────────────────────────────────────┘
│
│    Offline inference:
│      reference clip (5–10s) ──► ReferenceEncoder ──► r     (once)
│      for each block in input file:
│          x = embed_phone + embed_pitch + embed_speaker
│            + style_projection(r)   ◄── constant broadcast
│          y_block = Vocoder(x, pitch, kv_speaker_embedding)
│
│    Realtime inference:
│      session start:
│          reference clip ──► ReferenceEncoder ──► r         (once, ~5ms)
│          engine.style_vec = r
│      per 200ms block:
│          x = embed_phone + embed_pitch + embed_speaker
│            + style_projection(engine.style_vec)             (broadcast, ~0ms)
│          y_block = Vocoder(x, pitch, kv_speaker_embedding)
│      update_params WS message:
│          new_ref ──► ReferenceEncoder ──► engine.style_vec  (can update mid-session)
│  ```
│
│  The weight of style_projection is zero-initialised at the start of fine-tuning, so the model behaves identically to the baseline at the beginning of training.
│  The reference encoder contribution grows gradually as the model learns to use it. This means fine-tuning from an existing Beatrice 2 checkpoint is safe — the
│  baseline behaviour is preserved initially and degrades gracefully back to it if the reference encoder fails.
│
│  ────────────────────────────────────────────────────────────────────────────────
│
│  What It Cannot Fix
│
│  The reference encoder captures global speaking style — the average character of the reference clip. It cannot:
│
│  - Change word-level stress (which words get emphasised in this specific sentence)
│  - Change phrase-level rhythm within an utterance (the pauses between this specific set of words)
│  - Transfer sentence-final intonation shapes (questions rising at the end)
│
│  Those require local prosody conditioning, which is the VAE Prosody Encoder step (Phase 2 in the prosody roadmap). The reference encoder is the first layer: it
│  makes the converted voice feel like the target speaker's habitual style. The later phases make it move like them locally. Both are needed for production
│  quality; the reference encoder is the higher-leverage first step.