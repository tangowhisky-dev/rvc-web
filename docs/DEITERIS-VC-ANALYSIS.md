# deiteris/voice-changer — Analysis for Integration with rvc-web

**Repo:** https://github.com/deiteris/voice-changer  
**Local:** `~/code/voice-changer`  
**Description:** Performance-focused fork of w-okada/voice-changer, **RVC-only**, FastAPI/Socket.IO backend in Python  
**License:** (same as w-okada original, effectively MIT-compatible)

---

## 1. What It Is

This is a fork of w-okada's voice-changer that strips everything except RVC and focuses entirely on making RVC streaming inference fast and reliable. The author explicitly removed other VC algorithms ("This version works only with RVC").

Key improvements over w-okada original:
- **DirectML support** — AMD/Intel GPU on Windows via ONNX Runtime DirectML
- **ONNX export pipeline** — converts PyTorch RVC to ONNX for faster inference on any platform
- **SOLA (Synchronized Overlap-Add)** with tunable `crossFadeOverlapSize`, `extraConvertSize`, `serverReadChunkSize`
- **Formant shift** — real-time formant shifting via decoder resampling (`n_res` in models.py), independent of pitch shift
- **skip_head / return_length** in the RVC model's `infer()` — outputs only the needed portion of the synthesis, reducing wasted computation at chunk boundaries
- **Silence detection** with busy-wait on silence to keep GPU clocks stable (explicit comment about Nvidia clock throttling)
- **safetensors** support in addition to `.pth`
- **torch.jit.script + optimize_for_inference** for PyTorch path (toggleable)
- Clean Python structure — settings via Pydantic, typed throughout, no notebook-style global state

---

## 2. Architecture

```
Audio in (float32 PCM, chunked)
    │
    ▼
VoiceChangerV2.on_request()
    │
    ├──► RVCr2.inference()
    │       │
    │       ├── circular_write(audio_in_16k, convert_buffer)  ← rolling ring buffer
    │       ├── circular_write(audio_in_16k, audio_buffer)    ← for volume measurement
    │       │
    │       └── Pipeline.exec()
    │               ├── PitchExtractor.extract()   (rmvpe/fcpe/crepe, ONNX or PyTorch)
    │               ├── Embedder.extract_features() (ContentVec, ONNX)
    │               ├── FAISS index search (optional)
    │               └── Inferencer.infer()          (PyTorch or ONNX)
    │                       SynthesizerTrnMs768NSFsid.infer(
    │                           skip_head=N,        ← discard leading context
    │                           return_length=M,    ← output only this many frames
    │                           formant_length=K    ← resample decoder output for formant
    │                       )
    │
    └──► VoiceChangerV2.process_audio()
             └── SOLA crossfade to stitch chunk boundary
```

**Transport:** FastAPI REST (`POST /test`) and Socket.IO, both over HTTP/S. Payload: msgpack-encoded `[timestamp, int16_pcm_bytes]`. Response: msgpack `{audio, perf, vol, ping, sendTimestamp}`.

**Default settings:**
- `serverReadChunkSize = 192` → `192 * 128 = 24576 samples` @ 48kHz = **512ms** chunk
- `crossFadeOverlapSize = 0.1s`  
- `extraConvertSize = 0.5s` (context fed to model but not returned)  
- `inputSampleRate = outputSampleRate = 48000`

With a 512ms chunk and 0.5s extra context, the latency floor is ~512ms. Users tune `serverReadChunkSize` down (minimum is where `perf < chunk_duration`).

---

## 3. How It Differs From rvc-web's Current RVC Path

rvc-web has its own RVC training pipeline and runs inference via uvicorn. The deiteris fork is a **different server** that would run alongside or replace the inference layer. Key differences:

| Dimension | rvc-web inference (current) | deiteris voice-changer |
|---|---|---|
| Purpose | Batch + on-demand conversion | Real-time streaming |
| Chunking | None (full utterance) | Ring buffer + SOLA |
| Skip head / return length | Not implemented | Yes — only decodes needed frames |
| Formant shift | Not implemented | Yes — decoder-level resampling |
| ONNX inference | No | Yes (ContentVec, RMVPE, vocoder all ONNX-able) |
| JIT compile | No | Optional torch.jit.script |
| DirectML | No | Yes |
| Transport | HTTP REST (FastAPI) | HTTP REST + Socket.IO, both msgpack |
| Model format | .pth / .index | .pth / .safetensors / .onnx + .index |
| Multi-slot | Via profiles | Yes (slot-based) |
| Training | Full pipeline | Not included (inference only) |

---

## 4. The skip_head / return_length Modification — Worth Understanding

This is the most architecturally interesting change. Standard RVC's `SynthesizerTrnMs768NSFsid.infer()` processes the full padded context and returns the full output — you then discard the context portion. This fork adds:

```python
# In models.py SynthesizerTrnMs768NSFsid.infer():
flow_head = max(skip_head - 24, 0)
dec_head = skip_head - flow_head

z = z[:, :, dec_head : dec_head + return_length]     # slice before decoder
x_mask = x_mask[:, :, dec_head : dec_head + return_length]
nsff0 = nsff0[:, skip_head : skip_head + return_length]
o = self.dec(z * x_mask, nsff0, g=g, n_res=formant_length)
```

The flow network (posterior encoder + normalizing flow) still processes the full buffer for context, but the **decoder (NSF-HiFi-GAN) only runs on `return_length` frames**. The decoder is the most VRAM-intensive part; this halves the wasted compute at chunk boundaries. The `-24` offset keeps a small flow tail to avoid boundary artifacts.

rvc-web does not have this — its batch inference runs the full decoder on the full utterance. If rvc-web adds streaming, this is the pattern to follow.

---

## 5. Integration Feasibility with rvc-web

### Option A: Run as sidecar, talk to it via HTTP/Socket.IO

The server exposes `POST /test` at `http://127.0.0.1:18888/test` (configurable). The protocol is simple:

```python
import msgspec.msgpack as msgpack
import numpy as np, time

# Send: [timestamp_ms, int16_pcm_bytes]
payload = msgpack.encode([round(time.time() * 1000), pcm_int16.tobytes()])
resp = requests.post("http://127.0.0.1:18888/test", data=payload)
result = msgpack.decode(resp.content)
# result["audio"] → int16 PCM bytes
# result["perf"] → [0, inference_ms, 0]
# result["vol"] → RMS volume
```

This is already used by the official web client and Colab notebooks. The server manages its own model slot system, separate from rvc-web's profile system.

**Problem:** Model management is independent. Models are loaded via the voice-changer's own REST API (`/model_slots`, etc.) — not from rvc-web's `data/profiles/` directory structure. You'd need to either bridge the two model registries or replicate model files.

### Option B: Extract the pipeline code, integrate directly

The `Pipeline.py`, `RVCr2.py`, and modified `models.py` are clean Python/PyTorch. They can be lifted into rvc-web's backend with moderate effort. The main additions needed in rvc-web:

1. **Ring buffer logic** (`circular_write` in `common/TorchUtils.py`) — simple circular write
2. **skip_head/return_length slicing** in `models.py` (surgical edit to `SynthesizerTrnMs768NSFsid.infer`)
3. **SOLA crossfade** in the streaming output layer
4. **Silence detection / busy-wait** to keep GPU clocks warm

The ONNX path can be adopted selectively — ONNX ContentVec and ONNX RMVPE are faster on CPU and DirectML.

### Option C: Expose rvc-web's models through the deiteris interface

Run the deiteris server pointing at rvc-web's model files. The slot system loads `.pth` + `.index` files — rvc-web's `G_*.pth` checkpoints and `*.index` files are compatible. You'd need a thin adapter to copy/symlink from `data/profiles/<id>/` into the voice-changer's `model_dir`.

---

## 6. What This Fork Does Not Have

- No training code (inference only)
- No profile management (separate from rvc-web)
- No Beatrice, no RT-VC, no other VC algorithms
- No model training pipeline, no preprocessing, no f0 extraction for training
- Windows-first defaults (VAC virtual cable instructions, DirectML focus) — Linux/macOS work but are secondary
- Model slot UI is separate from rvc-web's profile UI — two separate web UIs would exist

---

## 7. Bottom Line

**This fork is a well-engineered, production-quality RVC streaming inference server.** The code is clean, the modifications are architecturally sound (skip_head/return_length, SOLA, ring buffers), and it's the right implementation to study if rvc-web wants to add real-time streaming.

**For rvc-web specifically:**

- If you want a quick real-time path with minimal engineering: **run it as a sidecar** (Option A). The msgpack REST protocol is trivial to call from rvc-web's backend. Users load their RVC model into the voice-changer slot once; rvc-web handles training and offline conversion as before.

- If you want deeper integration (single UI, shared model management, shared profiles): **Option B** — extract the ring buffer, skip_head, and SOLA patterns and wire them into rvc-web's existing `RVCPipeline`. The modified `models.py` with skip_head support is the key surgical change.

- **Option B is ~2–3 days of engineering work.** The changes to models.py are well-isolated. The ring buffer and SOLA logic are ~100 lines each. The main complexity is the streaming audio I/O path (WebSocket or audio device driver), which rvc-web doesn't currently have at all.

The deiteris fork doesn't solve the fundamental RVC latency floor — ContentVec (HuBERT) still needs ~20ms context window, RMVPE still needs ~15ms, and NSF-HiFi-GAN still upsamples 200 frames at a time. The practical minimum with this approach is ~150–250ms latency, not the 38ms Beatrice achieves. But it reuses RVC models trained with rvc-web's existing pipeline with no retraining, which is a significant practical advantage.
