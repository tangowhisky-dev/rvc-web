# Realtime Audio I/O Architecture — Client-Side Migration Plan

**Status**: Documented for future implementation  
**Date**: 2026-04-06  
**Author**: Analysis session

---

## Problem with Current Architecture

The current realtime inference UI presents audio input and output devices that are detected on the **backend server**, not the client device. This is architecturally incorrect in a client-server model.

```
Current (wrong):
  Server Mic → sounddevice (server) → RVC pipeline → sounddevice (server) → Server Speakers
  UI just shows server device names — irrelevant to the actual user's machine
```

The user's browser has no access to the server's audio hardware, so device selection in the UI is meaningless unless the app is running locally.

---

## Proposed Architecture

```
Correct (client-side I/O):
  Browser Mic → AudioWorklet capture → WebSocket (PCM chunks) → RVC pipeline (server)
                                                                       ↓
  Browser Speakers ← AudioContext playback ← WebSocket (PCM chunks) ←┘
```

Audio capture and playback happen entirely in the browser using Web Audio API. The server only does signal processing (RVC inference). The WebSocket carries raw PCM float32 frames in both directions.

---

## Latency Budget (Excluding Network)

| Stage | Latency | Notes |
|---|---|---|
| Browser mic capture buffer | 10–20 ms | AudioWorklet at 48kHz; 128-sample chunks accumulated to 200ms block |
| Accumulation buffer (browser) | 85–170 ms | ~10 AudioWorklet callbacks needed to fill one 200ms RVC block |
| Resample 48k→16k (browser) | ~1 ms | Linear interpolation, trivial |
| WebSocket frame encode + send | ~0.5 ms | ArrayBuffer binary frame, no JSON serialisation |
| **Network RTT** | *excluded* | LAN <1ms · WiFi 2–10ms · WAN 20–200ms |
| Server receive + queue | ~0.5 ms | asyncio event loop dequeue |
| **RVC block processing** | **120–200 ms** | **Dominant term** — F0 (RMVPE) + FAISS + vocoder on MPS/CPU |
| WebSocket send back | ~0.5 ms | |
| Browser receive + jitter buffer | 50–100 ms | Needed to absorb processing time variance |
| Browser AudioContext playback | 10–20 ms | Output buffer scheduling |
| **Total (excl. network)** | **~280–510 ms** | |

**The dominant term is RVC inference itself (120–200ms).** All other stages combined add ≤100ms. This is the same order as the current architecture — the architectural fix does not materially worsen latency.

### Comparison with Current Architecture

| | Current (server I/O) | Proposed (client I/O) |
|---|---|---|
| Capture latency | ~0ms (sounddevice DMA) | ~15ms (browser buffer) |
| Network contribution | 0 (local only) | RTT (LAN: <1ms) |
| Accumulation buffer | 0 (sounddevice fills block) | ~128ms |
| RVC processing | 120–200ms | 120–200ms (unchanged) |
| Playback latency | ~0ms (sounddevice DMA) | ~15ms (AudioContext) |
| **End-to-end (local)** | **~200ms** | **~300–400ms** |
| **Works over network** | ❌ No | ✅ Yes |
| **Correct device** | ❌ Server's mic | ✅ User's mic |

The tradeoff is ~100ms additional latency in exchange for correctness and network capability. Acceptable for voice conversion monitoring.

---

## Implementation Scope

### Backend changes

**`backend/app/_realtime_worker.py`**
- Remove all `sounddevice` imports and I/O
- Replace mic capture loop with an `asyncio.Queue` that receives PCM chunks from the WebSocket handler
- Replace speaker output with a send queue that pushes processed PCM chunks back to the WebSocket handler
- Block size and sample rate remain unchanged (200ms blocks, 32k internal SR, 48k I/O SR)

**`backend/app/routers/realtime.py`**
- WebSocket protocol change: binary frames (raw `float32` little-endian PCM at 48kHz) replace the current control-only JSON WebSocket
- Control messages (start/stop/params) remain JSON; audio frames are distinguished by message type or a separate binary channel
- Session start no longer requires `input_device_id` / `output_device_id`

**`backend/app/realtime.py`**
- `RealtimeSession` loses `input_device` / `output_device` fields
- Worker process receives audio via IPC queue from WebSocket handler thread

### Frontend changes

**`frontend/app/realtime/page.tsx`**

1. **Device enumeration** — replace `GET /api/devices` (server sounddevice list) with `navigator.mediaDevices.enumerateDevices()` (browser MediaDevices API). Show mic inputs and speaker outputs from the client machine.

2. **Mic capture** — `AudioWorklet` node on the selected `MediaStream`. Accumulate 128-sample worklet chunks into 200ms blocks (~9600 samples at 48kHz). Send as binary `Float32Array` over the data WebSocket.

3. **Playback** — maintain a small jitter buffer (50–100ms) of received PCM chunks. Schedule each chunk via `AudioBufferSourceNode.start(ctx.currentTime + offset)` for glitch-free output.

4. **WebSocket protocol** — binary frames for audio (ArrayBuffer); JSON frames for control (start/stop/params/status). Distinguish by a 1-byte type prefix or separate sockets.

### Protocol sketch

```
Client → Server (binary, per 200ms block):
  [1 byte: 0x01 = audio_in] [float32 LE samples × 9600]

Server → Client (binary, per processed block):
  [1 byte: 0x02 = audio_out] [float32 LE samples × 9600]

Client → Server (JSON):
  {"type": "start", "params": {...}}
  {"type": "params", ...}
  {"type": "stop"}

Server → Client (JSON):
  {"type": "status", "state": "processing"}
  {"type": "error", "message": "..."}
```

---

## Key Risks

1. **AudioWorklet browser support** — available in all modern browsers (Chrome 66+, Firefox 76+, Safari 14.1+). `ScriptProcessorNode` is deprecated but available as fallback.

2. **getUserMedia permissions** — browser will prompt for mic permission on first use. HTTPS required in production (HTTP works on localhost).

3. **Jitter buffer tuning** — if the server's processing time varies (GPU contention, GC pauses), the jitter buffer size needs tuning. Too small → glitches; too large → extra latency.

4. **AudioContext sample rate** — browsers default to 48kHz; some Android devices use 44.1kHz. Must resample to 48kHz before sending if device SR differs.

5. **Binary WebSocket in Next.js** — standard WebSocket API handles binary frames natively. No library changes needed.

---

## Files to Touch

```
backend/app/_realtime_worker.py     — remove sounddevice I/O, add queue-based I/O
backend/app/routers/realtime.py     — binary WS protocol, remove device params
backend/app/realtime.py             — RealtimeSession cleanup
frontend/app/realtime/page.tsx      — AudioWorklet capture, browser device enum, PCM playback
frontend/app/realtime/AudioCapture.ts   (new) — AudioWorklet processor module
frontend/app/realtime/AudioPlayer.ts    (new) — jitter buffer + scheduled playback
```

---

## Reference

- Web Audio API AudioWorklet: https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet
- MediaDevices.getUserMedia: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- Current block params: `_BLOCK_48K=9600` (200ms), `_SOLA_BUF_MS_DEFAULT=20ms`, gate threshold −55dB
- Current RVC processing time on MPS: ~120–180ms per 200ms block (measured)
