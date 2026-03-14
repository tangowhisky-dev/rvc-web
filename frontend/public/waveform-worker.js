/**
 * waveform-worker.js — OffscreenCanvas Web Worker for live waveform rendering.
 *
 * Receives messages from the main thread:
 *
 *   { type: 'init', canvases: [OffscreenCanvas, OffscreenCanvas] }
 *     — transfers two OffscreenCanvas objects for input and output waveforms.
 *
 *   { type: 'waveform_in', samples: '<base64 float32>', sr: 48000 }
 *     — raw float32 samples for the input (microphone) channel.
 *
 *   { type: 'waveform_out', samples: '<base64 float32>', sr: 48000 }
 *     — raw float32 samples for the output (converted) channel.
 *
 *   { type: 'done' }
 *     — session stopped; clears both canvases.
 *
 * The worker decodes the base64-encoded float32 byte array, then draws a
 * waveform polyline onto the appropriate OffscreenCanvas. Drawing is cheap
 * enough to do synchronously at ~20fps — no requestAnimationFrame needed.
 */

'use strict';

let canvasIn = null;
let canvasOut = null;
let ctxIn = null;
let ctxOut = null;

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

self.onmessage = function (e) {
  const msg = e.data;
  if (!msg || !msg.type) return;

  switch (msg.type) {
    case 'init':
      _handleInit(msg);
      break;

    case 'waveform_in':
      _drawWaveform(ctxIn, canvasIn, msg.samples, '#3b82f6');
      break;

    case 'waveform_out':
      _drawWaveform(ctxOut, canvasOut, msg.samples, '#10b981');
      break;

    case 'done':
      _clearCanvas(ctxIn, canvasIn);
      _clearCanvas(ctxOut, canvasOut);
      break;

    default:
      // Ignore unknown message types (e.g. 'error' from WS)
      break;
  }
};

// ---------------------------------------------------------------------------
// Init: store OffscreenCanvas references
// ---------------------------------------------------------------------------

function _handleInit(msg) {
  if (!msg.canvases || msg.canvases.length < 2) {
    console.error('[waveform-worker] init missing canvases array');
    return;
  }

  canvasIn = msg.canvases[0];
  canvasOut = msg.canvases[1];

  try {
    ctxIn = canvasIn.getContext('2d');
    ctxOut = canvasOut.getContext('2d');
  } catch (err) {
    console.error('[waveform-worker] Failed to get 2d context:', err);
  }

  // Draw flat zero-lines on init so the canvases are not blank
  _drawFlatLine(ctxIn, canvasIn, '#3b82f6');
  _drawFlatLine(ctxOut, canvasOut, '#10b981');
}

// ---------------------------------------------------------------------------
// Draw: decode base64 → Float32Array → polyline
// ---------------------------------------------------------------------------

function _drawWaveform(ctx, canvas, base64Samples, color) {
  if (!ctx || !canvas) return;

  // Decode base64 string → Uint8Array → Float32Array
  let samples;
  try {
    const binaryStr = atob(base64Samples);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }
    samples = new Float32Array(bytes.buffer);
  } catch (err) {
    console.error('[waveform-worker] Failed to decode samples:', err);
    return;
  }

  if (samples.length === 0) return;

  const w = canvas.width;
  const h = canvas.height;
  const mid = h / 2;

  // Clear
  ctx.clearRect(0, 0, w, h);

  // Draw center reference line
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  ctx.moveTo(0, mid);
  ctx.lineTo(w, mid);
  ctx.stroke();

  // Draw waveform
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';

  const step = w / samples.length;

  for (let i = 0; i < samples.length; i++) {
    // Clamp sample to [-1, 1] to avoid spikes
    const s = Math.max(-1, Math.min(1, samples[i]));
    const x = i * step;
    const y = mid + s * mid * 0.88;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _drawFlatLine(ctx, canvas, color) {
  if (!ctx || !canvas) return;
  const mid = canvas.height / 2;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  ctx.strokeStyle = color + '33'; // 20% opacity
  ctx.lineWidth = 1;
  ctx.moveTo(0, mid);
  ctx.lineTo(canvas.width, mid);
  ctx.stroke();
}

function _clearCanvas(ctx, canvas) {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  _drawFlatLine(ctx, canvas, ctx === ctxIn ? '#3b82f6' : '#10b981');
}
