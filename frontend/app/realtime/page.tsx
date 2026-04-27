'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { TipsPanel } from '../TipsPanel';
import { SettingsGuide } from '../SettingsGuide';
import { ProfilePicker } from '../ProfilePicker';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Device {
  id: number;
  name: string;
  is_input: boolean;
  is_output: boolean;
}

interface Profile {
  id: string;
  name: string;
  status: string;
  total_epochs_trained: number;
  embedder?: string;
  vocoder?: string;
  pipeline?: string;
  best_model_path?: string | null;
  best_epoch?: number | null;
  best_avg_gen_loss?: number | null;
}

interface SessionParams {
  pitch: number;
  index_rate: number;
  protect: number;
  silence_threshold_db: number;
  output_gain: number;
  noise_reduction: boolean;
  noise_reduction_output: boolean;
  sola_crossfade_ms: number;
  // Beatrice 2 params (ignored for RVC profiles)
  pitch_shift_semitones: number;
  formant_shift_semitones: number;
}

type SessionState = 'idle' | 'starting' | 'active' | 'stopping';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
const WS_BASE = (API ?? 'http://localhost:8000').replace('http://', 'ws://').replace('https://', 'wss://');
const DEFAULT_PARAMS: SessionParams = { pitch: 0, index_rate: 0.50, protect: 0.33, silence_threshold_db: -55, output_gain: 1.0, noise_reduction: true, noise_reduction_output: false, sola_crossfade_ms: 20, pitch_shift_semitones: 0, formant_shift_semitones: 0 };

// ---------------------------------------------------------------------------
// Utility: find default device by name fragment
// ---------------------------------------------------------------------------

function findDefault(devices: Device[], fragments: string[]): number | null {
  const lower = fragments.map((f) => f.toLowerCase());
  for (const d of devices) {
    const n = d.name.toLowerCase();
    if (lower.every((f) => n.includes(f))) return d.id;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Param Slider
// ---------------------------------------------------------------------------

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-baseline">
        <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
          {label}
        </span>
        <span className="text-[13px] font-mono text-cyan-300 tabular-nums">
          {value.toFixed(step < 1 ? 2 : 0)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-[3px] appearance-none bg-zinc-700 rounded-full cursor-pointer
                   accent-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Waveform Canvas
// ---------------------------------------------------------------------------

function WaveformCanvas({
  label,
  color,
  canvasRef,
}: {
  label: string;
  color: string;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
          {label}
        </span>
      </div>
      <div className="relative rounded-lg overflow-hidden border border-zinc-800 bg-zinc-950">
        {/* Horizontal center line */}
        <div className="absolute inset-x-0 top-1/2 h-px bg-zinc-800 pointer-events-none" />
        <canvas
          ref={canvasRef}
          width={800}
          height={120}
          className="w-full h-[120px] block"
          style={{ imageRendering: 'pixelated' }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page Component
// ---------------------------------------------------------------------------

export default function RealtimePage() {
  // Devices
  const [inputDevices, setInputDevices] = useState<Device[]>([]);
  const [outputDevices, setOutputDevices] = useState<Device[]>([]);
  const [inputDeviceId, setInputDeviceId] = useState<number | null>(null);
  const [outputDeviceId, setOutputDeviceId] = useState<number | null>(null);
  const [devicesError, setDevicesError] = useState<string | null>(null);

  // Profiles
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [profileId, setProfileId] = useState<string | null>(null);
  const [useBest, setUseBest] = useState(false);

  // Session
  const [sessionState, setSessionState] = useState<SessionState>('idle');

  // Save audio
  const [saveEnabled, setSaveEnabled] = useState(false);
  const [savePath, setSavePath] = useState('');
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [savedPaths, setSavedPaths] = useState<{input?: string; output?: string; duration_s?: number}>({});

  // Resolve default save path on mount
  useEffect(() => {
    // Ask backend for the user's Downloads directory as default
    fetch(`${API}/api/realtime/default-save-dir`)
      .then((r) => r.json())
      .then((d) => setSavePath(d.path + '/rvc_output.wav'))
      .catch(() => setSavePath('~/Documents/audio/rvc_output.wav'));
  }, []);

  // The backend default-save-dir endpoint returns a full absolute path, so savePath
  // is always absolute in the normal flow. expandedSavePath is only needed if the user
  // manually types ~/... (error-fallback case). Keep it as the text-field display value.
  const expandedSavePath = savePath;
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);

  // Params
  const [params, setParams] = useState<SessionParams>(DEFAULT_PARAMS);

  // F0 prior stats for the selected profile (fetched when profile changes)
  const [profileF0Stats, setProfileF0Stats] = useState<{
    mean_f0: number; std_f0: number;
    p5_f0?: number; p25_f0?: number; p50_f0?: number; p75_f0?: number; p95_f0?: number;
    vel_std?: number; voiced_rate?: number; f0_hist?: number[];
  } | null>(null);
  // Whether to apply F0 prior normalization in realtime (requires profile stats)
  const [autoPitchRt, setAutoPitchRt] = useState(false);

  // Canvas key — incremented on each session start to force fresh DOM canvas elements.
  // Once transferControlToOffscreen() is called, the canvas cannot be re-transferred.
  // Incrementing this key causes React to unmount + remount the canvas elements,
  // giving us brand-new nodes that have never been transferred.
  const [canvasKey, setCanvasKey] = useState(0);

  // Canvas refs — NOT transferred (we draw on them directly via the worker messaging)
  const canvasInRef = useRef<HTMLCanvasElement | null>(null);
  const canvasOutRef = useRef<HTMLCanvasElement | null>(null);

  // Track whether the canvas has been transferred to an OffscreenCanvas.
  // After transfer, we must NOT call transferControlToOffscreen again on the same node.
  const canvasTransferredRef = useRef(false);

  // Refs for cleanup
  const workerRef = useRef<Worker | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const paramsRef = useRef<SessionParams>(DEFAULT_PARAMS);
  const stopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Keep paramsRef in sync
  useEffect(() => {
    paramsRef.current = params;
  }, [params]);

  // Keep sessionIdRef in sync
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  // Fetch profile F0 stats whenever profileId changes
  useEffect(() => {
    if (!profileId) { setProfileF0Stats(null); return; }
    fetch(`${API}/api/offline/speaker_f0/${profileId}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => setProfileF0Stats(d))
      .catch(() => setProfileF0Stats(null));
  }, [profileId]);

  // ---------------------------------------------------------------------------
  // Fetch devices on mount
  // ---------------------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    async function loadDevices() {
      try {
        const res = await fetch(`${API}/api/devices`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const all: Device[] = await res.json();

        const inputs = all.filter((d) => d.is_input);
        const outputs = all.filter((d) => d.is_output);

        if (cancelled) return;
        setInputDevices(inputs);
        setOutputDevices(outputs);

        // Auto-select defaults
        const defaultMic = findDefault(inputs, ['macbook', 'microphone']) ??
          findDefault(inputs, ['built-in', 'input']) ??
          inputs[0]?.id ??
          null;
        const defaultOut = findDefault(outputs, ['blackhole', '2ch']) ??
          outputs[0]?.id ??
          null;

        setInputDeviceId(defaultMic);
        setOutputDeviceId(defaultOut);
      } catch (err) {
        if (!cancelled) {
          setDevicesError(
            `Cannot reach backend (${err instanceof Error ? err.message : String(err)}). ` +
              'Start the backend: conda run -n rvc uvicorn backend.app.main:app --reload'
          );
        }
      }
    }

    async function loadProfiles() {
      try {
        const res = await fetch(`${API}/api/profiles`);
        if (!res.ok) return;
        const all: Profile[] = await res.json();
        const trained = all.filter((p) => p.status === 'trained');
        if (!cancelled) {
          setProfiles(trained);
          if (trained.length > 0) setProfileId(trained[0].id);
        }
      } catch (_) {
        // Non-fatal — profiles section will show empty state
      }
    }

    async function checkActiveSession() {
      try {
        const res = await fetch(`${API}/api/realtime/status`);
        if (!res.ok) return;
        const status = await res.json();
        if (cancelled) return;
        if (status.active && status.session_id) {
          // Session already running — restore state and reconnect WS
          setSessionId(status.session_id);
          sessionIdRef.current = status.session_id;
          setSessionState('active');
          // Reopen WS for live waveform feed
          const ws = new WebSocket(`${WS_BASE}/ws/realtime/${status.session_id}`);
          wsRef.current = ws;
          ws.onmessage = (event) => {
            try {
              const msg = JSON.parse(event.data as string);
              if (workerRef.current) workerRef.current.postMessage(msg);
              if (msg.type === 'done') {
                setSessionState('idle');
                setSessionId(null);
                sessionIdRef.current = null;
                cleanup();
              }
            } catch (_) {}
          };
          ws.onerror = () => { cleanup(); setSessionState('idle'); };
          ws.onclose = () => {
            if (sessionIdRef.current !== null) {
              setSessionState('idle');
              setSessionId(null);
              sessionIdRef.current = null;
              cleanup();
            }
          };
        }
      } catch (_) {
        // Non-fatal
      }
    }

    loadDevices();
    loadProfiles();
    checkActiveSession();
    return () => {
      cancelled = true;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------------------------------------------------------------------------
  // Cleanup helper
  // ---------------------------------------------------------------------------

  const cleanup = useCallback(() => {
    if (stopTimeoutRef.current !== null) {
      clearTimeout(stopTimeoutRef.current);
      stopTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }
    // Reset canvas-transferred flag so the next session can re-transfer fresh canvas nodes.
    // The actual canvas remount is triggered by bumping canvasKey in handleStart.
    canvasTransferredRef.current = false;
  }, []);

  // ---------------------------------------------------------------------------
  // Hot-update params during active session
  // ---------------------------------------------------------------------------

  const pushParams = useCallback(
    async (next: SessionParams) => {
      const sid = sessionIdRef.current;
      if (!sid) return;
      try {
        await fetch(`${API}/api/realtime/params`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sid, ...next }),
        });
      } catch (_) {
        // Non-fatal — params will be applied on next block
      }
    },
    []
  );

  const handleParamChange = useCallback(
    (key: keyof SessionParams, value: number | boolean) => {
      const next = { ...paramsRef.current, [key]: value };
      setParams(next);
      if (sessionIdRef.current) {
        pushParams(next);
      }
    },
    [pushParams]
  );

  // ---------------------------------------------------------------------------
  // Stable refs for current state (for polling loop)
  // ---------------------------------------------------------------------------

  const sessionStateRef = useRef<typeof sessionState>('idle');
  useEffect(() => {
    sessionStateRef.current = sessionState;
  }, [sessionState]);

  // ---------------------------------------------------------------------------
  // Poll session status continuously
  // ---------------------------------------------------------------------------

  useEffect(() => {
    // Poll status every 1s to detect transitions
    const interval = setInterval(async () => {
      // Skip polling if idle
      if (sessionStateRef.current === 'idle' || sessionStateRef.current === 'stopping') return;

      try {
        const res = await fetch(`${API}/api/realtime/status`);
        if (!res.ok) return;
        const status = await res.json();

        // Transition: starting → active once backend confirms
        if (status.active && sessionStateRef.current === 'starting') {
          setSessionState('active');
        }
        // Transition: active → idle when session stops
        else if (!status.active && sessionStateRef.current === 'active') {
          setSessionState('idle');
          setSessionId(null);
          sessionIdRef.current = null;
          cleanup();
        }
      } catch (_) {
        // Ignore network errors; keep polling
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [cleanup]); // Only depends on cleanup, not sessionState

  // ---------------------------------------------------------------------------
  // Start session
  // ---------------------------------------------------------------------------

  const handleStart = useCallback(async () => {
    if (inputDeviceId === null || outputDeviceId === null) {
      setSessionError('Select input and output devices first.');
      return;
    }
    if (!profileId) {
      setSessionError('No trained profile available. Train a voice profile first.');
      return;
    }

    setSessionState('starting');
    setSessionError(null);
    setSaveStatus(null);
    setSavedPaths({});

    // Bump canvas key BEFORE the async work so React remounts the canvas elements.
    // This gives us fresh DOM nodes that haven't had transferControlToOffscreen called yet.
    setCanvasKey((k) => k + 1);

    try {
      // savePath is loaded from the backend (full absolute path via /api/realtime/default-save-dir).
      // In the normal flow it's already absolute. Pass it as-is — the Python backend will handle
      // any remaining ~ via os.path.expanduser if the user typed it manually.
      const resolvedSavePath = savePath;

      // POST /api/realtime/start
      const startRes = await fetch(`${API}/api/realtime/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_id: profileId,
          input_device_id: inputDeviceId,
          output_device_id: outputDeviceId,
          ...params,
          use_best: useBest,
          ...(saveEnabled && resolvedSavePath ? { save_path: resolvedSavePath } : {}),
          // F0 prior: send target stats as soft-clip guardrails when autoPitchRt is on.
          // Affine normalization is NOT applied in realtime (no source file CDF available).
          ...(autoPitchRt && profileF0Stats ? {
            f0_norm_params: {
              // Soft-clip bounds
              ...(profileF0Stats.p5_f0 && profileF0Stats.p95_f0 ? {
                p5_tgt: profileF0Stats.p5_f0,
                p95_tgt: profileF0Stats.p95_f0,
              } : {}),
            },
          } : {}),
        }),
      });

      if (startRes.status === 409) {
        // Session already active — reconnect to it
        const statusRes = await fetch(`${API}/api/realtime/status`);
        if (statusRes.ok) {
          const status = await statusRes.json();
          if (status.active && status.session_id) {
            setSessionId(status.session_id);
            sessionIdRef.current = status.session_id;
            setSessionState('active');
            return;
          }
        }
        throw new Error('Session already active — stop it first');
      }

      if (!startRes.ok) {
        const body = await startRes.json().catch(() => ({ detail: startRes.statusText }));
        throw new Error(body.detail ?? `HTTP ${startRes.status}`);
      }

      const { session_id } = await startRes.json();
      setSessionId(session_id);
      sessionIdRef.current = session_id;
      // Don't set to 'active' yet — let polling detect it

      // ---------------------------------------------------------------------------
      // Launch Web Worker
      // ---------------------------------------------------------------------------
      const worker = new Worker('/waveform-worker.js');
      workerRef.current = worker;

      // Wait one microtask tick so React can flush the canvasKey state update
      // and the refs point to the newly-mounted canvas DOM nodes.
      await new Promise<void>((resolve) => setTimeout(resolve, 0));

      const canvasIn = canvasInRef.current;
      const canvasOut = canvasOutRef.current;

      if (canvasIn && canvasOut && !canvasTransferredRef.current) {
        try {
          // Attempt OffscreenCanvas transfer. Guard with canvasTransferredRef so we
          // never call transferControlToOffscreen twice on the same node (InvalidStateError).
          const offscreenIn = (canvasIn as HTMLCanvasElement & {
            transferControlToOffscreen: () => OffscreenCanvas;
          }).transferControlToOffscreen();
          const offscreenOut = (canvasOut as HTMLCanvasElement & {
            transferControlToOffscreen: () => OffscreenCanvas;
          }).transferControlToOffscreen();

          canvasTransferredRef.current = true;

          worker.postMessage(
            { type: 'init', canvases: [offscreenIn, offscreenOut] },
            [offscreenIn as unknown as Transferable, offscreenOut as unknown as Transferable]
          );
        } catch (err) {
          console.warn('[realtime] OffscreenCanvas transfer failed, using fallback:', err);
          // OffscreenCanvas not supported in this context — send dimensions only
          worker.postMessage({
            type: 'init_fallback',
            widthIn: canvasIn.width,
            heightIn: canvasIn.height,
            widthOut: canvasOut.width,
            heightOut: canvasOut.height,
          });
        }
      }

      // ---------------------------------------------------------------------------
      // Open WebSocket
      // ---------------------------------------------------------------------------
      const ws = new WebSocket(`${WS_BASE}/ws/realtime/${session_id}`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data as string);
          worker.postMessage(msg);

          if (msg.type === 'done') {
            setSessionState('idle');
            setSessionId(null);
            sessionIdRef.current = null;
            cleanup();
          } else if (msg.type === 'save_complete') {
            // Two save_complete events arrive: one for output, one for input.
            // Accumulate both so the UI shows both paths simultaneously.
            setSavedPaths(prev => {
              const isInput = msg.path?.includes('rvc_input');
              const updated = isInput
                ? { ...prev, input: msg.path }
                : { ...prev, output: msg.path, duration_s: msg.duration_s };
              const parts = [];
              if (updated.output) parts.push(`Output: ${updated.output}`);
              if (updated.input)  parts.push(`Input:  ${updated.input}`);
              if (parts.length) setSaveStatus(`✓ Saved (${updated.duration_s ?? '?'}s)\n${parts.join('\n')}`);
              return updated;
            });
          } else if (msg.type === 'save_error') {
            setSaveStatus(`✗ Save failed: ${msg.error}`);
          }
        } catch (_) {
          // Malformed message — ignore
        }
      };

      ws.onerror = () => {
        setSessionError('WebSocket error — check backend logs.');
        setSessionState('idle');
        cleanup();
      };

      ws.onclose = () => {
        if (sessionIdRef.current !== null) {
          // Unexpected close during active session
          setSessionState('idle');
          setSessionId(null);
          sessionIdRef.current = null;
          cleanup();
        }
      };
    } catch (err) {
      setSessionError(err instanceof Error ? err.message : String(err));
      setSessionState('idle');
      cleanup();
    }
  }, [inputDeviceId, outputDeviceId, profileId, params, saveEnabled, savePath, cleanup]);

  // ---------------------------------------------------------------------------
  // Stop session
  // ---------------------------------------------------------------------------

  const handleStop = useCallback(async () => {
    const sid = sessionIdRef.current;
    if (!sid) return;

    setSessionState('stopping');

    try {
      await fetch(`${API}/api/realtime/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid }),
      });
    } catch (_) {
      // Best-effort — cleanup regardless
    }

    // Do NOT close the WS or terminate the worker here.
    // The backend worker will finish (including the save thread), the drain thread
    // will push None into the deque, and the WS will receive 'done' — which triggers
    // cleanup + idle transition in the ws.onmessage handler.
    //
    // If the WS is already closed or the backend is unreachable, fall back to force-cleanup.
    const ws = wsRef.current;
    if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
      cleanup();
      setSessionState('idle');
      setSessionId(null);
      sessionIdRef.current = null;
      return;
    }

    // Fallback: if 'done' doesn't arrive within 6s after stop, force-cleanup so
    // the UI never stays stuck in 'stopping'. Covers worker crashes, save-thread
    // hangs, or any future bug that prevents 'stopped' from being emitted.
    stopTimeoutRef.current = setTimeout(() => {
      stopTimeoutRef.current = null;
      cleanup();
      setSessionState('idle');
      setSessionId(null);
      sessionIdRef.current = null;
    }, 6000);
    // ws.onmessage({ type: 'done' }) calls cleanup() which cancels this timeout.
  }, [cleanup]);

  // ---------------------------------------------------------------------------
  // Cleanup on unmount
  // ---------------------------------------------------------------------------

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  // ---------------------------------------------------------------------------
  // Derived UI state
  // ---------------------------------------------------------------------------

  const isActive = sessionState === 'active';
  const isBusy = sessionState === 'starting' || sessionState === 'stopping';
  const canStart = sessionState === 'idle' && inputDeviceId !== null && outputDeviceId !== null && !!profileId;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center">
            <svg
              width="12"
              height="12"
              viewBox="0 0 12 12"
              fill="none"
              className="text-cyan-400"
            >
              <path
                d="M1 6 Q3 2 5 6 Q7 10 9 6 Q10 4 11 6"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                fill="none"
              />
            </svg>
          </div>
          <h1 className="text-sm font-mono font-medium tracking-wide">
            RVC <span className="text-cyan-400">Realtime</span>
          </h1>
        </div>

        {/* Session status badge */}
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full transition-colors ${
              isActive
                ? 'bg-cyan-400 shadow-[0_0_6px_rgba(34,211,238,0.6)] animate-pulse'
                : isBusy
                ? 'bg-amber-400'
                : 'bg-zinc-600'
            }`}
          />
          <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
            {sessionState === 'idle'
              ? 'idle'
              : sessionState === 'starting'
              ? 'starting…'
              : sessionState === 'active'
              ? 'live'
              : 'stopping…'}
          </span>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-6 py-8 flex flex-col gap-8">
        {/* Active session notice */}
        {isActive && (
          <div className="rounded-lg border border-cyan-800/60 bg-cyan-950/30 px-4 py-3 text-[13px] font-mono text-cyan-300 flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse shrink-0" />
            Session live{sessionId ? ` · ${sessionId.slice(0, 12)}…` : ''} — stop when done.
          </div>
        )}

        {/* Error banner */}
        {(devicesError || sessionError) && (
          <div className="rounded-lg border border-red-800/60 bg-red-950/40 px-4 py-3 text-[13px] font-mono text-red-300">
            {devicesError || sessionError}
          </div>
        )}

        {/* Device selectors */}
        <section>
          <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500 mb-4">
            Audio Devices
          </h2>
          <div className="grid grid-cols-2 gap-4">
            {/* Input */}
            <div className="flex flex-col gap-2">
              <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                <span className="text-blue-400">↓</span> Input
              </label>
              <select
                value={inputDeviceId ?? ''}
                disabled={isActive || isBusy}
                onChange={(e) => setInputDeviceId(Number(e.target.value))}
                className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                           font-mono text-zinc-200 focus:outline-none focus:border-cyan-600
                           disabled:opacity-40 disabled:cursor-not-allowed
                           hover:border-zinc-600 transition-colors"
              >
                {inputDevices.length === 0 && (
                  <option value="">Loading devices…</option>
                )}
                {inputDevices.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Output */}
            <div className="flex flex-col gap-2">
              <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                <span className="text-emerald-400">↑</span> Output
              </label>
              <select
                value={outputDeviceId ?? ''}
                disabled={isActive || isBusy}
                onChange={(e) => setOutputDeviceId(Number(e.target.value))}
                className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                           font-mono text-zinc-200 focus:outline-none focus:border-cyan-600
                           disabled:opacity-40 disabled:cursor-not-allowed
                           hover:border-zinc-600 transition-colors"
              >
                {outputDevices.length === 0 && (
                  <option value="">Loading devices…</option>
                )}
                {outputDevices.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Voice Profile */}
            <div className="col-span-2 flex flex-col gap-2">
              <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                <span className="text-cyan-400">◈</span> Voice Profile
              </label>
              <ProfilePicker
                profiles={profiles}
                selectedId={profileId}
                onChange={setProfileId}
                disabled={isActive || isBusy}
                emptyMessage="No trained profiles — train one in the Training tab first"
              />
              {/* Use best variant checkbox */}
              <label className="flex items-center gap-2 cursor-pointer select-none mt-1">
                <input
                  type="checkbox"
                  checked={useBest}
                  disabled={!profiles.find(p => p.id === profileId)?.best_model_path || isActive || isBusy}
                  onChange={(e) => setUseBest(e.target.checked)}
                  className="w-4 h-4 rounded border border-zinc-600 bg-zinc-900 accent-amber-500 focus:ring-2 focus:ring-amber-500/50 disabled:opacity-40"
                />
                <span className="text-[11px] font-mono text-zinc-300">
                  Use best variant
                </span>
              </label>
              {/* F0 prior guardrail — soft-clips pitch to target's speaking range */}
              {profileF0Stats?.p5_f0 && profileF0Stats?.p95_f0 && (
                <label className="flex items-center gap-2 cursor-pointer select-none mt-0.5">
                  <input
                    type="checkbox"
                    checked={autoPitchRt}
                    disabled={isActive || isBusy}
                    onChange={(e) => setAutoPitchRt(e.target.checked)}
                    className="w-4 h-4 rounded border border-zinc-600 bg-zinc-900 accent-cyan-500 focus:ring-2 focus:ring-cyan-500/50 disabled:opacity-40"
                  />
                  <span className="text-[11px] font-mono text-zinc-300">
                    F0 guardrail{' '}
                    <span className="text-zinc-500">
                      [{profileF0Stats.p5_f0.toFixed(0)}–{profileF0Stats.p95_f0.toFixed(0)} Hz]
                    </span>
                  </span>
                </label>
              )}
            </div>
          </div>
        </section>

        {/* Waveform canvases */}
        <section>
          <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500 mb-4">
            Waveform Monitor
          </h2>
          {/* canvasKey forces React to unmount+remount canvas elements on each session start,
              ensuring transferControlToOffscreen can be called on fresh DOM nodes. */}
          <div key={canvasKey} className="flex flex-col gap-3">
            <WaveformCanvas
              label="Input — Mic"
              color="#3b82f6"
              canvasRef={canvasInRef}
            />
            <WaveformCanvas
              label="Output — Converted"
              color="#10b981"
              canvasRef={canvasOutRef}
            />

            {/* Save Audio */}
            <div className="flex flex-col gap-3 pt-3 border-t border-zinc-800">
              <label className={`flex items-center gap-2 cursor-pointer select-none ${isBusy || isActive ? 'opacity-40 cursor-not-allowed' : ''}`}>
                <input
                  type="checkbox"
                  checked={saveEnabled}
                  disabled={isBusy || isActive}
                  onChange={(e) => setSaveEnabled(e.target.checked)}
                  className="w-4 h-4 rounded border border-zinc-600 bg-zinc-900
                             accent-cyan-400 cursor-pointer disabled:cursor-not-allowed"
                />
                <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
                  Save Audio
                </span>
              </label>

              {saveEnabled && (
                <div className="flex flex-col gap-1">
                  <input
                    type="text"
                    value={expandedSavePath}
                    disabled={isBusy || isActive}
                    onChange={(e) => setSavePath(e.target.value)}
                    placeholder="/Users/tango16/Documents/audio/rvc_output.wav"
                    className="w-full bg-zinc-900 border border-zinc-700 rounded-md px-3 py-1.5
                               text-[12px] font-mono text-zinc-200 focus:outline-none
                               focus:border-cyan-600 disabled:opacity-40 disabled:cursor-not-allowed
                               hover:border-zinc-600 transition-colors"
                  />
                  <p className="text-[10px] font-mono text-zinc-600">
                    Full absolute path — audio is written by the backend process
                  </p>
                </div>
              )}

              {saveStatus && (
                <div className={`text-[11px] font-mono ${saveStatus.startsWith('✓') ? 'text-emerald-400' : 'text-red-400'} whitespace-pre-wrap`}>
                  {saveStatus}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Controls: Params + Start/Stop */}
        <section>
          <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500 mb-4">
            Inference Parameters
          </h2>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5 flex flex-col gap-5">
            {(() => {
              const selProfile = profiles.find(p => p.id === profileId);
              const isB2 = selProfile?.pipeline === 'beatrice2';
              return isB2 ? (
                <>
                  {/* Beatrice 2: pitch shift + formant shift */}
                  <div className="grid grid-cols-2 gap-6">
                    <ParamSlider
                      label={`Pitch Shift (${params.pitch_shift_semitones > 0 ? '+' : ''}${params.pitch_shift_semitones} st)`}
                      value={params.pitch_shift_semitones}
                      min={-12}
                      max={12}
                      step={0.5}
                      disabled={false}
                      onChange={(v) => handleParamChange('pitch_shift_semitones', v)}
                    />
                    <ParamSlider
                      label={`Formant Shift (${params.formant_shift_semitones > 0 ? '+' : ''}${params.formant_shift_semitones} st)`}
                      value={params.formant_shift_semitones}
                      min={-3}
                      max={3}
                      step={0.25}
                      disabled={false}
                      onChange={(v) => handleParamChange('formant_shift_semitones', v)}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-6">
                    <ParamSlider
                      label={`Silence Gate (${params.silence_threshold_db} dBFS)`}
                      value={params.silence_threshold_db}
                      min={-70}
                      max={-10}
                      step={1}
                      disabled={false}
                      onChange={(v) => handleParamChange('silence_threshold_db', v)}
                    />
                    <ParamSlider
                      label={`Output Volume (${params.output_gain.toFixed(2)}×)`}
                      value={params.output_gain}
                      min={0.1}
                      max={3.0}
                      step={0.05}
                      disabled={false}
                      onChange={(v) => handleParamChange('output_gain', v)}
                    />
                  </div>
                </>
              ) : (
                <>
                  {/* RVC: pitch + index_rate + protect */}
                  <div className="grid grid-cols-3 gap-6">
                    <ParamSlider
                      label="Pitch"
                      value={params.pitch}
                      min={-12}
                      max={12}
                      step={1}
                      disabled={false}
                      onChange={(v) => handleParamChange('pitch', v)}
                    />
                    <ParamSlider
                      label="Index Rate"
                      value={params.index_rate}
                      min={0}
                      max={1}
                      step={0.05}
                      disabled={false}
                      onChange={(v) => handleParamChange('index_rate', v)}
                    />
                    <ParamSlider
                      label="Protect"
                      value={params.protect}
                      min={0}
                      max={1}
                      step={0.05}
                      disabled={false}
                      onChange={(v) => handleParamChange('protect', v)}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-6">
                    <ParamSlider
                      label={`Silence Gate (${params.silence_threshold_db} dBFS)`}
                      value={params.silence_threshold_db}
                      min={-70}
                      max={-10}
                      step={1}
                      disabled={false}
                      onChange={(v) => handleParamChange('silence_threshold_db', v)}
                    />
                    <ParamSlider
                      label={`Output Volume (${params.output_gain.toFixed(2)}×)`}
                      value={params.output_gain}
                      min={0.1}
                      max={3.0}
                      step={0.05}
                      disabled={false}
                      onChange={(v) => handleParamChange('output_gain', v)}
                    />
                  </div>
                </>
              );
            })()}

            {/* Noise Reduction + SOLA Crossfade — side by side */}
            <div className="flex gap-2 items-stretch">
              {/* Noise Reduction toggle */}
              <div className="flex-1 flex items-center justify-between px-1 py-2 rounded-lg bg-zinc-900/60 border border-zinc-800">
                <div className="flex flex-col gap-0.5">
                  <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">Noise Reduction</span>
                  <span className="text-[11px] text-zinc-500">
                    {params.noise_reduction
                      ? 'RNNoise active — mic, room & fan noise suppressed'
                      : 'Disabled — raw mic signal passed to model'}
                  </span>
                </div>
                <button
                  onClick={() => handleParamChange('noise_reduction', !params.noise_reduction)}
                  className={`ml-3 shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                    params.noise_reduction ? 'bg-cyan-500' : 'bg-zinc-700'
                  }`}
                  aria-label="Toggle noise reduction"
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
                      params.noise_reduction ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {/* Output Noise Reduction toggle */}
              <div className="flex-1 flex items-center justify-between px-1 py-2 rounded-lg bg-zinc-900/60 border border-zinc-800">
                <div className="flex flex-col gap-0.5">
                  <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">Output NR</span>
                  <span className="text-[11px] text-zinc-500">
                    {params.noise_reduction_output
                      ? 'RNNoise on output — vocoder noise floor suppressed (+~10ms)'
                      : 'Disabled — raw model output to speakers'}
                  </span>
                </div>
                <button
                  onClick={() => handleParamChange('noise_reduction_output', !params.noise_reduction_output)}
                  className={`ml-3 shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                    params.noise_reduction_output ? 'bg-cyan-500' : 'bg-zinc-700'
                  }`}
                  aria-label="Toggle output noise reduction"
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
                      params.noise_reduction_output ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {/* SOLA Crossfade slider */}
              <div className="flex-1 flex flex-col gap-1.5 px-1 py-2 rounded-lg bg-zinc-900/60 border border-zinc-800">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">SOLA Crossfade</span>
                  <span className="text-[11px] font-mono text-zinc-300 tabular-nums">
                    {params.sola_crossfade_ms === 0 ? 'off' : `${params.sola_crossfade_ms} ms`}
                  </span>
                </div>
                <input
                  type="range" min={0} max={50} step={10}
                  value={params.sola_crossfade_ms}
                  onChange={(e) => handleParamChange('sola_crossfade_ms', Number(e.target.value))}
                  className="w-full h-[3px] appearance-none bg-zinc-700 rounded-full cursor-pointer accent-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed"
                />
                <span className="text-[10px] text-zinc-500">
                  {params.sola_crossfade_ms === 0
                    ? 'Disabled — raw block boundary, may click'
                    : `${params.sola_crossfade_ms}ms overlap-add window. Longer = smoother joins, more latency`}
                </span>
              </div>
            </div>

            {/* Start / Stop */}
            <div className="flex items-center gap-4 pt-2 border-t border-zinc-800">
              <button
                onClick={isActive ? handleStop : handleStart}
                disabled={isBusy || (!isActive && !canStart)}
                className={`flex-1 py-3 rounded-lg font-mono text-[13px] font-medium
                            tracking-wider uppercase transition-all
                            disabled:opacity-30 disabled:cursor-not-allowed
                            ${
                              isActive
                                ? 'bg-red-900/60 border border-red-700/60 text-red-300 hover:bg-red-800/60'
                                : 'bg-cyan-900/40 border border-cyan-600/40 text-cyan-300 hover:bg-cyan-800/40 hover:border-cyan-500/60'
                            }`}
              >
                {sessionState === 'starting'
                  ? '⟳  Starting…'
                  : sessionState === 'stopping'
                  ? '⟳  Stopping…'
                  : isActive
                  ? '◼  Stop Session'
                  : '▶  Start Session'}
              </button>

              {sessionId && (
                <span className="text-[10px] font-mono text-zinc-600 truncate max-w-[180px]">
                  {sessionId}
                </span>
              )}
            </div>
          </div>
        </section>

        {/* Footer instructions */}
        <footer className="text-[11px] font-mono text-zinc-600 space-y-1 pb-4">
          <p>
            Route BlackHole 2ch to your DAW or app to hear the converted voice.
          </p>
          <p>
            Backend:{' '}
            <code className="text-zinc-500">
              conda run -n rvc uvicorn backend.app.main:app --reload
            </code>
          </p>
        </footer>

        {/* Tips */}
        <TipsPanel tips={[
          {
            icon: '🎛️',
            title: 'Start with these defaults',
            body: 'pitch=0, index_rate=0.50, protect=0.33. Adjust one parameter at a time so you can hear the effect clearly.',
          },
          {
            icon: '🎵',
            title: 'Pitch shift (semitones)',
            body: 'Positive values raise the pitch; negative lower it. For male-to-female conversion try +10 to +12. For female-to-male try −10 to −12. Even multiples of 12 (octave shifts) are the least artefact-prone.',
          },
          {
            icon: '🗂️',
            title: 'Index rate — how much FAISS retrieval blends in',
            body: 'Higher values (→1.0) pull the voice closer to the training speaker identity at the cost of occasional tonal artefacts. Lower values (→0) rely more on the neural synthesis alone.',
          },
          {
            icon: '🔡',
            title: 'Protect — consonant preservation',
            body: 'Consonants (t, s, p, k) are easily mangled by pitch shifting. Values near 0.33 protect them. If speech sounds lisped or "watery" try raising this toward 0.5.',
          },
          {
            icon: '🔕',
            title: 'Silence threshold',
            body: 'Segments below this level (dBFS) are passed through unprocessed to keep the model warm without converting silence. −45 dB is a good starting point; raise to −35 in noisy environments.',
          },
          {
            icon: '⚡',
            title: 'Latency',
            body: 'Block size is fixed at 200 ms. Total round-trip latency ≈ 200 ms block + ~50–150 ms model inference + audio driver overhead. MPS (Apple Silicon) is fastest for this codebase.',
          },
        ]} />

        {/* Parameter reference */}
        <SettingsGuide settings={[
          {
            name: 'Pitch',
            range: '−24 → +24 semitones',
            default: '0',
            badge: 'cyan',
            summary: 'Shifts the pitch of the converted voice up or down in semitones. Does not change the voice identity — only the musical pitch of what the model outputs.',
            details: 'Each semitone is one step on a piano keyboard. 12 semitones = one octave. Octave shifts (±12, ±24) sound the most natural because the harmonic relationships are preserved. Fractional or odd-number shifts can introduce slight tonal artefacts in some voices. This is applied post-synthesis, so it does not affect speaker identity retrieval.',
            raise: 'Converting a male voice to sound female-range (+10 to +12), or when the converted voice sounds too low compared to the original speaker.',
            lower: 'Converting a female voice to sound male-range (−10 to −12), or when the converted voice sounds unnaturally high or thin.',
          },
          {
            name: 'Index Rate',
            range: '0.0 → 1.0',
            default: '0.75',
            badge: 'violet',
            summary: 'Controls how much the FAISS speaker index blends into the output. Higher values pull the voice identity closer to the training speaker; lower values rely more on the neural synthesis alone.',
            details: 'During training, a FAISS index of HuBERT/SPIN feature vectors from the training audio is built. At inference, the input features are retrieved against this index and the nearest-neighbour cluster centroid is blended in. This "anchors" the voice identity to what was heard in training. At 0 the index is bypassed entirely — synthesis depends only on the model weights. At 1 the retrieved features dominate.',
            raise: 'The converted voice doesn\'t sound enough like the target speaker, or the voice identity drifts mid-sentence. More index = stronger identity pull.',
            lower: 'You hear tonal buzzing, metallic artefacts, or over-processing, especially on vowels. The index is over-correcting — dial it back toward 0.5.',
          },
          {
            name: 'Protect',
            range: '0.0 → 0.5',
            default: '0.33',
            badge: 'emerald',
            summary: 'Protects unvoiced consonants (t, s, p, k, sh, f) from being pitch-shifted and over-processed by the model. These sounds have no fundamental pitch, so shifting them degrades clarity.',
            details: 'The model detects segments where the signal is unvoiced (no clear F0) and passes them through with reduced voice conversion applied, preserving the sharpness of consonants. At 0 everything is converted fully — consonants may sound lisped, watery, or blurred. At 0.5 (maximum) consonants are almost untouched, preserving crispness at the cost of slightly less voice character on those segments.',
            raise: 'Speech sounds lisped, words blend together, or "s" and "t" sounds are soft and indistinct. Consonant crispness is the priority.',
            lower: 'Voice character sounds inconsistent — some phonemes clearly match the target voice but others don\'t. You want more uniform conversion across all sounds.',
          },
          {
            name: 'Silence Gate',
            range: '−70 → −10 dBFS',
            default: '−55 dBFS',
            badge: 'amber',
            summary: 'Audio below this loudness threshold is passed through unprocessed instead of being converted. Prevents the model from "hallucinating" voice during pauses and keeps GPU kernels warm without wasted inference.',
            details: 'Speech typically sits between −30 and −10 dBFS on peaks. Room noise and breath is usually below −60 dBFS. The gate compares the RMS level of each 200 ms block against this threshold. Blocks below threshold are zeroed before conversion and the silence is passed through directly. This also prevents subtle artefacts that occur when the model is fed near-silence and tries to generate phonemes.',
            raise: 'Background noise is being converted and you hear a constant low-level voice-like texture in quiet sections. Raise toward −40 to cut it out.',
            lower: 'Soft-spoken parts or quiet phrases are being silenced and not converted at all. Lower toward −65 to let even quieter audio through the gate.',
          },
        ]} />
      </div>
    </main>
  );
}
