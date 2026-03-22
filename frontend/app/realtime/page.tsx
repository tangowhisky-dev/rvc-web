'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

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
}

interface SessionParams {
  pitch: number;
  index_rate: number;
  protect: number;
  silence_threshold_db: number;
}

type SessionState = 'idle' | 'starting' | 'active' | 'stopping';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API = 'http://localhost:8000';
const DEFAULT_PARAMS: SessionParams = { pitch: 0, index_rate: 0.75, protect: 0.33, silence_threshold_db: -45 };

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
      .then((d) => setSavePath(d.path + '/rvc_output.mp3'))
      .catch(() => setSavePath('~/Downloads/rvc_output.mp3'));
  }, []);

  // The backend default-save-dir endpoint returns a full absolute path, so savePath
  // is always absolute in the normal flow. expandedSavePath is only needed if the user
  // manually types ~/... (error-fallback case). Keep it as the text-field display value.
  const expandedSavePath = savePath;
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);

  // Params
  const [params, setParams] = useState<SessionParams>(DEFAULT_PARAMS);

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

  // Keep paramsRef in sync
  useEffect(() => {
    paramsRef.current = params;
  }, [params]);

  // Keep sessionIdRef in sync
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

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

    loadDevices();
    loadProfiles();
    return () => {
      cancelled = true;
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Cleanup helper
  // ---------------------------------------------------------------------------

  const cleanup = useCallback(() => {
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
    (key: keyof SessionParams, value: number) => {
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
          ...(saveEnabled && resolvedSavePath ? { save_path: resolvedSavePath } : {}),
        }),
      });

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
      const ws = new WebSocket(`ws://localhost:8000/ws/realtime/${session_id}`);
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
    }
    // Otherwise: let ws.onmessage({ type: 'done' }) do the cleanup.
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
              {profiles.length === 0 ? (
                <div className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                                font-mono text-zinc-500">
                  No trained profiles — train one in the Training tab first
                </div>
              ) : (
                <select
                  value={profileId ?? ''}
                  disabled={isActive || isBusy}
                  onChange={(e) => setProfileId(e.target.value)}
                  className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                             font-mono text-zinc-200 focus:outline-none focus:border-cyan-600
                             disabled:opacity-40 disabled:cursor-not-allowed
                             hover:border-zinc-600 transition-colors"
                >
                  {profiles.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
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
                    placeholder="/Users/tango16/Downloads/rvc_output.mp3"
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
            <div className="grid grid-cols-1 gap-6">
              <ParamSlider
                label={`Silence Gate (${params.silence_threshold_db} dBFS)`}
                value={params.silence_threshold_db}
                min={-70}
                max={-10}
                step={1}
                disabled={false}
                onChange={(v) => handleParamChange('silence_threshold_db', v)}
              />
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
      </div>
    </main>
  );
}
