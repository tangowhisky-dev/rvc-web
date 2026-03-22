'use client';

import { useEffect, useRef, useState } from 'react';

const API = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Profile {
  id: string;
  name: string;
  status: string;
}

interface HardwareInfo {
  total_ram_gb: number;
  available_ram_gb: number;
  ram_used_pct: number;
  cpu_cores: number;
  mps_available: boolean;
  mps_allocated_mb: number;
  sweet_spot_batch_size: number;
  max_safe_batch_size: number;
}

interface TrainingMsg {
  type: 'log' | 'phase' | 'done' | 'error' | 'keepalive' | 'epoch' | 'epoch_done' | 'index_done';
  message?: string;
  phase?: string;
  elapsed_s?: number;
  epoch?: number;
}

type JobState = 'idle' | 'running' | 'done' | 'failed';

function nowStamp() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false });
}

// ---------------------------------------------------------------------------
// Phase bar
// ---------------------------------------------------------------------------

const PHASES = ['preprocess', 'extract_f0', 'extract_feature', 'train', 'index', 'done'] as const;
const PHASE_LABELS: Record<string, string> = {
  preprocess: 'Preprocess', extract_f0: 'F0 Extract',
  extract_feature: 'Features', train: 'Train', index: 'Index', done: 'Done',
};

function PhaseBar({ currentPhase, jobDone }: { currentPhase: string | null; jobDone: boolean }) {
  if (!currentPhase) return null;
  const active = jobDone ? 'done' : currentPhase;
  const activeIdx = PHASES.indexOf(active as (typeof PHASES)[number]);
  return (
    <div className="flex gap-1.5">
      {PHASES.map((p, i) => {
        let cls: string;
        if (p === active && p === 'done')
          cls = 'bg-emerald-700 text-emerald-100 shadow-[0_0_8px_rgba(16,185,129,0.4)]';
        else if (p === active)
          cls = 'bg-cyan-600 text-white shadow-[0_0_8px_rgba(8,145,178,0.5)]';
        else if (i < (activeIdx < 0 ? 0 : activeIdx))
          cls = 'bg-zinc-700 text-zinc-400';
        else
          cls = 'bg-zinc-900 text-zinc-600 border border-zinc-800';
        return (
          <div key={p} className={`flex-1 py-1.5 rounded text-center text-[10px] font-mono font-medium uppercase tracking-wider transition-all ${cls}`}>
            {PHASE_LABELS[p]}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Batch Size Selector — slider with sweet-spot and max-safe markers
// ---------------------------------------------------------------------------

// Batch size must be a power of two — RVC's data loader expects 2^n
const POW2 = [1, 2, 4, 8, 16, 32] as const;
type Pow2 = typeof POW2[number];

function nearestPow2Index(v: number): number {
  let best = 0;
  for (let i = 0; i < POW2.length; i++) {
    if (POW2[i] <= v) best = i;
  }
  return best;
}

function BatchSizeSelector({
  value, onChange, disabled, hw,
}: {
  value: number;
  onChange: (v: number) => void;
  disabled: boolean;
  hw: HardwareInfo | null;
}) {
  const sweet   = hw?.sweet_spot_batch_size ?? 8;
  const maxSafe = hw?.max_safe_batch_size   ?? 16;

  // Slider index (0–5) → actual pow2 value
  const idx       = nearestPow2Index(value);
  const maxIdx    = POW2.length - 1;
  const sweetIdx  = nearestPow2Index(sweet);
  const safeIdx   = nearestPow2Index(maxSafe);

  // Map index to 0–100% position on the track
  const idxPct = (i: number) => (i / maxIdx) * 100;

  const isRisky = idx > safeIdx;
  const isSweet = POW2[idx] === sweet;
  const fillColor = isRisky ? '#ef4444' : '#06b6d4';
  const fillPct   = idxPct(idx);

  function badge() {
    if (isSweet)  return { text: 'sweet spot', cls: 'bg-emerald-900/50 text-emerald-300' };
    if (isRisky)  return { text: 'risky',      cls: 'bg-red-900/50 text-red-300' };
    if (idx <= 1) return { text: 'safe / slow', cls: 'bg-zinc-800 text-zinc-400' };
    return null;
  }
  const b = badge();

  return (
    <div className="flex flex-col gap-3">
      {/* Label row */}
      <div className="flex items-center justify-between">
        <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-1.5">
          <span className="text-cyan-400">▦</span> Batch Size
        </label>
        {hw && (
          <span className="text-[10px] font-mono text-zinc-500">
            {hw.total_ram_gb} GB RAM · {hw.available_ram_gb} GB free
            {hw.mps_available && ' · MPS ✓'}
          </span>
        )}
      </div>

      {/* Slider row */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 pt-5">

          {/* Sweet-spot pin */}
          <div
            className="absolute top-0 -translate-x-1/2 flex flex-col items-center gap-0.5 pointer-events-none"
            style={{ left: `${idxPct(sweetIdx)}%` }}
          >
            <span className="text-[9px] font-mono text-emerald-400 whitespace-nowrap leading-none">
              {sweet}
            </span>
            <div className="w-1 h-1 rounded-full bg-emerald-500" />
          </div>

          {/* Max-safe boundary line — only shown when not already at max */}
          {safeIdx < maxIdx && (
            <div
              className="absolute top-3.5 bottom-0 w-px bg-amber-500/40 pointer-events-none"
              style={{ left: `${idxPct(safeIdx)}%` }}
            />
          )}

          {/* The slider moves by index; snap to POW2 on change */}
          <input
            type="range"
            min={0}
            max={maxIdx}
            step={1}
            value={idx}
            disabled={disabled}
            onChange={(e) => onChange(POW2[Number(e.target.value)])}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer
                       disabled:opacity-40 disabled:cursor-not-allowed"
            style={{
              accentColor: fillColor,
              background: `linear-gradient(to right, ${fillColor} ${fillPct}%, #27272a ${fillPct}%)`,
            }}
          />

          {/* Tick marks — one per POW2 value */}
          <div className="flex justify-between mt-1 pointer-events-none">
            {POW2.map((p, i) => (
              <span
                key={p}
                className={`text-[9px] font-mono -translate-x-1/2 ${
                  i === idx ? (isRisky ? 'text-red-400' : isSweet ? 'text-emerald-400' : 'text-zinc-200')
                  : i <= idx ? 'text-zinc-500'
                  : 'text-zinc-700'
                }`}
                style={{ width: 0, textAlign: 'center', display: 'inline-block' }}
              >
                {p}
              </span>
            ))}
          </div>
        </div>

        {/* Value + badge */}
        <div className="flex items-center gap-2 shrink-0">
          <span className={`text-2xl font-mono font-bold tabular-nums w-10 text-right ${
            isRisky ? 'text-red-400' : isSweet ? 'text-emerald-400' : 'text-zinc-100'
          }`}>
            {POW2[idx]}
          </span>
          {b && (
            <span className={`px-1.5 py-0.5 rounded text-[9px] uppercase tracking-wide font-mono ${b.cls}`}>
              {b.text}
            </span>
          )}
        </div>
      </div>

      {/* Legend */}
      {hw && (
        <div className="flex items-center gap-5 text-[10px] font-mono text-zinc-600">
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block shrink-0" />
            sweet spot: <span className="text-emerald-400 ml-0.5">{sweet}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-px h-3 bg-amber-500/60 inline-block shrink-0" />
            max safe: <span className="text-zinc-400 ml-0.5">{maxSafe}</span>
            {isRisky && <span className="text-red-400 ml-1">↑ over limit</span>}
          </span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function TrainingPage() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [selectedId, setSelectedId] = useState('');
  const [profilesError, setProfilesError] = useState<string | null>(null);

  const [epochs, setEpochs] = useState(20);
  const [batchSize, setBatchSize] = useState(8);
  const [hw, setHw] = useState<HardwareInfo | null>(null);

  const [logLines, setLogLines] = useState<string[]>([]);
  const [currentPhase, setCurrentPhase] = useState<string | null>(null);
  const [jobState, setJobState] = useState<JobState>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const jobStateRef = useRef<JobState>('idle');

  useEffect(() => { jobStateRef.current = jobState; }, [jobState]);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines]);
  useEffect(() => () => { wsRef.current?.close(); wsRef.current = null; }, []);

  // Fetch profiles + hardware on mount
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [pRes, hwRes] = await Promise.all([
          fetch(`${API}/api/profiles`),
          fetch(`${API}/api/training/hardware`),
        ]);
        if (!pRes.ok) throw new Error(`HTTP ${pRes.status}`);
        const data: Profile[] = await pRes.json();
        if (cancelled) return;
        setProfiles(data);
        if (data.length) setSelectedId(data[0].id);

        if (hwRes.ok) {
          const hwData: HardwareInfo = await hwRes.json();
          if (!cancelled) {
            setHw(hwData);
            setBatchSize(hwData.sweet_spot_batch_size);
          }
        }
      } catch (err) {
        if (!cancelled) setProfilesError(
          `Cannot reach backend (${err instanceof Error ? err.message : String(err)}). ` +
          'Start the backend: conda run -n rvc uvicorn backend.app.main:app --reload'
        );
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  function appendLog(line: string, ts?: string) {
    const stamp = ts ?? nowStamp();
    setLogLines((prev) => {
      const next = [...prev, `[${stamp}] ${line}`];
      return next.length > 500 ? next.slice(-500) : next;
    });
  }

  function closeWs() {
    wsRef.current?.close();
    wsRef.current = null;
  }

  async function handleStart() {
    if (!selectedId || jobState === 'running') return;
    setErrorMsg(null);
    setLogLines([]);
    setCurrentPhase(null);

    try {
      const res = await fetch(`${API}/api/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: selectedId, epochs, batch_size: batchSize }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }

      setJobState('running');
      jobStateRef.current = 'running';

      const ws = new WebSocket(`${WS_BASE}/ws/training/${selectedId}`);
      wsRef.current = ws;

      ws.onmessage = (ev: MessageEvent) => {
        try {
          const msg = JSON.parse(ev.data as string) as TrainingMsg;
          if (msg.phase && msg.type !== 'done') setCurrentPhase(msg.phase);
          if (msg.type === 'log') {
            if (msg.message) appendLog(msg.message);
          } else if (msg.type === 'phase') {
            if (msg.message) appendLog(`── ${msg.phase?.toUpperCase() ?? ''}: ${msg.message}`);
          } else if (msg.type === 'epoch_done') {
            if (msg.message) appendLog(`✓ ${msg.message}`);
          } else if (msg.type === 'done') {
            setCurrentPhase('done');
            setJobState('done');
            jobStateRef.current = 'done';
            if (msg.message) appendLog(`✓ ${msg.message}`);
            closeWs();
          } else if (msg.type === 'error') {
            setJobState('failed');
            jobStateRef.current = 'failed';
            const t = msg.message ?? 'Unknown error';
            setErrorMsg(t);
            appendLog(`ERROR: ${t}`);
            closeWs();
          }
        } catch { /* malformed JSON */ }
      };

      ws.onerror = () => {
        appendLog('(WebSocket error)');
        if (jobStateRef.current === 'running') {
          setJobState('failed');
          jobStateRef.current = 'failed';
          setErrorMsg('WebSocket error — check backend logs.');
        }
        closeWs();
      };

      ws.onclose = () => {
        if (jobStateRef.current === 'running') {
          appendLog('(connection closed)');
          setJobState('idle');
          jobStateRef.current = 'idle';
        }
      };
    } catch (err) {
      const m = err instanceof Error ? err.message : String(err);
      setErrorMsg(m);
      setJobState('failed');
      jobStateRef.current = 'failed';
    }
  }

  async function handleCancel() {
    if (!selectedId) return;
    try {
      await fetch(`${API}/api/training/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: selectedId }),
      });
    } catch { /* best-effort */ }
    closeWs();
    setJobState('idle');
    jobStateRef.current = 'idle';
    appendLog('(training cancelled)');
  }

  const isRunning = jobState === 'running';
  const canStart  = !!selectedId && jobState !== 'running';

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="text-cyan-400">
              <circle cx="2" cy="6" r="1.2" fill="currentColor"/>
              <circle cx="6" cy="2" r="1.2" fill="currentColor"/>
              <circle cx="6" cy="10" r="1.2" fill="currentColor"/>
              <circle cx="10" cy="6" r="1.2" fill="currentColor"/>
              <line x1="3.2" y1="6" x2="4.8" y2="2.8" stroke="currentColor" strokeWidth="0.8"/>
              <line x1="3.2" y1="6" x2="4.8" y2="9.2" stroke="currentColor" strokeWidth="0.8"/>
              <line x1="7.2" y1="2.8" x2="8.8" y2="6" stroke="currentColor" strokeWidth="0.8"/>
              <line x1="7.2" y1="9.2" x2="8.8" y2="6" stroke="currentColor" strokeWidth="0.8"/>
            </svg>
          </div>
          <h1 className="text-sm font-mono font-medium tracking-wide">
            RVC <span className="text-cyan-400">Training</span>
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full transition-colors ${
            isRunning ? 'bg-cyan-400 shadow-[0_0_6px_rgba(34,211,238,0.6)] animate-pulse'
            : jobState === 'done' ? 'bg-emerald-400'
            : jobState === 'failed' ? 'bg-red-400'
            : 'bg-zinc-600'
          }`} />
          <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
            {isRunning ? 'training' : jobState === 'done' ? 'done' : jobState === 'failed' ? 'failed' : 'idle'}
          </span>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-6 py-8 flex flex-col gap-8">
        {(profilesError || (jobState === 'failed' && errorMsg)) && (
          <div className="rounded-lg border border-red-800/60 bg-red-950/40 px-4 py-3 text-[13px] font-mono text-red-300">
            {profilesError || errorMsg}
          </div>
        )}

        <section>
          <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500 mb-4">
            Training Configuration
          </h2>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5 flex flex-col gap-5">

            {/* Profile + Epochs row */}
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-2">
                <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                  <span className="text-cyan-400">◈</span> Voice Profile
                </label>
                {profiles.length === 0 ? (
                  <div className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px] font-mono text-zinc-500">
                    {profilesError ? 'Backend unreachable' : 'No profiles — upload one in Library'}
                  </div>
                ) : (
                  <select
                    value={selectedId}
                    disabled={isRunning}
                    onChange={(e) => setSelectedId(e.target.value)}
                    className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                               font-mono text-zinc-200 focus:outline-none focus:border-cyan-600
                               disabled:opacity-40 disabled:cursor-not-allowed hover:border-zinc-600 transition-colors"
                  >
                    {profiles.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                        {p.status === 'trained' ? ' ✓' : p.status === 'training' ? ' ⟳' : p.status === 'failed' ? ' ✗' : ''}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              <div className="flex flex-col gap-2">
                <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                  <span className="text-cyan-400">⟳</span> Epochs
                </label>
                <input
                  type="number" value={epochs} min={1} max={200}
                  disabled={isRunning}
                  onChange={(e) => setEpochs(Math.max(1, Math.min(200, Number(e.target.value))))}
                  className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                             font-mono text-zinc-200 focus:outline-none focus:border-cyan-600
                             disabled:opacity-40 disabled:cursor-not-allowed hover:border-zinc-600 transition-colors"
                />
              </div>
            </div>

            {/* Batch size slider */}
            <div className="border-t border-zinc-800/60 pt-4">
              <BatchSizeSelector
                value={batchSize}
                onChange={setBatchSize}
                disabled={isRunning}
                hw={hw}
              />
            </div>

            {/* Actions */}
            <div className="flex items-center gap-3 pt-2 border-t border-zinc-800">
              <button
                onClick={handleStart}
                disabled={!canStart}
                className="flex-1 py-3 rounded-lg font-mono text-[13px] font-medium tracking-wider uppercase
                           transition-all bg-cyan-900/40 border border-cyan-600/40 text-cyan-300
                           hover:bg-cyan-800/40 hover:border-cyan-500/60
                           disabled:opacity-30 disabled:cursor-not-allowed"
              >
                {jobState === 'done' ? '↺ Train Again' : '▶ Start Training'}
              </button>
              {isRunning && (
                <button
                  onClick={handleCancel}
                  className="flex-1 py-3 rounded-lg font-mono text-[13px] font-medium tracking-wider uppercase
                             transition-all bg-red-900/40 border border-red-700/40 text-red-300
                             hover:bg-red-800/40 hover:border-red-600/60"
                >
                  ◼ Cancel
                </button>
              )}
            </div>
          </div>
        </section>

        {currentPhase !== null && (
          <section>
            <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500 mb-3">Progress</h2>
            <PhaseBar currentPhase={currentPhase} jobDone={jobState === 'done'} />
          </section>
        )}

        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">Training Log</h2>
            {logLines.length > 0 && (
              <span className="text-[10px] font-mono text-zinc-600">{logLines.length} lines</span>
            )}
          </div>
          <div
            ref={logRef}
            className="overflow-y-auto max-h-80 bg-zinc-950 rounded-lg p-3 font-mono text-[11px] border border-zinc-800"
          >
            {logLines.length === 0 ? (
              <div className="text-zinc-600 select-none">
                {isRunning ? 'Waiting for log output…' : 'No training output yet. Start a job to see logs.'}
              </div>
            ) : logLines.map((line, i) => {
              const m = line.match(/^(\[\d{2}:\d{2}:\d{2}\]) ([\s\S]*)$/);
              const ts      = m ? m[1] : null;
              const content = m ? m[2] : line;
              const isError = content.startsWith('ERROR:') || content.startsWith('[stderr]');
              const isMuted = content.startsWith('(');
              const isPhase = content.startsWith('──');
              const isEpoch = content.startsWith('✓');
              return (
                <div key={i} className={`leading-relaxed whitespace-pre-wrap break-all flex gap-2 ${
                  isError ? 'text-red-400'
                  : isMuted ? 'text-zinc-500 italic'
                  : isPhase ? 'text-cyan-400 font-medium'
                  : isEpoch ? 'text-emerald-400'
                  : 'text-zinc-300'
                }`}>
                  {ts && <span className="shrink-0 text-zinc-600 select-none">{ts}</span>}
                  <span>{content}</span>
                </div>
              );
            })}
          </div>
        </section>

        <footer className="text-[11px] font-mono text-zinc-600 pb-4">
          Training data must be uploaded as a voice sample in the Library tab first.
        </footer>
      </div>
    </main>
  );
}
