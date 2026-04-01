'use client';

import { useEffect, useRef, useState } from 'react';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, Legend,
} from 'recharts';
import { TipsPanel } from '../TipsPanel';
import { LossGuide } from '../SettingsGuide';
import { ProfilePicker } from '../ProfilePicker';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
const WS_BASE = (API ?? 'http://localhost:8000').replace('http://', 'ws://').replace('https://', 'wss://');

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Profile {
  id: string;
  name: string;
  status: string;
  total_epochs_trained: number;
  needs_retraining: boolean;
  embedder: string;
  vocoder: string;
  audio_files: { id: string; duration: number | null }[];
}

interface HardwareInfo {
  total_ram_gb: number;
  available_ram_gb: number;
  ram_used_pct: number;
  cpu_cores: number;
  mps_available: boolean;
  mps_allocated_mb: number;
  cuda_available: boolean;
  gpu_name?: string | null;
  gpu_vram_gb?: number | null;
  sweet_spot_batch_size: number;
  max_safe_batch_size: number;
}

interface TrainingMsg {
  type: 'log' | 'phase' | 'done' | 'error' | 'keepalive' | 'epoch' | 'epoch_done' | 'index_done';
  message?: string;
  phase?: string;
  elapsed_s?: number;
  epoch?: number;
  losses?: {
    loss_disc?: number;
    loss_gen?: number;
    loss_fm?: number;
    loss_mel?: number;
    loss_kl?: number;
    loss_spk?: number;
  };
}

interface EpochPoint {
  epoch: number;
  loss_mel: number;
  loss_gen: number;
  loss_disc: number;
  loss_fm: number;
  loss_kl: number;
  loss_spk: number;
}

interface EpochLossPoint {
  epoch: number;
  loss_mel: number | null;
  loss_gen: number | null;
  loss_disc: number | null;
  loss_fm: number | null;
  loss_kl: number | null;
  loss_spk: number | null;
  trained_at: string;
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
// Loss chart — SVG sparklines for key training signals
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Generator loss components that stack to form loss_gen_all
// (matches train.py: loss_gen + loss_fm + loss_mel + loss_kl + loss_spk)
// ---------------------------------------------------------------------------
const GEN_STACK = [
  { key: 'loss_spk', label: 'Spk',  color: '#ec4899' }, // pink    — rendered bottom → up
  { key: 'loss_kl',  label: 'KL',   color: '#f87171' }, // red
  { key: 'loss_fm',  label: 'FM',   color: '#34d399' }, // emerald
  { key: 'loss_gen', label: 'Gen',  color: '#a78bfa' }, // violet
  { key: 'loss_mel', label: 'Mel',  color: '#06b6d4' }, // cyan — top (most important)
] as const;

// Custom tooltip: shows epoch, all component values, total gen loss, disc loss
function ChartTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string; dataKey: string }>;
  label?: number;
}) {
  if (!active || !payload?.length) return null;

  // Collect component values from stacked areas
  const vals: Record<string, number> = {};
  for (const entry of payload) {
    vals[entry.dataKey] = entry.value ?? 0;
  }
  const total = GEN_STACK.reduce((s, g) => s + (vals[g.key] ?? 0), 0);
  const disc  = vals['loss_disc'] ?? null;

  return (
    <div className="bg-zinc-900/95 border border-zinc-700 rounded-md px-3 py-2 font-mono text-[10px] shadow-xl min-w-[140px]">
      <div className="text-zinc-400 mb-1.5">epoch {label}</div>
      {GEN_STACK.slice().reverse().map(g => vals[g.key] != null && (
        <div key={g.key} className="flex justify-between gap-4">
          <span style={{ color: g.color }}>{g.label}</span>
          <span className="text-zinc-300">{(vals[g.key] ?? 0).toFixed(3)}</span>
        </div>
      ))}
      <div className="border-t border-zinc-700 mt-1 pt-1 flex justify-between gap-4">
        <span className="text-white font-semibold">Total Gen</span>
        <span className="text-white font-semibold">{total.toFixed(3)}</span>
      </div>
      {disc != null && (
        <div className="flex justify-between gap-4 mt-0.5">
          <span style={{ color: '#f59e0b' }}>Disc</span>
          <span className="text-zinc-300">{disc.toFixed(3)}</span>
        </div>
      )}
    </div>
  );
}

function LossChart({ points, totalEpochs: totalEpochsProp }: { points: EpochPoint[]; totalEpochs?: number }) {
  // Which generator components are visible (all on by default)
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const toggle = (key: string) =>
    setHidden(prev => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });

  if (points.length < 2) {
    return (
      <div className="h-36 flex items-center justify-center text-zinc-600 font-mono text-[11px]">
        {points.length === 0 ? 'Chart will appear after first epoch' : 'Waiting for second epoch…'}
      </div>
    );
  }

  const totalEpochs = totalEpochsProp ?? Math.max(...points.map(p => p.epoch));
  const currentEpoch = points[points.length - 1].epoch;

  // Build data array for Recharts — one object per epoch point
  const data = points.map(p => ({
    epoch:     p.epoch,
    loss_mel:  hidden.has('loss_mel')  ? 0 : (p.loss_mel  ?? 0),
    loss_gen:  hidden.has('loss_gen')  ? 0 : (p.loss_gen  ?? 0),
    loss_fm:   hidden.has('loss_fm')   ? 0 : (p.loss_fm   ?? 0),
    loss_kl:   hidden.has('loss_kl')   ? 0 : (p.loss_kl   ?? 0),
    loss_spk:  hidden.has('loss_spk')  ? 0 : (p.loss_spk  ?? 0),
    loss_disc: p.loss_disc ?? 0,
  }));

  // Latest values for the status row
  const last  = points[points.length - 1];
  const first = points[0];
  const totalLast  = GEN_STACK.reduce((s, g) => s + (last[g.key as keyof EpochPoint]  as number ?? 0), 0);
  const totalFirst = GEN_STACK.reduce((s, g) => s + (first[g.key as keyof EpochPoint] as number ?? 0), 0);

  const axisStyle = { fontSize: 9, fontFamily: 'monospace', fill: '#71717a' };

  return (
    <div className="flex flex-col gap-3">

      {/* ── Stacked area: generator components + total envelope ─────────── */}
      <div>
        <div className="text-[10px] font-mono text-zinc-500 mb-1 flex items-center gap-2">
          <span className="text-cyan-400">▲</span> Generator loss
          <span className="text-zinc-600 ml-1">(stacked — outer edge = total)</span>
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 32 }}>
            <defs>
              {GEN_STACK.map(g => (
                <linearGradient key={g.key} id={`grad_${g.key}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={g.color} stopOpacity={0.35} />
                  <stop offset="95%" stopColor={g.color} stopOpacity={0.05} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
            <XAxis
              dataKey="epoch"
              type="number"
              domain={[1, totalEpochs]}
              tickCount={3}
              tick={axisStyle}
              tickLine={false}
              axisLine={{ stroke: '#3f3f46' }}
            />
            <YAxis
              tick={axisStyle}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => v.toFixed(1)}
              width={30}
            />
            <Tooltip content={<ChartTooltip />} />
            {/* Progress marker: where training currently is */}
            {currentEpoch < totalEpochs && (
              <ReferenceLine
                x={currentEpoch}
                stroke="#52525b"
                strokeDasharray="3 3"
                strokeWidth={1}
              />
            )}
            {/* Stack order: spk → kl → fm → gen → mel (mel on top, most visible) */}
            {GEN_STACK.map(g => (
              <Area
                key={g.key}
                type="monotone"
                dataKey={g.key}
                stackId="gen"
                stroke={hidden.has(g.key) ? 'transparent' : g.color}
                strokeWidth={hidden.has(g.key) ? 0 : 1.5}
                fill={hidden.has(g.key) ? 'transparent' : `url(#grad_${g.key})`}
                isAnimationActive={false}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* ── Discriminator line ───────────────────────────────────────────── */}
      <div>
        <div className="text-[10px] font-mono text-zinc-500 mb-1 flex items-center gap-2">
          <span className="text-amber-400">─</span> Discriminator loss
          <span className="text-zinc-600 ml-1">(separate scale)</span>
        </div>
        <ResponsiveContainer width="100%" height={60}>
          <LineChart data={data} margin={{ top: 2, right: 8, bottom: 0, left: 32 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
            <XAxis
              dataKey="epoch"
              type="number"
              domain={[1, totalEpochs]}
              tickCount={3}
              tick={axisStyle}
              tickLine={false}
              axisLine={{ stroke: '#3f3f46' }}
            />
            <YAxis
              tick={axisStyle}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => v.toFixed(1)}
              width={30}
            />
            <Tooltip content={<ChartTooltip />} />
            {currentEpoch < totalEpochs && (
              <ReferenceLine x={currentEpoch} stroke="#52525b" strokeDasharray="3 3" strokeWidth={1} />
            )}
            <Line
              type="monotone"
              dataKey="loss_disc"
              stroke="#f59e0b"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── Clickable legend + last-value status row ─────────────────────── */}
      <div className="flex flex-wrap gap-x-3 gap-y-1.5">
        {/* Total gen loss summary (always shown) */}
        <span className="flex items-center gap-1.5 font-mono text-[10px] mr-2">
          <span className="inline-block w-3 h-px border-t-2 border-dashed border-zinc-400" />
          <span className="text-zinc-300 font-semibold">Total</span>
          <span className="text-zinc-200 font-semibold">{totalLast.toFixed(3)}</span>
          {(() => {
            const delta = ((totalLast - totalFirst) / Math.max(totalFirst, 1e-9)) * 100;
            return (
              <span className={delta < 0 ? 'text-emerald-400' : 'text-red-400'}>
                {delta < 0 ? '↓' : '↑'}{Math.abs(delta).toFixed(0)}%
              </span>
            );
          })()}
        </span>

        {/* Per-component toggle buttons */}
        {GEN_STACK.slice().reverse().map(g => {
          const lv  = last[g.key as keyof EpochPoint]  as number ?? 0;
          const fv  = first[g.key as keyof EpochPoint] as number ?? 0;
          const pct = fv > 0 ? ((lv - fv) / fv) * 100 : 0;
          const off = hidden.has(g.key);
          return (
            <button
              key={g.key}
              onClick={() => toggle(g.key)}
              className="flex items-center gap-1.5 font-mono text-[10px] rounded px-1.5 py-0.5
                         hover:bg-zinc-800 transition-colors cursor-pointer"
              title={off ? `Show ${g.label}` : `Hide ${g.label}`}
            >
              <span
                className="inline-block w-3 h-1.5 rounded-sm transition-opacity"
                style={{ backgroundColor: g.color, opacity: off ? 0.25 : 1 }}
              />
              <span style={{ color: off ? '#52525b' : g.color }}>{g.label}</span>
              {!off && (
                <>
                  <span className="text-zinc-400">{lv.toFixed(3)}</span>
                  <span className={pct < 0 ? 'text-emerald-400' : 'text-red-400'}>
                    {pct < 0 ? '↓' : '↑'}{Math.abs(pct).toFixed(0)}%
                  </span>
                </>
              )}
            </button>
          );
        })}

        {/* Discriminator (not toggleable — separate chart) */}
        {(() => {
          const lv  = last.loss_disc  ?? 0;
          const fv  = first.loss_disc ?? 0;
          const pct = fv > 0 ? ((lv - fv) / fv) * 100 : 0;
          return (
            <span className="flex items-center gap-1.5 font-mono text-[10px]">
              <span className="inline-block w-3 h-0.5 rounded-full bg-amber-400" />
              <span className="text-amber-400">Disc</span>
              <span className="text-zinc-400">{lv.toFixed(3)}</span>
              <span className={pct < 0 ? 'text-emerald-400' : 'text-red-400'}>
                {pct < 0 ? '↓' : '↑'}{Math.abs(pct).toFixed(0)}%
              </span>
            </span>
          );
        })()}
      </div>
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
            {hw.cuda_available && ` · CUDA ✓${hw.gpu_name ? ` (${hw.gpu_name}` : ''}${hw.gpu_vram_gb ? `, ${hw.gpu_vram_gb} GB VRAM` : ''}${hw.gpu_name || hw.gpu_vram_gb ? ')' : ''}`}
            {!hw.cuda_available && hw.mps_available && ' · MPS ✓'}
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

        {/* Value + badge — fixed-width container so the slider track never reflows */}
        <div className="flex items-center gap-2 shrink-0 w-36">
          <span className={`text-2xl font-mono font-bold tabular-nums w-10 text-right shrink-0 ${
            isRisky ? 'text-red-400' : isSweet ? 'text-emerald-400' : 'text-zinc-100'
          }`}>
            {POW2[idx]}
          </span>
          {/* Render badge placeholder even when empty so width is stable */}
          <span className={`px-1.5 py-0.5 rounded text-[9px] uppercase tracking-wide font-mono w-20 text-center ${
            b ? b.cls : 'invisible'
          }`}>
            {b ? b.text : 'safe / slow'}
          </span>
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

  const [epochs, setEpochs] = useState(300);
  const [batchSize, setBatchSize] = useState(8);
  const [overtrainEnabled, setOvertrainEnabled] = useState(false);
  const [overtrainThreshold, setOvertrainThreshold] = useState(50);
  const [speakerLossWeight, setSpeakerLossWeight] = useState(0);
  const [hw, setHw] = useState<HardwareInfo | null>(null);

  const [logLines, setLogLines] = useState<string[]>([]);
  const [currentPhase, setCurrentPhase] = useState<string | null>(null);
  const [jobState, setJobState] = useState<JobState>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [epochPoints, setEpochPoints] = useState<EpochPoint[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const jobStateRef = useRef<JobState>('idle');

  useEffect(() => { jobStateRef.current = jobState; }, [jobState]);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines]);
  useEffect(() => () => { wsRef.current?.close(); wsRef.current = null; }, []);

  // Load historical epoch losses whenever the selected profile changes.
  // New epochs from a live run are appended on top of these.
  useEffect(() => {
    if (!selectedId) { setEpochPoints([]); return; }
    let cancelled = false;
    fetch(`${API}/api/training/losses/${selectedId}`)
      .then(r => r.ok ? r.json() : [])
      .then((rows: EpochLossPoint[]) => {
        if (cancelled) return;
        setEpochPoints(rows.map(r => ({
          epoch:     r.epoch,
          loss_mel:  r.loss_mel  ?? 0,
          loss_gen:  r.loss_gen  ?? 0,
          loss_disc: r.loss_disc ?? 0,
          loss_fm:   r.loss_fm   ?? 0,
          loss_kl:   r.loss_kl   ?? 0,
          loss_spk:  r.loss_spk  ?? 0,
        })));
      })
      .catch(() => { if (!cancelled) setEpochPoints([]); });
    return () => { cancelled = true; };
  }, [selectedId]);

  // -------------------------------------------------------------------------
  // attachWs — wire up a WebSocket for profile_id and drive UI state from it.
  // Called both from handleStart (new job) and on mount (reconnect to in-progress).
  // -------------------------------------------------------------------------
  function attachWs(profileId: string) {
    closeWs();
    const ws = new WebSocket(`${WS_BASE}/ws/training/${profileId}`);
    wsRef.current = ws;

    ws.onmessage = (ev: MessageEvent) => {
      try {
        const msg = JSON.parse(ev.data as string) as TrainingMsg;
        if (msg.type === 'keepalive') return; // silent keepalive — do nothing
        if (msg.phase && msg.type !== 'done') setCurrentPhase(msg.phase);
        if (msg.type === 'log') {
          if (msg.message) appendLog(msg.message);
        } else if (msg.type === 'phase') {
          if (msg.message) appendLog(`── ${msg.phase?.toUpperCase() ?? ''}: ${msg.message}`);
        } else if (msg.type === 'epoch_done') {
          if (msg.message) appendLog(`✓ ${msg.message}`);
          if (msg.epoch != null && msg.losses) {
            const l = msg.losses;
            setEpochPoints(prev => [...prev, {
              epoch:     msg.epoch!,
              loss_mel:  l.loss_mel  ?? 0,
              loss_gen:  l.loss_gen  ?? 0,
              loss_disc: l.loss_disc ?? 0,
              loss_fm:   l.loss_fm   ?? 0,
              loss_kl:   l.loss_kl   ?? 0,
              loss_spk:  l.loss_spk  ?? 0,
            }]);
          }
        } else if (msg.type === 'done') {
          setCurrentPhase('done');
          setJobState('done');
          jobStateRef.current = 'done';
          if (msg.message) appendLog(`✓ ${msg.message}`);
          // Refresh profiles so epoch count updates
          fetch(`${API}/api/profiles`).then(r => r.ok ? r.json() : null).then(d => { if (d) setProfiles(d); }).catch(() => {});
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
      // onerror fires if the WS upgrade itself fails (backend down, port wrong, etc.)
      // Don't treat this as a training failure — the job may still be running.
      // Log it and let onclose decide state.
      appendLog('(WebSocket connection error — will retry on next action)');
      closeWs();
      if (jobStateRef.current === 'running') {
        // Don't flip to failed — just go idle so user can reconnect
        setJobState('idle');
        jobStateRef.current = 'idle';
      }
    };

    ws.onclose = () => {
      if (jobStateRef.current === 'running') {
        appendLog('(connection closed — training may still be running)');
        setJobState('idle');
        jobStateRef.current = 'idle';
      }
    };
  }

  // -------------------------------------------------------------------------
  // On mount: fetch profiles + hardware, then check if any profile has a
  // running job and reconnect to its WS stream.
  // -------------------------------------------------------------------------
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

        if (hwRes.ok) {
          const hwData: HardwareInfo = await hwRes.json();
          if (!cancelled) {
            setHw(hwData);
            setBatchSize(hwData.sweet_spot_batch_size);
          }
        }

        // Check every profile for a running job; pick the first one found.
        let resumedId: string | null = null;
        for (const p of data) {
          try {
            const sRes = await fetch(`${API}/api/training/status/${p.id}`);
            if (sRes.ok) {
              const s = await sRes.json();
              if (s.status === 'training' || s.phase === 'train' || s.phase === 'preprocess' || s.phase === 'extract_f0' || s.phase === 'extract_feature' || s.phase === 'index') {
                resumedId = p.id;
                break;
              }
            }
          } catch { /* no job for this profile */ }
        }

        if (cancelled) return;

        if (resumedId) {
          // Pre-select the profile that's training
          setSelectedId(resumedId);
          setJobState('running');
          jobStateRef.current = 'running';
          appendLog(`(reconnected — training for profile ${data.find(p => p.id === resumedId)?.name ?? resumedId} already in progress)`);
          attachWs(resumedId);
        } else if (data.length) {
          setSelectedId(data[0].id);
        }
      } catch (err) {
        if (!cancelled) setProfilesError(
          `Cannot reach backend (${err instanceof Error ? err.message : String(err)}). ` +
          'Run: bash scripts/start.sh'
        );
      }
    }
    load();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
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
    // Don't clear epochPoints — historical losses stay; new epochs append on top

    try {
      const res = await fetch(`${API}/api/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: selectedId, epochs, batch_size: batchSize, overtrain_threshold: overtrainEnabled ? overtrainThreshold : 0, c_spk: speakerLossWeight }),
      });

      if (res.status === 409) {
        // Job already running for this profile — connect to it instead of erroring.
        appendLog('(job already running — reconnecting to in-progress training)');
        setJobState('running');
        jobStateRef.current = 'running';
        attachWs(selectedId);
        return;
      }

      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }

      setJobState('running');
      jobStateRef.current = 'running';
      attachWs(selectedId);
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
        {/* In-progress notice */}
        {isRunning && (
          <div className="rounded-lg border border-cyan-800/60 bg-cyan-950/30 px-4 py-3 text-[13px] font-mono text-cyan-300 flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse shrink-0" />
            Training in progress — cancel to interrupt, or watch the log and chart below.
          </div>
        )}

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
            {/* Profile + Epochs — profile list takes most of the width */}
            <div className="flex gap-4 items-start">
              {/* Profile list — flex-1 so it takes remaining space */}
              <div className="flex flex-col gap-2 flex-1 min-w-0">
                <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                  <span className="text-cyan-400">◈</span> Voice Profile
                </label>
                {profiles.length === 0 ? (
                  <div className="bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-[12px] font-mono text-zinc-500">
                    {profilesError ? 'Backend unreachable' : 'No profiles — upload one in Library'}
                  </div>
                ) : (
                  <ProfilePicker
                    profiles={profiles}
                    selectedId={selectedId}
                    onChange={setSelectedId}
                    disabled={isRunning}
                  />
                )}
              </div>

              {/* Epochs — narrow fixed width */}
              <div className="flex flex-col gap-2 w-48 shrink-0">
                <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                  <span className="text-cyan-400">⟳</span> Epochs
                </label>
                <input
                  type="number" value={epochs} min={1} max={200}
                  disabled={isRunning}
                  onChange={(e) => setEpochs(Math.max(1, Math.min(200, Number(e.target.value))))}
                  className="w-full bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                             font-mono text-zinc-200 focus:outline-none focus:border-cyan-600
                             disabled:opacity-40 disabled:cursor-not-allowed hover:border-zinc-600 transition-colors"
                />
                {/* Resume / retrain hint */}
                {(() => {
                  const sel = profiles.find(p => p.id === selectedId);
                  if (!sel) return null;
                  return (
                    <div className="flex flex-col gap-1">
                      {sel.total_epochs_trained > 0 && (
                        <span className="text-[10px] font-mono text-amber-500/80 leading-tight">
                          ↳ resume {sel.total_epochs_trained} → {sel.total_epochs_trained + epochs}
                        </span>
                      )}
                      {sel.needs_retraining && (
                        <span className="text-[10px] font-mono text-amber-400 leading-tight">
                          ⚠ retrain recommended
                        </span>
                      )}
                    </div>
                  );
                })()}
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

            {/* Overtraining Stop + Speaker Loss Weight — same row */}
            <div className="border-t border-zinc-800/60 pt-4">
              <div className="flex flex-wrap gap-x-8 gap-y-4">

                {/* Overtraining Stop */}
                <div className="flex flex-col gap-2">
                  <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                    <span className="text-amber-400">⚠</span> Overtraining Stop
                  </label>
                  {/* Checkbox row */}
                  <label className="flex items-center gap-2 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={overtrainEnabled}
                      disabled={isRunning}
                      onChange={(e) => setOvertrainEnabled(e.target.checked)}
                      className="w-4 h-4 accent-amber-500 disabled:opacity-40 cursor-pointer"
                    />
                    <span className="text-[12px] font-mono text-zinc-300">
                      {overtrainEnabled ? 'Enabled' : 'Disabled'}
                    </span>
                  </label>
                  {/* Slider — only shown when enabled */}
                  {overtrainEnabled && (
                    <div className="flex flex-col gap-1 mt-1">
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min={10} max={200} step={5}
                          value={overtrainThreshold}
                          disabled={isRunning}
                          onChange={(e) => setOvertrainThreshold(Number(e.target.value))}
                          className="w-40 accent-amber-500 disabled:opacity-40"
                        />
                        <span className="text-[12px] font-mono text-amber-400 w-8 text-right">
                          {overtrainThreshold}
                        </span>
                      </div>
                      <span className="text-[10px] font-mono text-zinc-500">
                        stop after {overtrainThreshold} epoch{overtrainThreshold !== 1 ? 's' : ''} without improvement
                      </span>
                    </div>
                  )}
                  <span className="text-[10px] font-mono text-zinc-600">
                    Tracks epoch-average loss · best model always saved · recommended: 50
                  </span>
                </div>

                {/* Speaker Loss Weight */}
                <div className="flex flex-col gap-2 min-w-[180px]">
                  <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                    <span className="text-pink-400">🎤</span> Speaker Loss Weight (c_spk)
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="range" value={speakerLossWeight} min={0} max={3} step={0.5}
                      disabled={isRunning}
                      onChange={(e) => setSpeakerLossWeight(Number(e.target.value))}
                      className="flex-1 accent-pink-600"
                    />
                    <span className="text-[13px] font-mono text-pink-300 w-8 text-right">
                      {speakerLossWeight === 0 ? 'off' : speakerLossWeight.toFixed(1)}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-zinc-600">
                    {speakerLossWeight === 0
                      ? 'disabled — no speaker identity loss'
                      : `ECAPA-TDNN cosine loss · recommended: 2.0–3.0 for voice cloning`}
                  </span>
                </div>

              </div>
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

        {epochPoints.length > 0 ? (
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">Loss Curves</h2>
              <span className="text-[10px] font-mono text-zinc-600">
                {epochPoints.length} epoch{epochPoints.length !== 1 ? 's' : ''}
                {isRunning ? ` · +${epochs} target` : ''}
              </span>
            </div>
            <div className="bg-zinc-950 rounded-lg p-3 border border-zinc-800">
              <LossChart
                points={epochPoints}
                totalEpochs={isRunning
                  ? Math.max(epochPoints.length > 0 ? epochPoints[epochPoints.length - 1].epoch : 0, epochs)
                  : undefined}
              />
            </div>
          </section>
        ) : null}

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
          Training data must be uploaded as voice sample(s) in the Library tab first.
        </footer>

        {/* Tips */}
        <TipsPanel tips={[
          {
            icon: '📊',
            title: 'Epoch count: 200–500 for speech, 500–1000 for singing',
            body: 'Results at 25 epochs are audible but rough. Speech cloning typically becomes convincing at 200+ epochs. More training data means fewer epochs needed for the same quality.',
          },
          {
            icon: '🛑',
            title: 'Overtraining detection is active',
            body: 'When the generator loss stops improving and starts climbing, training halts automatically. This is a sign to stop — more epochs past this point hurt quality.',
          },
          {
            icon: '🎙️',
            title: 'Audio quality matters more than quantity',
            body: 'One hour of clean, single-speaker, dry (no reverb) audio beats 10 hours of noisy or multi-speaker recordings. Remove silence, music beds, and cross-talk before uploading.',
          },
          {
            icon: '🔁',
            title: 'Batch size and save frequency',
            body: 'Larger batches train faster but need more GPU memory. Save every 10–25 epochs so you can roll back to an earlier checkpoint if overtraining is detected.',
          },
          {
            icon: '🗂️',
            title: 'SPIN-v2 embedder gives the best speaker identity preservation',
            body: 'ContentVec is a good fallback if SPIN-v2 produces artefacts. HuBERT is the most forgiving with noisy or reverberant data.',
          },
          {
            icon: '🔊',
            title: 'RefineGAN vs HiFi-GAN',
            body: 'RefineGAN produces crisper, more detailed speech at the cost of slightly longer training time. For quick experiments, HiFi-GAN converges faster and is more forgiving.',
          },
          {
            icon: '🎤',
            title: 'Speaker Loss (ECAPA-TDNN) for better voice cloning',
            body: 'The training pipeline now includes an optional speaker identity loss using ECAPA-TDNN. Set c_spk > 0 in the training config to activate it. This loss directly optimises for speaker similarity, making the cloned voice sound more like the target speaker. Recommended weight: 1.0–3.0.',
          },
        ]} />

        {/* Loss reference */}
        <LossGuide losses={[
          {
            key: 'loss_mel',
            color: '#06b6d4',
            name: 'Mel Loss',
            what: 'Measures the difference between the spectrogram of the generated audio and the real audio on a mel-frequency scale — the same scale human hearing uses. This is the single most important signal for perceptual audio quality. A low mel loss means the generated audio sounds tonally similar to the training data: correct pitch, correct timbre, correct vowel shapes.',
            healthy: 'Steadily decreasing over epochs, eventually plateauing. Typical good values are below 1.0; excellent fine-tunes reach 0.3–0.6. This should be the primary metric you watch.',
            warning: 'Rising mel loss after it has plateaued is a strong overtraining signal — the model is fitting noise rather than speech structure. Training halts automatically when this is detected.',
          },
          {
            key: 'loss_gen',
            color: '#a78bfa',
            name: 'Generator Loss',
            what: 'The adversarial loss from the discriminator\'s verdict on generated audio. The generator (voice synthesis network) is trying to fool the discriminator into believing its output is real. A lower generator loss means the discriminator is having a harder time telling generated from real — which generally means the audio sounds more natural.',
            healthy: 'Should decrease early and then oscillate in a moderate range as the generator and discriminator reach a dynamic equilibrium. Some oscillation is normal and expected in GAN training.',
            warning: 'If generator loss collapses to near zero, the discriminator has stopped providing useful gradient — mode collapse. If it climbs sharply, the discriminator has become too strong and the generator is struggling.',
          },
          {
            key: 'loss_disc',
            color: '#f59e0b',
            name: 'Discriminator Loss',
            what: 'How well the discriminator can distinguish real training audio from generated audio. This should stay in a moderate range throughout training — not too high (discriminator is failing) and not too low (discriminator is too dominant, starving the generator of useful gradient).',
            healthy: 'Stays roughly stable in a moderate range after initial settling. A healthy training run has generator and discriminator locked in a productive adversarial tension.',
            warning: 'Near-zero discriminator loss means it has become too strong — the generator gets no useful gradient and output quality stagnates. Very high discriminator loss means the generator is winning easily — the discriminator needs to catch up.',
          },
          {
            key: 'loss_fm',
            color: '#34d399',
            name: 'Feature Matching Loss',
            what: 'Compares the internal feature activations of the discriminator between real and generated audio, at multiple layers. Rather than just asking "real or fake?", it asks "do the intermediate representations match?". This stabilises GAN training and improves fine-grained detail in the output — voice texture, breathiness, and consonant crispness.',
            healthy: 'Decreases steadily alongside mel loss, usually converging a bit faster. Values below 5.0 are typical for a well-trained model.',
            warning: 'Feature matching loss plateauing high while mel loss is still falling suggests the discriminator architecture may be a bottleneck — less actionable, but worth noting if output sounds blurry at high epoch counts.',
          },
          {
            key: 'loss_kl',
            color: '#f87171',
            name: 'KL Divergence Loss',
            what: 'The Kullback–Leibler divergence between the posterior (encoder\'s estimate of the latent from real audio) and the prior (the model\'s prior over the latent space). This regularises the latent space so it stays smooth and continuous, which prevents the decoder from memorising training examples instead of learning to generalise.',
            healthy: 'Decreases quickly and stays low (< 2.0) throughout training. Once it has settled it should barely move.',
            warning: 'A climbing KL loss late in training can indicate the model is over-regularising and losing expressiveness. In practice this is rare — KL loss is usually the most stable of the five signals.',
          },
          {
            key: 'loss_spk',
            color: '#ec4899',
            name: 'Speaker Loss',
            what: 'Cosine distance between ECAPA-TDNN speaker embeddings of generated vs. real audio. This loss directly optimises for speaker identity preservation — forcing the cloned voice to sound like the target speaker, not just phonetically correct. Uses the converted ecapa_tdnn.pt model (assets/ecapa/). Only active when c_spk > 0 in the training config.',
            healthy: 'Should decrease steadily toward 0. Values below 0.3 indicate strong speaker identity match. The ECAPA-TDNN encoder is frozen and only provides reference embeddings — it does not add trainable parameters.',
            warning: 'If speaker loss stays high while other losses are good, the model is learning content but not speaker identity — consider increasing c_spk weight. If speaker loss drops but audio quality degrades, the weight may be too high and conflicting with mel loss.',
          },
        ]} />

      </div>
    </main>
  );
}
