'use client';

import { useEffect, useRef, useState } from 'react';
import {
  ComposedChart, Line, LineChart,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ReferenceDot,
  ResponsiveContainer,
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
  pipeline: string;
  best_epoch: number | null;
  best_avg_gen_loss: number | null;
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
  type: 'log' | 'phase' | 'done' | 'error' | 'keepalive' | 'epoch' | 'epoch_done' | 'index_done' | 'step_done' | 'progress';
  message?: string;
  phase?: string;
  elapsed_s?: number;
  epoch?: number;
  is_best?: boolean;
  best_epoch?: number;
  avg_gen?: number;
  best_avg_gen?: number;
  // Beatrice 2 step fields
  step?: number;
  total_steps?: number;
  progress_pct?: number;
  losses?: {
    // RVC epoch losses
    loss_disc?: number;
    loss_gen?: number;
    loss_fm?: number;
    loss_mel?: number;
    loss_kl?: number;
    loss_spk?: number;
    // Beatrice 2 step losses
    loss_loud?: number;
    loss_ap?: number;
    loss_adv?: number;
    loss_d?: number;
    utmos?: number;
    is_best?: number;
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

interface BeatriceStepPoint {
  step: number;
  loss_mel:  number | null;
  loss_loud: number | null;
  loss_ap:   number | null;
  loss_adv:  number | null;
  loss_fm:   number | null;
  loss_d:    number | null;
  utmos:     number | null;
  is_best:   number | null;  // 1 if this step was best UTMOS
  trained_at: string;
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
const PHASES_B2 = ['preprocess', 'extract_f0', 'extract_feature', 'train', 'done'] as const;
const PHASE_LABELS: Record<string, string> = {
  preprocess: 'Preprocess', extract_f0: 'F0 Extract',
  extract_feature: 'Features', train: 'Train', index: 'Index', done: 'Done',
};

function PhaseBar({ currentPhase, jobDone, pipeline }: { currentPhase: string | null; jobDone: boolean; pipeline?: string }) {
  if (!currentPhase) return null;
  const phases = pipeline === 'beatrice2' ? PHASES_B2 : PHASES;
  const active = jobDone ? 'done' : currentPhase;
  const activeIdx = phases.indexOf(active as any);
  return (
    <div className="flex gap-1.5">
      {phases.map((p, i) => {
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
// Loss chart — convergence-oriented visualization
// ---------------------------------------------------------------------------

// EMA smoothing — α=0.15 gives a ~6-epoch half-life.
// Returns an array the same length as input; earlier values are less smoothed
// but we start from the first value to avoid a biased warm-up.
function ema(values: number[], alpha = 0.15): number[] {
  const out: number[] = [];
  let s = values[0];
  for (const v of values) {
    s = alpha * v + (1 - alpha) * s;
    out.push(s);
  }
  return out;
}

// Slope of a line fit through the last `window` EMA values, normalised by
// the mean value so it is scale-independent (units: fraction per epoch).
function recentSlope(values: number[], window = 15): number {
  const tail = values.slice(-window);
  if (tail.length < 2) return 0;
  const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
  if (mean === 0) return 0;
  // Simple linear regression slope
  const n = tail.length;
  const xMean = (n - 1) / 2;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - xMean) * (tail[i] - mean);
    den += (i - xMean) ** 2;
  }
  return den === 0 ? 0 : (num / den) / mean;
}

// Convergence status derived from mel EMA slope and KL EMA slope.
// Mel is the primary quality signal; KL indicates latent alignment.
//
// Thresholds use normalised slope (fractional change per epoch):
//   mel 'still improving' : < -0.001  (≈ 0.02 units/epoch at mel=19)
//   mel 'overtraining'    : >  0.005
//   kl  'latent converging': < -0.002  (≈ 0.004 units/epoch at kl=2)
//
// Rationale: late-stage fine-tuning has small absolute drops that are still
// meaningful.  The old thresholds (-0.003 mel, -0.005 kl) were calibrated for
// early training where mel≈35 and produced false-positive "converged" once mel
// settled below ~20 and both slopes slowed proportionally.
function convergenceStatus(
  melSlope: number,
  klSlope: number,
  epochCount: number,
): { label: string; color: string; detail: string } {
  // Need at least 30 epochs: EMA with α=0.15 takes ~20 epochs to wash out
  // early transients, so the 15-epoch slope window isn't reliable before that.
  if (epochCount < 30) {
    return { label: 'warming up', color: '#71717a', detail: 'Not enough epochs for trend analysis.' };
  }
  // Rising mel = overtraining signal
  if (melSlope > 0.005) {
    return {
      label: 'possible overtraining',
      color: '#f87171',
      detail: 'Mel loss is rising — spectral quality may be degrading. Consider stopping.',
    };
  }
  // Mel still dropping meaningfully
  if (melSlope < -0.001) {
    return {
      label: 'still improving',
      color: '#34d399',
      detail: 'Mel loss is still trending down — more training likely helps.',
    };
  }
  // Mel flat, KL still dropping — latent alignment in progress
  if (klSlope < -0.002) {
    return {
      label: 'latent converging',
      color: '#60a5fa',
      detail: 'Mel has plateaued but KL is still dropping — prior/posterior alignment continuing.',
    };
  }
  // Both flat — converged
  return {
    label: 'converged',
    color: '#a78bfa',
    detail: 'Both mel and KL have plateaued. Model has extracted what it can from this data.',
  };
}

function LossTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ dataKey: string; value: number; color: string }>;
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  const vals: Record<string, number> = {};
  for (const e of payload) vals[e.dataKey] = e.value;

  const rows: Array<{ key: string; label: string; color: string }> = [
    { key: 'mel_raw',  label: 'Mel (raw)',   color: '#06b6d4' },
    { key: 'mel_ema',  label: 'Mel (trend)', color: '#06b6d4' },
    { key: 'gen_raw',  label: 'Gen (raw)',   color: '#a78bfa' },
    { key: 'gen_ema',  label: 'Gen (trend)', color: '#a78bfa' },
    { key: 'kl_raw',   label: 'KL (raw)',    color: '#f87171' },
    { key: 'kl_ema',   label: 'KL (trend)',  color: '#f87171' },
    { key: 'fm_raw',   label: 'FM (raw)',    color: '#34d399' },
    { key: 'disc_raw', label: 'Disc',        color: '#f59e0b' },
  ];

  return (
    <div className="bg-zinc-900/95 border border-zinc-700 rounded-md px-3 py-2 font-mono text-[10px] shadow-xl min-w-[150px]">
      <div className="text-zinc-400 mb-1.5 font-semibold">epoch {label}</div>
      {rows.map(r => vals[r.key] != null ? (
        <div key={r.key} className="flex justify-between gap-4">
          <span style={{ color: r.color, opacity: r.key.endsWith('_raw') ? 0.55 : 1 }}>{r.label}</span>
          <span className="text-zinc-300">{vals[r.key].toFixed(3)}</span>
        </div>
      ) : null)}
    </div>
  );
}

function LossChart({
  points,
  totalEpochs: totalEpochsProp,
  bestEpoch,
}: {
  points: EpochPoint[];
  totalEpochs?: number;
  bestEpoch?: number | null;
}) {
  if (points.length < 2) {
    return (
      <div className="h-36 flex items-center justify-center text-zinc-600 font-mono text-[11px]">
        {points.length === 0 ? 'Chart will appear after first epoch' : 'Waiting for second epoch…'}
      </div>
    );
  }

  const totalEpochs  = totalEpochsProp ?? Math.max(...points.map(p => p.epoch));
  const currentEpoch = points[points.length - 1].epoch;
  const first        = points[0];
  const last         = points[points.length - 1];

  // Compute EMA series for mel, gen, kl, fm
  const melRaw  = points.map(p => p.loss_mel  ?? 0);
  const genRaw  = points.map(p => p.loss_gen  ?? 0);
  const klRaw   = points.map(p => p.loss_kl   ?? 0);
  const fmRaw   = points.map(p => p.loss_fm   ?? 0);
  const discRaw = points.map(p => p.loss_disc ?? 0);

  const melEma  = ema(melRaw);
  const genEma  = ema(genRaw);
  const klEma   = ema(klRaw);

  // Convergence signal from recent EMA slope
  const melSlope = recentSlope(melEma);
  const klSlope  = recentSlope(klEma);
  const status   = convergenceStatus(melSlope, klSlope, points.length);

  // Per-component change from first to last EMA
  function pctChange(raw: number[], e: number[]): string {
    const delta = ((e[e.length - 1] - raw[0]) / Math.max(raw[0], 1e-9)) * 100;
    return `${delta < 0 ? '↓' : '↑'}${Math.abs(delta).toFixed(0)}%`;
  }
  function pctColor(raw: number[], e: number[]): string {
    return e[e.length - 1] < raw[0] ? '#34d399' : '#f87171';
  }

  // Chart data — one point per epoch, both raw and EMA values
  const data = points.map((p, i) => ({
    epoch:    p.epoch,
    mel_raw:  melRaw[i],
    mel_ema:  melEma[i],
    gen_raw:  genRaw[i],
    gen_ema:  genEma[i],
    kl_raw:   klRaw[i],
    kl_ema:   klEma[i],
    fm_raw:   fmRaw[i],
    disc_raw: discRaw[i],
  }));

  const axisStyle = { fontSize: 9, fontFamily: 'monospace', fill: '#52525b' };
  const gridProps = { strokeDasharray: '3 3' as const, stroke: '#1f1f23', vertical: false };
  const xAxisProps = {
    dataKey: 'epoch' as const,
    type:    'number' as const,
    domain:  [1, totalEpochs] as [number, number],
    tickCount: 4,
    tick:    axisStyle,
    tickLine: false,
    axisLine: { stroke: '#3f3f46' },
  };

  // Best-epoch line — show only when it's meaningfully before current
  const showBestLine = bestEpoch != null && bestEpoch > 0 && bestEpoch < currentEpoch - 1;

  return (
    <div className="flex flex-col gap-3">

      {/* ── Convergence status banner ─────────────────────────────────────── */}
      <div
        className="flex items-start gap-2 rounded px-2.5 py-1.5 text-[10px] font-mono"
        style={{ backgroundColor: `${status.color}14`, borderLeft: `2px solid ${status.color}` }}
      >
        <span style={{ color: status.color }} className="font-semibold shrink-0 mt-px">
          {status.label}
        </span>
        <span className="text-zinc-500">{status.detail}</span>
      </div>

      {/* ── Primary chart: Mel (left axis) + Gen + KL (right axis) ────────── */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-3 text-[10px] font-mono">
            {/* Mel */}
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-0.5 rounded-full" style={{ backgroundColor: '#06b6d4' }} />
              <span style={{ color: '#06b6d4' }}>Mel</span>
              <span className="text-zinc-400">{melEma[melEma.length - 1].toFixed(2)}</span>
              <span style={{ color: pctColor(melRaw, melEma) }}>{pctChange(melRaw, melEma)}</span>
            </span>
            {/* Gen */}
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-0.5 rounded-full" style={{ backgroundColor: '#a78bfa' }} />
              <span style={{ color: '#a78bfa' }}>Gen</span>
              <span className="text-zinc-400">{genEma[genEma.length - 1].toFixed(3)}</span>
              <span style={{ color: pctColor(genRaw, genEma) }}>{pctChange(genRaw, genEma)}</span>
            </span>
            {/* KL */}
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-0.5 rounded-full" style={{ backgroundColor: '#f87171' }} />
              <span style={{ color: '#f87171' }}>KL</span>
              <span className="text-zinc-400">{klEma[klEma.length - 1].toFixed(3)}</span>
              <span style={{ color: pctColor(klRaw, klEma) }}>{pctChange(klRaw, klEma)}</span>
            </span>
          </div>
          <span className="text-[9px] font-mono text-zinc-600">faded = raw · solid = trend</span>
        </div>
        <ResponsiveContainer width="100%" height={170}>
          <ComposedChart data={data} margin={{ top: 4, right: 42, bottom: 0, left: 28 }}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xAxisProps} />
            {/* Left axis: Mel (larger values ~10–26) */}
            <YAxis
              yAxisId="mel"
              orientation="left"
              tick={axisStyle}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => v.toFixed(0)}
              width={26}
              domain={['auto', 'auto']}
            />
            {/* Right axis: Gen + KL (smaller values ~1–8) */}
            <YAxis
              yAxisId="small"
              orientation="right"
              tick={axisStyle}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => v.toFixed(1)}
              width={30}
              domain={[0, 'auto']}
            />
            <Tooltip content={<LossTooltip />} />

            {/* Best-epoch marker */}
            {showBestLine && (
              <ReferenceLine
                x={bestEpoch}
                yAxisId="mel"
                stroke="#f59e0b"
                strokeDasharray="4 3"
                strokeWidth={1.5}
                label={{
                  value: `best ${bestEpoch}`,
                  position: 'insideTopLeft',
                  fontSize: 8,
                  fontFamily: 'monospace',
                  fill: '#f59e0b',
                  dy: -2,
                }}
              />
            )}

            {/* Progress marker */}
            {currentEpoch < totalEpochs && (
              <ReferenceLine
                x={currentEpoch}
                yAxisId="mel"
                stroke="#3f3f46"
                strokeDasharray="3 3"
                strokeWidth={1}
              />
            )}

            {/* Mel — raw faded, EMA solid */}
            <Line yAxisId="mel"   type="monotone" dataKey="mel_raw"  stroke="#06b6d4" strokeWidth={1}   dot={false} strokeOpacity={0.2} isAnimationActive={false} />
            <Line yAxisId="mel"   type="monotone" dataKey="mel_ema"  stroke="#06b6d4" strokeWidth={2}   dot={false} isAnimationActive={false} />
            {/* Gen — raw faded, EMA solid */}
            <Line yAxisId="small" type="monotone" dataKey="gen_raw"  stroke="#a78bfa" strokeWidth={1}   dot={false} strokeOpacity={0.2} isAnimationActive={false} />
            <Line yAxisId="small" type="monotone" dataKey="gen_ema"  stroke="#a78bfa" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            {/* KL — raw faded, EMA solid */}
            <Line yAxisId="small" type="monotone" dataKey="kl_raw"   stroke="#f87171" strokeWidth={1}   dot={false} strokeOpacity={0.2} isAnimationActive={false} />
            <Line yAxisId="small" type="monotone" dataKey="kl_ema"   stroke="#f87171" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ── Secondary chart: FM + Disc ────────────────────────────────────── */}
      <div>
        <div className="flex items-center gap-3 mb-1 text-[10px] font-mono">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-0.5 rounded-full bg-emerald-400" />
            <span className="text-emerald-400">FM</span>
            <span className="text-zinc-400">{(fmRaw[fmRaw.length - 1] ?? 0).toFixed(2)}</span>
            <span style={{ color: pctColor(fmRaw, ema(fmRaw)) }}>{pctChange(fmRaw, ema(fmRaw))}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-0.5 rounded-full bg-amber-400" />
            <span className="text-amber-400">Disc</span>
            <span className="text-zinc-400">{(discRaw[discRaw.length - 1] ?? 0).toFixed(3)}</span>
            <span className="text-zinc-600 text-[9px] ml-1">≈ flat = healthy equilibrium</span>
          </span>
        </div>
        <ResponsiveContainer width="100%" height={70}>
          <ComposedChart data={data} margin={{ top: 2, right: 42, bottom: 0, left: 28 }}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xAxisProps} />
            <YAxis
              yAxisId="left"
              orientation="left"
              tick={axisStyle}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => v.toFixed(0)}
              width={26}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={axisStyle}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => v.toFixed(1)}
              width={30}
            />
            <Tooltip content={<LossTooltip />} />
            {showBestLine && (
              <ReferenceLine x={bestEpoch} yAxisId="left" stroke="#f59e0b" strokeDasharray="4 3" strokeWidth={1} strokeOpacity={0.5} />
            )}
            {currentEpoch < totalEpochs && (
              <ReferenceLine x={currentEpoch} yAxisId="left" stroke="#3f3f46" strokeDasharray="3 3" strokeWidth={1} />
            )}
            <Line yAxisId="left"  type="monotone" dataKey="fm_raw"   stroke="#34d399" strokeWidth={1.5} dot={false} strokeOpacity={0.4} isAnimationActive={false} />
            <Line yAxisId="right" type="monotone" dataKey="disc_raw" stroke="#f59e0b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ── Reading guide ─────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[9px] font-mono text-zinc-600 pt-1 border-t border-zinc-800/50">
        <span><span className="text-cyan-500">Mel</span> · left axis · still dropping = room to improve</span>
        <span><span className="text-red-400">KL</span> · right axis · near zero = latent aligned</span>
        <span><span className="text-violet-400">Gen</span> · right axis · flat = adversarial equilibrium</span>
        <span><span className="text-amber-400">best {bestEpoch ?? '—'}</span> · dashed amber = G_best.pth saved here</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// BeatriceStepChart — step-based training chart for Beatrice 2 profiles
//
// Matches the visual quality of LossChart (RVC):
//   - Stat cards: current value + % change from first point
//   - EMA smoothing on mel (main reconstruction signal)
//   - Custom tooltip showing all 6 losses
//   - Two panels: reconstruction (mel/loud/ap) + adversarial (adv/fm/disc)
// ---------------------------------------------------------------------------

function B2Tooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ dataKey: string; value: number; color: string; name: string }>;
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  const vals: Record<string, number> = {};
  for (const e of payload) if (e.value != null) vals[e.dataKey] = e.value;
  const rows: Array<{ key: string; label: string; color: string }> = [
    { key: 'utmos_raw', label: 'UTMOS ↑',   color: '#facc15' },
    { key: 'mel_raw',  label: 'Mel (raw)',   color: '#22d3ee' },
    { key: 'mel_ema',  label: 'Mel (trend)', color: '#22d3ee' },
    { key: 'loud_raw', label: 'Loud',        color: '#34d399' },
    { key: 'ap_raw',   label: 'AP',          color: '#a3e635' },
    { key: 'adv_raw',  label: 'Adv',         color: '#a78bfa' },
    { key: 'fm_raw',   label: 'FM',          color: '#fb923c' },
    { key: 'd_raw',    label: 'Disc',        color: '#f59e0b' },
  ];
  return (
    <div className="bg-zinc-900/95 border border-zinc-700 rounded-md px-3 py-2 font-mono text-[10px] shadow-xl min-w-[150px]">
      <div className="text-zinc-400 mb-1.5 font-semibold">step {label}</div>
      {rows.map(r => vals[r.key] != null ? (
        <div key={r.key} className="flex justify-between gap-4">
          <span style={{ color: r.color, opacity: r.key.endsWith('_raw') && !r.key.startsWith('mel') && !r.key.startsWith('utmos') ? 1 : r.key === 'mel_raw' ? 0.55 : 1 }}>{r.label}</span>
          <span className="text-zinc-300">{vals[r.key].toFixed(3)}</span>
        </div>
      ) : null)}
    </div>
  );
}

function BeatriceStepChart({ points }: { points: BeatriceStepPoint[] }) {
  if (points.length < 2) {
    return (
      <div className="h-36 flex items-center justify-center text-zinc-600 font-mono text-[11px]">
        {points.length === 0 ? 'Chart will appear after first steps' : 'Waiting for more steps…'}
      </div>
    );
  }

  // Extract raw series in step order
  const sorted = [...points].sort((a, b) => a.step - b.step);
  const melRaw  = sorted.map(p => p.loss_mel  ?? 0);
  const loudRaw = sorted.map(p => p.loss_loud ?? 0);
  const apRaw   = sorted.map(p => p.loss_ap   ?? 0);
  const advRaw  = sorted.map(p => p.loss_adv  ?? 0);
  const fmRaw   = sorted.map(p => p.loss_fm   ?? 0);
  const dRaw    = sorted.map(p => p.loss_d    ?? 0);

  // UTMOS — sparse (only at evaluation steps); null entries are gaps in the line
  const utmosPoints = sorted
    .map((p, i) => p.utmos != null ? { step: p.step, utmos_raw: p.utmos, i } : null)
    .filter((x): x is { step: number; utmos_raw: number; i: number } => x !== null);
  const bestUtmosPoint = sorted.find(p => p.is_best === 1);
  const bestUtmosStep  = bestUtmosPoint?.step ?? null;
  const bestUtmos      = bestUtmosPoint?.utmos ?? null;
  const latestUtmos    = utmosPoints[utmosPoints.length - 1]?.utmos_raw ?? null;
  const hasUtmos       = utmosPoints.length > 0;

  const melEma = ema(melRaw);

  function pctChange(raw: number[], e: number[]): string {
    const delta = ((e[e.length - 1] - raw[0]) / Math.max(Math.abs(raw[0]), 1e-9)) * 100;
    return `${delta < 0 ? '↓' : '↑'}${Math.abs(delta).toFixed(0)}%`;
  }
  function pctColor(raw: number[], e: number[]): string {
    return e[e.length - 1] < raw[0] ? '#34d399' : '#f87171';
  }

  const chartData = sorted.map((p, i) => ({
    step:      p.step,
    mel_raw:   melRaw[i],
    mel_ema:   melEma[i],
    loud_raw:  loudRaw[i],
    ap_raw:    apRaw[i],
    adv_raw:   advRaw[i],
    fm_raw:    fmRaw[i],
    d_raw:     dRaw[i],
    utmos_raw: p.utmos ?? undefined,   // undefined = gap in line (recharts skips nulls)
  }));

  const lastStep  = sorted[sorted.length - 1].step;
  const firstStep = sorted[0].step;
  const axisStyle = { fontSize: 9, fontFamily: 'monospace', fill: '#52525b' } as const;
  const gridProps = { strokeDasharray: '3 3' as const, stroke: '#1f1f23', vertical: false };
  const xAxisProps = {
    dataKey: 'step' as const,
    type:    'number' as const,
    domain:  [firstStep, lastStep] as [number, number],
    tickCount: 4,
    tick:    axisStyle,
    tickLine: false,
    axisLine: { stroke: '#3f3f46' },
  };

  return (
    <div className="flex flex-col gap-3">

      {/* ── UTMOS panel — quality ceiling signal ──────────────────────── */}
      {hasUtmos ? (
        <div className="border border-yellow-900/40 rounded-lg p-2 bg-yellow-950/10">
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-3 text-[10px] font-mono flex-wrap">
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-3 h-0.5 rounded-full" style={{ backgroundColor: '#facc15' }} />
                <span style={{ color: '#facc15' }} className="font-semibold">UTMOS</span>
                {latestUtmos != null && (
                  <span className="text-zinc-300 font-semibold">current: {latestUtmos.toFixed(3)}</span>
                )}
                {bestUtmos != null && bestUtmosStep != null && (
                  <span className="text-yellow-500/70 text-[9px]">★ best: {bestUtmos.toFixed(3)} @ step {bestUtmosStep}</span>
                )}
              </span>
            </div>
            <span className="text-[9px] font-mono text-zinc-600">higher = better (1–5 MOS scale)</span>
          </div>
          <ResponsiveContainer width="100%" height={90}>
            <ComposedChart data={chartData} margin={{ top: 4, right: 36, bottom: 0, left: 24 }}>
              <CartesianGrid {...gridProps} />
              <XAxis {...xAxisProps} hide />
              <YAxis yAxisId="left" orientation="left" tick={axisStyle} tickLine={false} axisLine={false}
                tickFormatter={(v: number) => v.toFixed(2)} width={28} domain={[1, 5]} />
              <Tooltip content={<B2Tooltip />} />
              {bestUtmosStep != null && (
                <ReferenceLine yAxisId="left" x={bestUtmosStep} stroke="#facc15" strokeOpacity={0.3} strokeDasharray="4 2" />
              )}
              <Line yAxisId="left" type="monotone" dataKey="utmos_raw" stroke="#facc15" strokeWidth={2}
                dot={{ r: 3, fill: '#facc15', strokeWidth: 0 }} connectNulls={true}
                isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <div className="text-[9px] font-mono text-yellow-700/80 mt-1 leading-tight">
            Evaluated every {utmosPoints.length > 1
              ? Math.round(((utmosPoints[utmosPoints.length - 1].step - utmosPoints[0].step) / (utmosPoints.length - 1)) / 100) * 100
              : '—'} steps · plateau = training ceiling reached
          </div>
        </div>
      ) : (
        <div className="border border-zinc-800/40 rounded-lg p-2 text-[10px] font-mono text-zinc-600">
          UTMOS will appear after the first evaluation checkpoint
          {' '}(every {Math.max(500, sorted.length > 0 ? Math.round((sorted[sorted.length-1].step) / 20 / 500) * 500 : 500)} steps)
        </div>
      )}

      {/* ── Reconstruction panel: Mel + Loud + AP ─────────────────────── */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-3 text-[10px] font-mono flex-wrap">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-0.5 rounded-full" style={{ backgroundColor: '#22d3ee' }} />
              <span style={{ color: '#22d3ee' }}>Mel</span>
              <span className="text-zinc-400">{melEma[melEma.length - 1].toFixed(3)}</span>
              <span style={{ color: pctColor(melRaw, melEma) }}>{pctChange(melRaw, melEma)}</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-0.5 rounded-full" style={{ backgroundColor: '#34d399' }} />
              <span style={{ color: '#34d399' }}>Loud</span>
              <span className="text-zinc-400">{loudRaw[loudRaw.length - 1].toFixed(3)}</span>
              <span style={{ color: pctColor(loudRaw, ema(loudRaw)) }}>{pctChange(loudRaw, ema(loudRaw))}</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-0.5 rounded-full" style={{ backgroundColor: '#a3e635' }} />
              <span style={{ color: '#a3e635' }}>AP</span>
              <span className="text-zinc-400">{apRaw[apRaw.length - 1].toFixed(3)}</span>
              <span style={{ color: pctColor(apRaw, ema(apRaw)) }}>{pctChange(apRaw, ema(apRaw))}</span>
            </span>
          </div>
          <span className="text-[9px] font-mono text-zinc-600">faded = raw · solid = trend</span>
        </div>
        <ResponsiveContainer width="100%" height={150}>
          <ComposedChart data={chartData} margin={{ top: 4, right: 36, bottom: 0, left: 24 }}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xAxisProps} hide />
            <YAxis yAxisId="left" orientation="left"  tick={axisStyle} tickLine={false} axisLine={false}
              tickFormatter={(v: number) => v.toFixed(1)} width={28} domain={['auto', 'auto']} />
            <YAxis yAxisId="right" orientation="right" tick={axisStyle} tickLine={false} axisLine={false}
              tickFormatter={(v: number) => v.toFixed(2)} width={32} domain={['auto', 'auto']} />
            <Tooltip content={<B2Tooltip />} />
            <Line yAxisId="left"  type="monotone" dataKey="mel_raw"  stroke="#22d3ee" strokeWidth={1}   dot={false} strokeOpacity={0.25} isAnimationActive={false} />
            <Line yAxisId="left"  type="monotone" dataKey="mel_ema"  stroke="#22d3ee" strokeWidth={2}   dot={false} isAnimationActive={false} />
            <Line yAxisId="right" type="monotone" dataKey="loud_raw" stroke="#34d399" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="right" type="monotone" dataKey="ap_raw"   stroke="#a3e635" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ── Adversarial panel: Adv + FM + Disc ────────────────────────── */}
      <div>
        <div className="flex items-center gap-3 mb-1 text-[10px] font-mono flex-wrap">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-0.5 rounded-full" style={{ backgroundColor: '#a78bfa' }} />
            <span style={{ color: '#a78bfa' }}>Adv</span>
            <span className="text-zinc-400">{advRaw[advRaw.length - 1].toFixed(3)}</span>
            <span style={{ color: pctColor(advRaw, ema(advRaw)) }}>{pctChange(advRaw, ema(advRaw))}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-0.5 rounded-full" style={{ backgroundColor: '#fb923c' }} />
            <span style={{ color: '#fb923c' }}>FM</span>
            <span className="text-zinc-400">{fmRaw[fmRaw.length - 1].toFixed(3)}</span>
            <span style={{ color: pctColor(fmRaw, ema(fmRaw)) }}>{pctChange(fmRaw, ema(fmRaw))}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-0.5 rounded-full" style={{ backgroundColor: '#f59e0b' }} />
            <span style={{ color: '#f59e0b' }}>Disc</span>
            <span className="text-zinc-400">{dRaw[dRaw.length - 1].toFixed(3)}</span>
            <span className="text-zinc-600 text-[9px] ml-1">≈ flat = healthy</span>
          </span>
        </div>
        <ResponsiveContainer width="100%" height={90}>
          <ComposedChart data={chartData} margin={{ top: 2, right: 36, bottom: 0, left: 24 }}>
            <CartesianGrid {...gridProps} />
            <XAxis {...xAxisProps}
              label={{ value: 'step', position: 'insideBottomRight', offset: -4, fontSize: 8, fontFamily: 'monospace', fill: '#52525b' }} />
            <YAxis yAxisId="left" orientation="left"  tick={axisStyle} tickLine={false} axisLine={false}
              tickFormatter={(v: number) => v.toFixed(1)} width={28} />
            <YAxis yAxisId="right" orientation="right" tick={axisStyle} tickLine={false} axisLine={false}
              tickFormatter={(v: number) => v.toFixed(2)} width={32} />
            <Tooltip content={<B2Tooltip />} />
            <Line yAxisId="left"  type="monotone" dataKey="adv_raw" stroke="#a78bfa" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="left"  type="monotone" dataKey="fm_raw"  stroke="#fb923c" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line yAxisId="right" type="monotone" dataKey="d_raw"   stroke="#f59e0b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="text-[9px] font-mono text-zinc-600 text-right">
        step {lastStep}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Batch Size Selector — slider with sweet-spot and max-safe markers
// ---------------------------------------------------------------------------

// Batch size must be a power of two — RVC's data loader expects 2^n
const POW2 = [1, 2, 4, 8, 16, 32, 64, 128, 256] as const;
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
                className={`text-[8px] font-mono -translate-x-1/2 ${
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
        <div className="flex items-center gap-2 shrink-0 w-40">
          <span className={`text-2xl font-mono font-bold tabular-nums w-14 text-right shrink-0 ${
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

  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(8);
  const [overtrainEnabled, setOvertrainEnabled] = useState(false);
  const [overtrainThreshold, setOvertrainThreshold] = useState(50);
  const [lossMode, setLossMode] = useState<'classic' | 'combined'>('classic');
  const [speakerLossWeight, setSpeakerLossWeight] = useState(3);
  const [advLoss, setAdvLoss] = useState<'lsgan' | 'tprls'>('lsgan');
  const [klAnneal, setKlAnneal] = useState(false);
  const [klAnnealEpochs, setKlAnnealEpochs] = useState(40);
  const [optimizer, setOptimizer] = useState<'adamw' | 'adamspd'>('adamw');
  const [hw, setHw] = useState<HardwareInfo | null>(null);

  const [logLines, setLogLines] = useState<string[]>([]);
  const [currentPhase, setCurrentPhase] = useState<string | null>(null);
  const [jobState, setJobState] = useState<JobState>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [epochPoints, setEpochPoints] = useState<EpochPoint[]>([]);
  const [stepPoints, setStepPoints]   = useState<BeatriceStepPoint[]>([]);
  const [historyKey, setHistoryKey]   = useState(0);  // increment to force history re-fetch

  const wsRef = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const jobStateRef = useRef<JobState>('idle');

  useEffect(() => { jobStateRef.current = jobState; }, [jobState]);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines]);
  useEffect(() => () => { wsRef.current?.close(); wsRef.current = null; }, []);

  // Load historical epoch/step losses whenever the selected profile changes.
  useEffect(() => {
    if (!selectedId) { setEpochPoints([]); setStepPoints([]); return; }
    const sel = profiles.find(p => p.id === selectedId);
    const isB2 = sel?.pipeline === 'beatrice2';
    let cancelled = false;
    fetch(`${API}/api/training/losses/${selectedId}`)
      .then(r => r.ok ? r.json() : [])
      .then((rows: (EpochLossPoint | BeatriceStepPoint)[]) => {
        if (cancelled) return;
        if (isB2) {
          setStepPoints((rows as BeatriceStepPoint[]).map(r => ({
            step:      r.step,
            loss_mel:  r.loss_mel  ?? null,
            loss_loud: r.loss_loud ?? null,
            loss_ap:   r.loss_ap   ?? null,
            loss_adv:  r.loss_adv  ?? null,
            loss_fm:   r.loss_fm   ?? null,
            loss_d:    r.loss_d    ?? null,
            utmos:     r.utmos     ?? null,
            is_best:   r.is_best   ?? null,
            trained_at: r.trained_at,
          })));
        } else {
          setEpochPoints((rows as EpochLossPoint[]).map(r => ({
            epoch:     (r as EpochLossPoint).epoch,
            loss_mel:  (r as EpochLossPoint).loss_mel  ?? 0,
            loss_gen:  (r as EpochLossPoint).loss_gen  ?? 0,
            loss_disc: (r as EpochLossPoint).loss_disc ?? 0,
            loss_fm:   (r as EpochLossPoint).loss_fm   ?? 0,
            loss_kl:   (r as EpochLossPoint).loss_kl   ?? 0,
            loss_spk:  (r as EpochLossPoint).loss_spk  ?? 0,
          })));
        }
      })
      .catch(() => { if (!cancelled) { setEpochPoints([]); setStepPoints([]); } });
    return () => { cancelled = true; };
  }, [selectedId, profiles, historyKey]);

  // -------------------------------------------------------------------------
  // attachWs — wire up a WebSocket for profile_id and drive UI state from it.
  // Called both from handleStart (new job) and on mount (reconnect to in-progress).
  // -------------------------------------------------------------------------
  function attachWs(profileId: string) {
    closeWs();

    // Seed chart with all historical losses before live events arrive.
    const selProfile = profiles.find(p => p.id === profileId);
    const isB2Profile = selProfile?.pipeline === 'beatrice2';
    fetch(`${API}/api/training/losses/${profileId}`)
      .then(r => r.ok ? r.json() : [])
      .then((rows: (EpochLossPoint | BeatriceStepPoint)[]) => {
        if (!rows.length) return;
        if (isB2Profile) {
          setStepPoints(prev => {
            const existingSteps = new Set(prev.map(p => p.step));
            const fresh = (rows as BeatriceStepPoint[]).filter(r => !existingSteps.has(r.step));
            if (!fresh.length) return prev;
            return [...prev, ...fresh].sort((a, b) => a.step - b.step);
          });
        } else {
          setEpochPoints(prev => {
            const existingEpochs = new Set(prev.map(p => p.epoch));
            const fresh = (rows as EpochLossPoint[])
              .filter(r => !existingEpochs.has(r.epoch))
              .map(r => ({
                epoch:     r.epoch,
                loss_mel:  r.loss_mel  ?? 0,
                loss_gen:  r.loss_gen  ?? 0,
                loss_disc: r.loss_disc ?? 0,
                loss_fm:   r.loss_fm   ?? 0,
                loss_kl:   r.loss_kl   ?? 0,
                loss_spk:  r.loss_spk  ?? 0,
              }));
            if (!fresh.length) return prev;
            return [...prev, ...fresh].sort((a, b) => a.epoch - b.epoch);
          });
        }
      })
      .catch(() => {});

    const ws = new WebSocket(`${WS_BASE}/ws/training/${profileId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      // Refresh history so UTMOS and past steps are loaded from DB immediately
      setHistoryKey(k => k + 1);
    };

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
            const point = {
              epoch:     msg.epoch!,
              loss_mel:  l.loss_mel  ?? 0,
              loss_gen:  l.loss_gen  ?? 0,
              loss_disc: l.loss_disc ?? 0,
              loss_fm:   l.loss_fm   ?? 0,
              loss_kl:   l.loss_kl   ?? 0,
              loss_spk:  l.loss_spk  ?? 0,
            };
            // Upsert by epoch — live event replaces any history row for same epoch
            setEpochPoints(prev => {
              const without = prev.filter(p => p.epoch !== point.epoch);
              return [...without, point].sort((a, b) => a.epoch - b.epoch);
            });
          }
          // Keep best_epoch marker in sync during live training
          if (msg.is_best && msg.best_epoch != null) {
            setProfiles(prev => prev.map(p =>
              p.id === profileId
                ? { ...p, best_epoch: msg.best_epoch!, best_avg_gen_loss: msg.best_avg_gen ?? p.best_avg_gen_loss }
                : p
            ));
          }
        } else if (msg.type === 'step_done') {
          // Beatrice 2 step event — losses come from DB via _fetch_step_losses
          if (msg.step != null && msg.losses) {
            const l = msg.losses;
            const pt: BeatriceStepPoint = {
              step:      msg.step,
              loss_mel:  l.loss_mel  ?? null,
              loss_loud: l.loss_loud ?? null,
              loss_ap:   l.loss_ap   ?? null,
              loss_adv:  l.loss_adv  ?? null,
              loss_fm:   l.loss_fm   ?? null,
              loss_d:    l.loss_d    ?? null,
              utmos:     l.utmos     ?? null,
              is_best:   l.is_best   ?? null,
              trained_at: new Date().toISOString(),
            };
            setStepPoints(prev => {
              const without = prev.filter(p => p.step !== pt.step);
              return [...without, pt].sort((a, b) => a.step - b.step);
            });
          }
          if (msg.message) appendLog(msg.message);
        } else if (msg.type === 'done') {
          setCurrentPhase('done');
          setJobState('done');
          jobStateRef.current = 'done';
          if (msg.message) appendLog(`✓ ${msg.message}`);
          // Refresh profiles so epoch count updates
          fetch(`${API}/api/profiles`).then(r => r.ok ? r.json() : null).then(d => { if (d) setProfiles(d); }).catch(() => {});
          closeWs();
        } else if (msg.type === 'error') {
          const t = msg.message ?? 'Unknown error';
          // User-cancel is not a terminal failure — the pipeline continues to
          // save the checkpoint and build the index. Keep the WS open and stay
          // in 'running' state so subsequent log/done messages are received.
          const isCancel = t.toLowerCase().includes('cancelled');
          if (isCancel) {
            appendLog(`⚠ ${t}`);
          } else {
            setJobState('failed');
            jobStateRef.current = 'failed';
            setErrorMsg(t);
            appendLog(`ERROR: ${t}`);
            closeWs();
          }
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
            // Only apply hardware recommendation when no job is running —
            // on reconnect, the running job's batch size is restored below.
            setBatchSize(hwData.sweet_spot_batch_size);
          }
        }

        // Check every profile for a running job; pick the first one found.
        let resumedId: string | null = null;
        let resumedStatus: { total_epoch?: number; batch_size?: number } = {};
        for (const p of data) {
          try {
            const sRes = await fetch(`${API}/api/training/status/${p.id}`);
            if (sRes.ok) {
              const s = await sRes.json();
              if (
                s.status === 'training' &&
                (s.phase === 'train' || s.phase === 'preprocess' || s.phase === 'extract_f0' || s.phase === 'extract_feature' || s.phase === 'index')
              ) {
                resumedId = p.id;
                resumedStatus = { total_epoch: s.total_epoch, batch_size: s.batch_size };
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
          // Restore the exact epochs + batch size from the running job so the
          // UI shows the correct values instead of defaults/hardware suggestions.
          if (resumedStatus.total_epoch != null) setEpochs(resumedStatus.total_epoch);
          if (resumedStatus.batch_size   != null) setBatchSize(resumedStatus.batch_size);
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
        body: JSON.stringify({ profile_id: selectedId, epochs, batch_size: batchSize, overtrain_threshold: overtrainEnabled ? overtrainThreshold : 0, c_spk: lossMode === 'classic' ? 0 : speakerLossWeight, loss_mode: lossMode, adv_loss: advLoss, kl_anneal: klAnneal, kl_anneal_epochs: klAnnealEpochs, optimizer }),
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
    // Do NOT close the WS here — the pipeline continues (index build +
    // artifact save) and will send log messages and a final 'done' event.
    // The 'done' handler will close the WS and set state to 'done'.
    appendLog('(cancel requested — waiting for checkpoint save and index build…)');
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

              {/* Epochs / Steps — narrow fixed width */}
              <div className="flex flex-col gap-2 w-48 shrink-0">
                {(() => {
                  const sel = profiles.find(p => p.id === selectedId);
                  const isB2 = sel?.pipeline === 'beatrice2';
                  const minB2Steps = isB2 ? Math.round(10000 * Math.sqrt(8 / batchSize)) : 1;
                  return (
                    <>
                      <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                        <span className="text-cyan-400">⟳</span> {isB2 ? 'Steps' : 'Epochs'}
                      </label>
                      <input
                        type="number" value={epochs}
                        min={isB2 ? minB2Steps : 1} max={isB2 ? 50000 : 1000}
                        step={isB2 ? 500 : 1}
                        disabled={isRunning}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          setEpochs(isB2 ? Math.max(minB2Steps, Math.min(50000, v)) : Math.max(1, Math.min(1000, v)));
                        }}
                        className="w-full bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                                   font-mono text-zinc-200 focus:outline-none focus:border-cyan-600
                                   disabled:opacity-40 disabled:cursor-not-allowed hover:border-zinc-600 transition-colors"
                      />
                      {/* Resume / retrain hint */}
                      {!isB2 && (
                        <div className="flex flex-col gap-1">
                          {sel && sel.total_epochs_trained > 0 && (
                            <span className="text-[10px] font-mono text-amber-500/80 leading-tight">
                              ↳ resume {sel.total_epochs_trained} → {sel.total_epochs_trained + epochs}
                            </span>
                          )}
                          {sel?.needs_retraining && (
                            <span className="text-[10px] font-mono text-amber-400 leading-tight">
                              ⚠ retrain recommended
                            </span>
                          )}
                        </div>
                      )}
                      {isB2 && (
                        <div className="flex flex-col gap-0.5">
                          <span className="text-[10px] font-mono text-amber-400/80 leading-tight">
                            ◈ Beatrice 2 · CUDA required
                          </span>
                          <span className="text-[10px] font-mono text-zinc-500 leading-tight">
                            min {minB2Steps.toLocaleString()} steps @ bs={batchSize}
                          </span>
                        </div>
                      )}
                    </>
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

            {/* Overtraining Stop + Speaker Loss Weight — RVC only */}
            {(() => {
              const sel2 = profiles.find(p => p.id === selectedId);
              if (sel2?.pipeline === 'beatrice2') return null;
              return (
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

              </div>

              {/* Loss Mode — full-width below the two columns */}
              <div className="flex flex-col gap-2 pt-3 border-t border-zinc-800/60">
                <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
                  Training Objective
                </label>
                <div className="flex flex-col gap-1.5">
                  {([
                    {
                      value: 'classic' as const,
                      label: 'Classic',
                      color: 'text-cyan-400',
                      dot: 'bg-cyan-500',
                      desc: 'Mel + KL + adversarial + feature matching. Standard RVC training — best for most use cases.',
                    },
                    {
                      value: 'combined' as const,
                      label: 'Combined',
                      color: 'text-violet-400',
                      dot: 'bg-violet-500',
                      desc: 'Classic + ECAPA speaker identity loss. Adds explicit speaker similarity pressure on top of standard training.',
                    },
                  ] as const).map(opt => {
                    const active = lossMode === opt.value;
                    return (
                      <button
                        key={opt.value}
                        disabled={isRunning}
                        onClick={() => setLossMode(opt.value)}
                        className={`flex items-start gap-3 rounded-lg px-3 py-2.5 text-left transition-colors border
                          ${active
                            ? 'border-zinc-600 bg-zinc-800/80'
                            : 'border-zinc-800 bg-zinc-900/40 hover:bg-zinc-800/40'
                          } ${isRunning ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                      >
                        {/* Radio dot */}
                        <span className="mt-0.5 w-3.5 h-3.5 rounded-full border-2 shrink-0 flex items-center justify-center
                          border-zinc-500"
                          style={{ borderColor: active ? undefined : undefined }}
                        >
                          {active && <span className={`w-1.5 h-1.5 rounded-full ${opt.dot}`} />}
                        </span>
                        <span className="flex flex-col gap-0.5">
                          <span className={`text-[12px] font-mono font-semibold ${active ? opt.color : 'text-zinc-400'}`}>
                            {opt.label}
                          </span>
                          <span className="text-[10px] font-mono text-zinc-500 leading-relaxed">
                            {opt.desc}
                          </span>
                        </span>
                      </button>
                    );
                  })}
                </div>

                {/* c_spk slider — only visible for combined */}
                {lossMode === 'combined' && (
                  <div className="flex items-center gap-3 mt-1 pl-1">
                    <span className="text-[10px] font-mono text-zinc-500 shrink-0 w-20">
                      c_spk weight
                    </span>
                    <input
                      type="range"
                      min={1} max={5} step={0.5}
                      value={speakerLossWeight}
                      disabled={isRunning}
                      onChange={e => setSpeakerLossWeight(Number(e.target.value))}
                      className="flex-1 accent-pink-500"
                    />
                    <span className="text-[12px] font-mono text-pink-300 w-6 text-right">
                      {speakerLossWeight.toFixed(1)}
                    </span>
                    <span className="text-[10px] font-mono text-zinc-600 shrink-0">
                      recommended 2–3
                    </span>
                  </div>
                )}
              </div>

              {/* Adversarial Loss */}
              <div className="flex flex-col gap-2 pt-3 border-t border-zinc-800/60">
                <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
                  Adversarial Loss
                </label>
                <div className="flex gap-2">
                  {([
                    { value: 'lsgan' as const, label: 'LSGAN', color: 'text-sky-400', desc: 'Least-squares GAN — original RVC. Stable baseline, can stall if discriminator dominates early.' },
                    { value: 'tprls' as const, label: 'TPRLS', color: 'text-amber-400', desc: 'Truncated Paired Relative LS — median-centering reduces mode collapse. Better when training from scratch or at low batch size.' },
                  ] as const).map(opt => {
                    const active = advLoss === opt.value;
                    return (
                      <button
                        key={opt.value}
                        disabled={isRunning}
                        onClick={() => setAdvLoss(opt.value)}
                        className={`flex-1 flex flex-col gap-1 rounded-lg px-3 py-2.5 text-left transition-colors border
                          ${active ? 'border-zinc-600 bg-zinc-800/80' : 'border-zinc-800 bg-zinc-900/40 hover:bg-zinc-800/40'}
                          ${isRunning ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                      >
                        <span className={`text-[12px] font-mono font-semibold ${active ? opt.color : 'text-zinc-400'}`}>{opt.label}</span>
                        <span className="text-[10px] font-mono text-zinc-500 leading-relaxed">{opt.desc}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* KL Annealing */}
              <div className="flex flex-col gap-2 pt-3 border-t border-zinc-800/60">
                <div className="flex items-center justify-between">
                  <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
                    KL Annealing
                  </label>
                  <button
                    disabled={isRunning}
                    onClick={() => setKlAnneal(v => !v)}
                    className={`relative w-9 h-5 rounded-full transition-colors shrink-0
                      ${klAnneal ? 'bg-emerald-600' : 'bg-zinc-700'}
                      ${isRunning ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                  >
                    <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform
                      ${klAnneal ? 'translate-x-4' : 'translate-x-0'}`} />
                  </button>
                </div>
                <span className="text-[10px] font-mono text-zinc-500 leading-relaxed">
                  Ramps KL weight 0→1 over N epochs then repeats. Prevents posterior collapse in early training.
                </span>
                {klAnneal && (
                  <div className="flex items-center gap-3 mt-0.5">
                    <span className="text-[10px] font-mono text-zinc-500 shrink-0 w-20">cycle epochs</span>
                    <input
                      type="range"
                      min={10} max={100} step={5}
                      value={klAnnealEpochs}
                      disabled={isRunning}
                      onChange={e => setKlAnnealEpochs(Number(e.target.value))}
                      className="flex-1 accent-emerald-500"
                    />
                    <span className="text-[12px] font-mono text-emerald-300 w-8 text-right">{klAnnealEpochs}</span>
                    <span className="text-[10px] font-mono text-zinc-600 shrink-0">recommended 30–50</span>
                  </div>
                )}
              </div>

              {/* Optimizer */}
              <div className="flex flex-col gap-2 pt-3 border-t border-zinc-800/60">
                <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
                  Optimizer
                </label>
                <div className="flex gap-2">
                  {([
                    { value: 'adamw' as const, label: 'AdamW', color: 'text-sky-400', desc: 'Standard adaptive optimizer. Best default for most fine-tuning runs.' },
                    { value: 'adamspd' as const, label: 'AdamSPD', color: 'text-violet-400', desc: 'Adam + Selective Projection Decay. Pulls weights toward pretrain anchor when gradient would increase drift. Reduces catastrophic forgetting on short runs.' },
                  ] as const).map(opt => {
                    const active = optimizer === opt.value;
                    return (
                      <button
                        key={opt.value}
                        disabled={isRunning}
                        onClick={() => setOptimizer(opt.value)}
                        className={`flex-1 flex flex-col gap-1 rounded-lg px-3 py-2.5 text-left transition-colors border
                          ${active ? 'border-zinc-600 bg-zinc-800/80' : 'border-zinc-800 bg-zinc-900/40 hover:bg-zinc-800/40'}
                          ${isRunning ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                      >
                        <span className={`text-[12px] font-mono font-semibold ${active ? opt.color : 'text-zinc-400'}`}>{opt.label}</span>
                        <span className="text-[10px] font-mono text-zinc-500 leading-relaxed">{opt.desc}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

            </div>
              ); // end !isB2 conditional
            })()}

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
            <PhaseBar currentPhase={currentPhase} jobDone={jobState === 'done'} pipeline={profiles.find(p => p.id === selectedId)?.pipeline} />
          </section>
        )}

        {(() => {
          const selProfile = profiles.find(p => p.id === selectedId);
          const isB2 = selProfile?.pipeline === 'beatrice2';
          if (isB2 && stepPoints.length > 0) return (
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">Loss Curves</h2>
              <span className="text-[10px] font-mono text-zinc-600">
                {stepPoints.length} step{stepPoints.length !== 1 ? 's' : ''}
                {isRunning ? ` · +${epochs} target` : ''}
              </span>
            </div>
            <div className="bg-zinc-950 rounded-lg p-3 border border-zinc-800">
              <BeatriceStepChart points={stepPoints} />
            </div>
          </section>
          );
          if (!isB2 && epochPoints.length > 0) return (
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
                totalEpochs={(() => {
                  const priorEpochs = selProfile?.total_epochs_trained ?? 0;
                  if (isRunning) {
                    return Math.max(
                      epochPoints.length > 0 ? epochPoints[epochPoints.length - 1].epoch : 0,
                      priorEpochs + epochs,
                    );
                  }
                  return undefined;
                })()}
                bestEpoch={selProfile?.best_epoch ?? null}
              />
            </div>
          </section>
          );
          return null;
        })()}

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
        {(() => {
          const _tipProfile = profiles.find(p => p.id === selectedId);
          const _isB2 = _tipProfile?.pipeline === 'beatrice2';
          if (_isB2) return (
            <TipsPanel tips={[
              {
                icon: '🔢',
                title: 'Minimum steps scale with batch size',
                body: 'The reference recipe is 10 000 steps at batch size 8. For other batch sizes use: min_steps = 10 000 × √(8 ÷ batch_size). Examples: bs 4 → 14 142 steps, bs 8 → 10 000, bs 16 → 7 071, bs 32 → 5 000, bs 64 → 3 536. The UI enforces this minimum automatically as you change batch size.',
              },
              {
                icon: '⚖️',
                title: 'Batch size trades gradient quality for step count',
                body: 'Larger batches produce lower-variance gradient estimates — each step is more reliable but you need fewer of them (√ rule, not linear). bs 32 at 5 000 steps sees 2× more total audio than bs 8 at 10 000 steps, but converges to similar quality. Use the largest batch that fits in VRAM.',
              },
              {
                icon: '📈',
                title: 'Good targets for a typical dataset',
                body: 'Quick test: 500–1 000 steps (bs 8). Usable voice: 2 000–3 000 steps. Good quality: 5 000–8 000 steps. Best quality: 10 000+ steps (bs 8) or 5 000+ steps (bs 32). Watch UTMOS at evaluation checkpoints — if it stops improving over 1 000 steps, you have found the ceiling for your dataset.',
              },
              {
                icon: '🔁',
                title: 'Resuming continues from the last checkpoint',
                body: 'Each resume run adds the requested steps on top of the previous total. Warmup (first 50 % of total steps, capped at 5 000) is computed from the cumulative step count, so the learning rate curve is always consistent regardless of how many times you resume.',
              },
              {
                icon: '🎙️',
                title: 'Audio quality matters more than quantity',
                body: 'Clean, single-speaker, dry (no reverb) audio is critical. The trainer preprocesses audio into 6-second chunks at 16 kHz — very short clips or heavy background noise will degrade convergence. 10–30 minutes of clean speech is a solid starting point.',
              },
              {
                icon: '📉',
                title: 'Mel loss is the key signal',
                body: 'Mel loss (L1 spectrogram error) is the primary quality proxy — lower is better. The adversarial and feature-matching losses stabilise training but plateau early. If mel loss is still decreasing, keep training. If it has been flat for 1 000+ steps, you are at the limit.',
              },
            ]} />
          );
          return (
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
          );
        })()}

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
