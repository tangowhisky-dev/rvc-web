'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { TipsPanel } from '../TipsPanel';
import { ProfilePicker } from '../ProfilePicker';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  ReferenceLine,
  ZAxis,
} from 'recharts';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Profile {
  id: string;
  name: string;
  model_path: string | null;
  index_path: string | null;
  total_epochs_trained: number;
  embedder: string;
  vocoder: string;
}

interface VoiceAnalysis {
  profile_input_similarity: number;
  profile_output_similarity: number;
  input_output_similarity: number;
  improvement: number;
  improvement_pct: number;
  quality_input: string;
  quality_output: string;
  summary: string;
  profile_emb: number[];
  input_emb: number[];
  output_emb: number[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtDuration(sec: number): string {
  if (!isFinite(sec) || sec < 0) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

// ---------------------------------------------------------------------------
// PlaybackWaveform — clean waveform display with click-to-seek cursor.
// No trimming handles. Accepts either a File or a blob URL + duration.
// ---------------------------------------------------------------------------

interface PlaybackWaveformProps {
  /** File object OR already-decoded peaks */
  file?: File;
  /** Object URL (for result audio from server) */
  url?: string;
  duration: number;
  color?: string;      // peak colour, default cyan
  label?: string;      // optional label shown in corner
}

function PlaybackWaveform({ file, url, duration, color = '#22d3ee', label }: PlaybackWaveformProps) {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const audioRef     = useRef<HTMLAudioElement | null>(null);
  const rafRef       = useRef<number>(0);
  const objectUrlRef = useRef<string | null>(null);

  const [peaks, setPeaks]           = useState<Float32Array | null>(null);
  const [loading, setLoading]       = useState(false);
  const [playing, setPlaying]       = useState(false);
  const [playheadSec, setPlayheadSec] = useState(0);
  const [cursorSec, setCursorSec]   = useState(0); // click-to-seek position

  // Resolve audio src
  const audioSrc = url ?? objectUrlRef.current;

  // Build object URL from File
  useEffect(() => {
    if (!file) return;
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = URL.createObjectURL(file);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = objectUrlRef.current;
      setPlaying(false);
      setPlayheadSec(0);
      setCursorSec(0);
    }
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, [file]);

  // Update audio src when url changes
  useEffect(() => {
    if (!url) return;
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = url;
      setPlaying(false);
      setPlayheadSec(0);
      setCursorSec(0);
    }
  }, [url]);

  // Decode peaks
  useEffect(() => {
    const src = file ?? url;
    if (!src) return;
    setLoading(true);
    setPeaks(null);

    let cancelled = false;
    const BINS = 1200;

    if (file) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 8000 });
          const buf = await audioCtx.decodeAudioData(e.target!.result as ArrayBuffer);
          audioCtx.close();
          if (cancelled) return;
          const ch = buf.getChannelData(0);
          const out = buildPeaks(ch, BINS);
          setPeaks(out);
        } catch { /* ignore */ }
        finally { if (!cancelled) setLoading(false); }
      };
      reader.readAsArrayBuffer(file);
    } else if (url) {
      fetch(url)
        .then(r => r.arrayBuffer())
        .then(async ab => {
          const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 8000 });
          const buf = await audioCtx.decodeAudioData(ab);
          audioCtx.close();
          if (cancelled) return;
          const ch = buf.getChannelData(0);
          const out = buildPeaks(ch, BINS);
          setPeaks(out);
          setLoading(false);
        })
        .catch(() => { if (!cancelled) setLoading(false); });
    }
    return () => { cancelled = true; };
  }, [file, url]);

  function buildPeaks(ch: Float32Array, bins: number): Float32Array {
    const spb = Math.max(1, Math.floor(ch.length / bins));
    const out = new Float32Array(bins);
    for (let i = 0; i < bins; i++) {
      let max = 0;
      const off = i * spb;
      for (let j = 0; j < spb; j++) {
        const v = Math.abs(ch[off + j] ?? 0);
        if (v > max) max = v;
      }
      out[i] = max;
    }
    return out;
  }

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#09090b';
    ctx.fillRect(0, 0, W, H);

    if (peaks) {
      const mid = H / 2;
      for (let i = 0; i < peaks.length; i++) {
        const x = (i / peaks.length) * W;
        const h = peaks[i] * mid * 0.88;
        const past = duration > 0 && (i / peaks.length) * duration < playheadSec;
        ctx.strokeStyle = past ? color : '#3f3f46';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, mid - h); ctx.lineTo(x, mid + h); ctx.stroke();
      }
    }

    // Cursor (seek position)
    if (duration > 0) {
      const cx = (cursorSec / duration) * W;
      ctx.strokeStyle = 'rgba(148,163,184,0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Playhead
    if (duration > 0 && playing) {
      const px = (playheadSec / duration) * W;
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
    }

    // Label
    if (label) {
      ctx.font = '10px monospace';
      ctx.fillStyle = 'rgba(148,163,184,0.6)';
      ctx.fillText(label, 6, 14);
    }

    // Time ruler
    if (duration > 0) {
      ctx.font = '9px monospace';
      ctx.fillStyle = '#52525b';
      for (const f of [0, 0.25, 0.5, 0.75, 1]) {
        const x = f * W;
        ctx.fillText(fmtDuration(f * duration), Math.min(x + 2, W - 28), H - 3);
      }
    }
  }, [peaks, playheadSec, cursorSec, duration, playing, color, label]);

  // Playback loop
  function tickPlayhead() {
    const a = audioRef.current;
    if (!a || a.paused) return;
    setPlayheadSec(a.currentTime);
    rafRef.current = requestAnimationFrame(tickPlayhead);
  }

  function getOrCreateAudio(): HTMLAudioElement {
    if (!audioRef.current) {
      const a = new Audio();
      a.src = (file ? objectUrlRef.current : url) ?? '';
      a.onended = () => { setPlaying(false); cancelAnimationFrame(rafRef.current); };
      audioRef.current = a;
    }
    return audioRef.current;
  }

  function togglePlay() {
    const a = getOrCreateAudio();
    if (playing) {
      a.pause();
      setPlaying(false);
      cancelAnimationFrame(rafRef.current);
    } else {
      a.currentTime = cursorSec;
      a.play().then(() => {
        setPlaying(true);
        rafRef.current = requestAnimationFrame(tickPlayhead);
      }).catch(() => {});
    }
  }

  useEffect(() => () => {
    audioRef.current?.pause();
    cancelAnimationFrame(rafRef.current);
  }, []);

  // Click-to-seek
  const xToSec = useCallback((clientX: number) => {
    const rect = containerRef.current!.getBoundingClientRect();
    return Math.max(0, Math.min(duration, ((clientX - rect.left) / rect.width) * duration));
  }, [duration]);

  const onCanvasClick = useCallback((e: React.MouseEvent) => {
    const sec = xToSec(e.clientX);
    setCursorSec(sec);
    if (audioRef.current && playing) {
      audioRef.current.currentTime = sec;
    }
  }, [xToSec, playing]);

  const hasSrc = !!(file || url);

  return (
    <div className="flex flex-col gap-2">
      <div ref={containerRef} className="relative w-full select-none">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/80 rounded-lg z-10">
            <span className="text-[11px] font-mono text-zinc-400 animate-pulse">Decoding waveform…</span>
          </div>
        )}
        {!hasSrc && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/60 rounded-lg z-10">
            <span className="text-[11px] font-mono text-zinc-600">No audio</span>
          </div>
        )}
        <canvas
          ref={canvasRef}
          width={1200} height={80}
          className="w-full h-20 rounded-lg cursor-crosshair"
          style={{ imageRendering: 'pixelated' }}
          onClick={onCanvasClick}
        />
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={togglePlay}
          disabled={!hasSrc}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-[11px] font-mono
                      transition-colors shrink-0 disabled:opacity-30 disabled:cursor-not-allowed ${
            playing
              ? 'bg-amber-900/40 border-amber-600/40 text-amber-300'
              : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-600'
          }`}
        >
          {playing ? <><span>■</span><span>Stop</span></> : <><span>▶</span><span>Play from {fmtDuration(cursorSec)}</span></>}
        </button>
        {hasSrc && (
          <span className="text-[10px] font-mono text-zinc-600">
            Click waveform to set playback position
          </span>
        )}
        {hasSrc && duration > 0 && (
          <span className="ml-auto text-[10px] font-mono text-zinc-500 shrink-0">
            {fmtDuration(playing ? playheadSec : cursorSec)} / {fmtDuration(duration)}
          </span>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Param slider
// ---------------------------------------------------------------------------

function ParamSlider({
  label, value, min, max, step, onChange, hint,
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; hint?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">{label}</label>
        <span className="text-[12px] font-mono text-zinc-300 tabular-nums">{value.toFixed(2)}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full h-1 accent-cyan-500 cursor-pointer"
      />
      {hint && <span className="text-[10px] font-mono text-zinc-600">{hint}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// VoiceAnalysisPanel — speaker embedding comparison with charts
// ---------------------------------------------------------------------------

function qualityColor(quality: string): string {
  switch (quality) {
    case 'Excellent': return '#10b981';
    case 'Good':      return '#22c55e';
    case 'Normal':    return '#f59e0b';
    case 'Bad':       return '#ef4444';
    case 'Very Poor': return '#dc2626';
    default:          return '#6b7280';
  }
}

function VoiceAnalysisPanel({
  analysis,
  loading,
  profileName,
}: {
  analysis: VoiceAnalysis | null;
  loading: boolean;
  profileName: string;
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-6 text-[12px] font-mono text-zinc-400">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
          Analyzing speaker embeddings…
        </div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="text-[11px] font-mono text-zinc-500 py-2 text-center">
        Enable this option before conversion to see analysis results.
      </div>
    );
  }

  // Bar chart data — only Profile↔Input and Profile↔Output
  const barData = [
    { name: 'Profile ↔ Input', value: Math.round(analysis.profile_input_similarity * 100), fill: '#f59e0b' },
    { name: 'Profile ↔ Output', value: Math.round(analysis.profile_output_similarity * 100), fill: '#22c55e' },
  ];

  // Diverging bar chart data — difference (Profile - Output) per dimension
  const diffData = analysis.profile_emb.map((val, i) => ({
    dim: i + 1,
    diff: val - analysis.output_emb[i],
    absDiff: Math.abs(val - analysis.output_emb[i]),
  }));
  const dimCount = analysis.profile_emb.length;

  // Scatter plot data — all 192 dimensions, colored by similarity
  const scatterData = analysis.profile_emb.map((val, i) => {
    const outputVal = analysis.output_emb[i];
    const diff = Math.abs(val - outputVal);
    return {
      x: val,
      y: outputVal,
      dim: i + 1,
      diff,
    };
  });

  // Compute min/max for reference line and axis domain
  const allVals = [...analysis.profile_emb, ...analysis.output_emb];
  const scatterMin = Math.min(...allVals) - 0.05;
  const scatterMax = Math.max(...allVals) + 0.05;
  const scatterRange = scatterMax - scatterMin;

  // Color function: green for similar, red for different
  // Use a tighter threshold so differences are more visible
  const diffToColor = (diff: number) => {
    const t = Math.min(diff / (scatterRange * 0.3), 1); // normalize relative to data range
    const r = Math.round(34 + t * (239 - 34));
    const g = Math.round(197 + t * (68 - 197));
    const b = Math.round(94 + t * (68 - 94));
    return `rgb(${r}, ${g}, ${b})`;
  };

  // Custom dot shape for scatter
  const ColoredDot = ({ cx, cy, payload }: any) => {
    const color = diffToColor(payload.diff);
    return (
      <circle
        cx={cx}
        cy={cy}
        r={4}
        fill={color}
        fillOpacity={0.8}
        stroke={color}
        strokeWidth={1}
      />
    );
  };

  // Line chart data — all 192 dimensions
  const lineData = analysis.profile_emb.map((val, i) => ({
    dim: i + 1,
    Profile: val,
    Output: analysis.output_emb[i],
  }));

  const outputQualityColor = qualityColor(analysis.quality_output);

  return (
    <div className="flex flex-col gap-5 pt-2">
      {/* Quality badge + summary */}
      <div className="flex items-start gap-4">
        <div
          className="px-3 py-1.5 rounded-lg text-[11px] font-mono font-bold uppercase tracking-wider"
          style={{
            backgroundColor: `${outputQualityColor}20`,
            color: outputQualityColor,
            border: `1px solid ${outputQualityColor}40`,
          }}
        >
          {analysis.quality_output}
        </div>
        <p className="text-[11px] font-mono text-zinc-300 leading-relaxed flex-1">
          {analysis.summary}
        </p>
      </div>

      {/* Similarity bars */}
      <div>
        <h3 className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-3">
          Speaker Similarity
        </h3>
        <ResponsiveContainer width="100%" height={140}>
          <BarChart data={barData} layout="vertical" margin={{ left: 15, right: 20, top: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10, fill: '#71717a' }} />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fontSize: 10, fill: '#a1a1aa' }}
              width={120}
            />
            <RechartsTooltip
              formatter={(value) => `${value}%`}
              contentStyle={{
                color: '#e5e7eb',
                backgroundColor: '#18181b',
                border: '1px solid #3f3f46',
                borderRadius: '6px',
                fontSize: '11px',
                fontFamily: 'monospace',
              }}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Diverging bar chart — difference per dimension */}
      <div>
        <h3 className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-3">
          Embedding Differences by Dimension
        </h3>
        <p className="text-[10px] font-mono text-zinc-600 mb-2">
          Shows how much each dimension differs. Bars above zero = Profile higher, below zero = Output higher. Closer to zero = more similar.
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={diffData} margin={{ top: 10, right: 20, bottom: 30, left: 70 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis
              dataKey="dim"
              type="number"
              domain={[1, dimCount]}
              tick={{ fontSize: 9, fill: '#71717a' }}
              label={{ value: 'Dimension Index', position: 'bottom', offset: 0, fontSize: 10, fill: '#a1a1aa' }}
            />
            <YAxis
              tick={{ fontSize: 9, fill: '#71717a' }}
              tickFormatter={(v: number) => v.toFixed(3)}
              label={{ value: 'Difference', angle: -90, position: 'insideLeft', offset: 0, 
                style: { textAnchor: 'middle', dominantBaseline: 'central' },
                fontSize: 10, fill: '#a1a1aa' }}
            />
            <RechartsTooltip
              contentStyle={{
                color: '#e5e7eb',
                backgroundColor: '#18181b',
                border: '1px solid #3f3f46',
                borderRadius: '6px',
                fontSize: '11px',
                fontFamily: 'monospace',
              }}
              formatter={(value: any) => (typeof value === 'number' ? value.toFixed(4) : value)}
            />
            <Bar dataKey="diff" fill="#a78bfa" fillOpacity={0.7} barSize={2} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Scatter plot — Profile vs Output embeddings */}
      <div>
        <h3 className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-3">
          Embedding Scatter (Profile vs Output)
        </h3>
        <p className="text-[10px] font-mono text-zinc-600 mb-2">
          Points near the diagonal line indicate similar dimensions. Color shows how close each dimension matches.
        </p>
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 40, left: 70 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis
                type="number"
                dataKey="x"
                name="Profile"
                domain={[scatterMin, scatterMax]}
                tick={{ fontSize: 9, fill: '#71717a' }}
                tickFormatter={(v: number) => v.toFixed(3)}
                label={{ value: 'Profile Embedding', position: 'bottom', offset: 0, fontSize: 10, fill: '#a1a1aa' }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="Output"
                domain={[scatterMin, scatterMax]}
                tick={{ fontSize: 9, fill: '#71717a' }}
                tickFormatter={(v: number) => v.toFixed(3)}
                label={{ value: 'Output Embedding', angle: -90, position: 'insideLeft', 
                  style: { textAnchor: 'middle', dominantBaseline: 'central' },
                  offset: 0, fontSize: 10, fill: '#a1a1aa' }}
              />
              <ZAxis range={[4, 4]} />
              <RechartsTooltip
                cursor={{ strokeDasharray: '3 3' }}
                formatter={(value, name) => {
                  if (name === 'x') return [`Profile: ${(value as number).toFixed(4)}`, 'Profile'];
                  if (name === 'y') return [`Output: ${(value as number).toFixed(4)}`, 'Output'];
                  if (name === 'diff') return [`Difference: ${(value as number).toFixed(4)}`, 'Diff'];
                  return [value, name];
                }}
                labelFormatter={(label) => `Dimension ${label}`}
                contentStyle={{
                  color: '#e5e7eb',
                  backgroundColor: '#18181b',
                  border: '1px solid #3f3f46',
                  borderRadius: '6px',
                  fontSize: '11px',
                  fontFamily: 'monospace',
                }}
                labelStyle={{ color: '#c9c9cb' }}
                itemStyle={{ color: '#c9c9cb' }}
              />
              {/* Diagonal reference line — y = x */}
              <ReferenceLine
                segment={[
                  { x: scatterMin, y: scatterMin },
                  { x: scatterMax, y: scatterMax },
                ]}
                stroke="#484a40"
                strokeWidth={2}
                strokeDasharray="6 4"
              />
              <Scatter
                name="Dimensions"
                data={scatterData}
                shape={<ColoredDot />}
              />
            </ScatterChart>
          </ResponsiveContainer>
      </div>

      {/* Line chart — all dimensions comparison */}
      <div>
        <h3 className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-3">
          Embedding Values by Dimension (All {dimCount})
        </h3>
        <p className="text-[10px] font-mono text-zinc-600 mb-2">
          Compare embedding values across all 192 dimensions. Matching lines indicate similar speaker characteristics.
        </p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={lineData} margin={{ top: 30, right: 20, bottom: 30, left: 70 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis
              dataKey="dim"
              type="number"
              domain={[1, dimCount]}
              tick={{ fontSize: 9, fill: '#71717a' }}
              label={{ value: 'Dimension Index', position: 'bottom', offset: 0, fontSize: 10, fill: '#a1a1aa' }}
            />
            <YAxis
              tick={{ fontSize: 9, fill: '#71717a' }}
              label={{ value: 'Embedding Value', angle: -90, position: 'insideLeft', 
                style: { textAnchor: 'middle', dominantBaseline: 'central' },
                offset: 0, fontSize: 10, fill: '#a1a1aa' }}
            />
            <RechartsTooltip
              contentStyle={{
                color: '#e5e7eb',
                backgroundColor: '#18181b',
                border: '1px solid #3f3f46',
                borderRadius: '6px',
                fontSize: '11px',
                fontFamily: 'monospace',
              }}
            />
            <Line
              type="monotone"
              dataKey="Profile"
              stroke="#f59e0b"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="Output"
              stroke="#a78bfa"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Legend
              align="center"
              verticalAlign="top"
              height={20}
              wrapperStyle={{ fontSize: '10px', fontFamily: 'monospace', color: '#a1a1aa' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function OfflinePage() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [profileId, setProfileId] = useState<string | null>(null);

  // Input audio
  const [inputFile, setInputFile]     = useState<File | null>(null);
  const [inputDuration, setInputDuration] = useState(0);

  // Inference params
  const [pitch, setPitch]           = useState(0);
  const [indexRate, setIndexRate]   = useState(0.50);
  const [protect, setProtect]       = useState(0.33);

  // Job state
  const [jobStatus, setJobStatus]   = useState<'idle' | 'running' | 'done' | 'error'>('idle');
  const [progress, setProgress]     = useState(0);
  const [jobError, setJobError]     = useState<string | null>(null);
  const [jobId, setJobId]           = useState<string | null>(null);

  // Output audio
  const [outputUrl, setOutputUrl]     = useState<string | null>(null);
  const [outputDuration, setOutputDuration] = useState(0);
  const [outputFilename, setOutputFilename] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  // Post-conversion analysis
  const [analyzeEnabled, setAnalyzeEnabled] = useState(false);
  const [analysis, setAnalysis]           = useState<VoiceAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const outputFilePathRef = useRef<string | null>(null);

  // Load profiles
  useEffect(() => {
    fetch(`${API}/api/offline/profiles`)
      .then(r => r.json())
      .then((data: Profile[]) => {
        setProfiles(data);
        const first = data.find(p => p.model_path && p.total_epochs_trained > 0);
        if (first) setProfileId(first.id);
      })
      .catch(() => {});
  }, []);

  // Get audio duration from File
  useEffect(() => {
    if (!inputFile) { setInputDuration(0); return; }
    const url = URL.createObjectURL(inputFile);
    const a = new Audio(url);
    a.onloadedmetadata = () => {
      setInputDuration(a.duration);
      URL.revokeObjectURL(url);
    };
    a.onerror = () => URL.revokeObjectURL(url);
  }, [inputFile]);

  // Get output duration from blob URL
  useEffect(() => {
    if (!outputUrl) { setOutputDuration(0); return; }
    const a = new Audio(outputUrl);
    a.onloadedmetadata = () => setOutputDuration(a.duration);
  }, [outputUrl]);

  // Run analysis when output is ready and enabled, OR when analysis is enabled after output is ready
  useEffect(() => {
    if (!outputUrl || !profileId || !analyzeEnabled || !outputFilePathRef.current) return;
    runAnalysis();
  }, [outputUrl, analyzeEnabled]);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setInputFile(f);
    setOutputUrl(null);
    setJobStatus('idle');
    setJobError(null);
    setProgress(0);
    setJobId(null);
    setAnalysis(null);
    outputFilePathRef.current = null;
  }

  async function handleConvert() {
    if (!inputFile || !profileId) return;
    setJobStatus('running');
    setProgress(0);
    setJobError(null);
    setOutputUrl(null);
    setJobId(null);

    const form = new FormData();
    form.append('profile_id', profileId);
    form.append('pitch', String(pitch));
    form.append('index_rate', String(indexRate));
    form.append('protect', String(protect));
    form.append('file', inputFile);

    const abort = new AbortController();
    abortRef.current = abort;

    try {
      const res = await fetch(`${API}/api/offline/convert`, {
        method: 'POST',
        body: form,
        signal: abort.signal,
      });

      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => res.statusText);
        setJobStatus('error');
        setJobError(text);
        return;
      }

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const msg = JSON.parse(line.slice(6));
            if (msg.type === 'progress') {
              setProgress(Math.round(msg.fraction * 100));
            } else             if (msg.type === 'done') {
              setJobId(msg.job_id);
              setJobStatus('done');
              setProgress(100);
              // Fetch result as blob for in-browser playback
              const dlRes = await fetch(`${API}/api/offline/result/${msg.job_id}`);
              if (dlRes.ok) {
                const blob = await dlRes.blob();
                const blobUrl = URL.createObjectURL(blob);
                setOutputUrl(blobUrl);
                // Derive output filename from input file, preserving extension
                const cd = dlRes.headers.get('content-disposition') ?? '';
                const m = cd.match(/filename="?([^"]+)"?/);
                if (m?.[1]) {
                  setOutputFilename(m[1]);
                } else if (inputFile) {
                  const stem = inputFile.name.replace(/\.[^.]+$/, '');
                  const ext  = inputFile.name.match(/\.[^.]+$/)?.[0] ?? '.wav';
                  setOutputFilename(`${stem}_rvc${ext}`);
                } else {
                  setOutputFilename('output_rvc.wav');
                }
                // Store the server path for analysis
                outputFilePathRef.current = msg.result_path || null;
              }
            } else if (msg.type === 'error') {
              setJobStatus('error');
              setJobError(msg.message);
            }
          } catch { /* ignore parse errors */ }
        }
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        setJobStatus('error');
        setJobError(String(err));
      }
    }
  }

  function handleCancel() {
    abortRef.current?.abort();
    setJobStatus('idle');
    setProgress(0);
  }

  function handleDownload() {
    if (!outputUrl || !outputFilename) return;
    const a = document.createElement('a');
    a.href = outputUrl;
    a.download = outputFilename;
    a.click();
  }

  async function runAnalysis() {
    if (!profileId || !outputFilePathRef.current || !inputFile) return;
    setAnalysisLoading(true);
    try {
      // Upload input file to temp location
      const inputForm = new FormData();
      inputForm.append('file', inputFile);
      const uploadRes = await fetch(`${API}/api/offline/upload-temp`, {
        method: 'POST',
        body: inputForm,
      });
      if (!uploadRes.ok) {
        const errText = await uploadRes.text().catch(() => '');
        throw new Error(`Upload failed: ${uploadRes.status} ${errText}`);
      }
      const uploadData = await uploadRes.json();
      const inputPath = uploadData.path;

      // Run analysis
      const res = await fetch(`${API}/api/offline/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_id: profileId,
          input_audio_path: inputPath,
          output_audio_path: outputFilePathRef.current,
        }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail || `Analysis failed: ${res.status}`);
      }
      const data = await res.json();
      setAnalysis(data);
    } catch (err: any) {
      console.error('Analysis error:', err);
      // Set a minimal error state so user sees something
      setAnalysis({
        profile_input_similarity: 0,
        profile_output_similarity: 0,
        input_output_similarity: 0,
        improvement: 0,
        improvement_pct: 0,
        quality_input: 'Bad',
        quality_output: 'Bad',
        summary: `Analysis failed: ${err.message || 'Unknown error'}. Make sure the backend is running and ECAPA model is available.`,
        profile_emb: new Array(192).fill(0),
        input_emb: new Array(192).fill(0),
        output_emb: new Array(192).fill(0),
      });
    } finally {
      setAnalysisLoading(false);
    }
  }

  const selectedProfile = profiles.find(p => p.id === profileId);
  const canConvert = !!inputFile && !!profileId && !!selectedProfile?.model_path && jobStatus !== 'running';

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="max-w-3xl mx-auto px-6 py-8 flex flex-col gap-8">

        {/* Header */}
        <div>
          <h1 className="text-[18px] font-mono font-semibold text-zinc-100 tracking-tight">
            Offline Inference
          </h1>
          <p className="text-[12px] font-mono text-zinc-500 mt-1">
            Convert an audio file using a trained voice profile. Same pipeline as realtime.
          </p>
        </div>

        {/* Profile selector */}
        <section className="flex flex-col gap-3 p-4 rounded-xl border border-zinc-800 bg-zinc-900/50">
          <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
            Voice Profile
          </label>
          <ProfilePicker
            profiles={profiles.map(p => ({
              id: p.id,
              name: p.name,
              total_epochs_trained: p.total_epochs_trained,
              embedder: p.embedder,
              vocoder: p.vocoder,
            }))}
            selectedId={profileId}
            onChange={setProfileId}
            emptyMessage="No trained profiles found."
          />
        </section>

        {/* Inference params */}
        <section className="flex flex-col gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-900/50">
          <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
            Inference Parameters
          </label>
          <div className="grid grid-cols-3 gap-6">
            <ParamSlider
              label="Pitch" value={pitch} min={-24} max={24} step={1}
              onChange={setPitch} hint="Semitones shift"
            />
            <ParamSlider
              label="Index Rate" value={indexRate} min={0} max={1} step={0.01}
              onChange={setIndexRate} hint="FAISS retrieval blend"
            />
            <ParamSlider
              label="Protect" value={protect} min={0} max={0.5} step={0.01}
              onChange={setProtect} hint="Consonant preservation"
            />
          </div>

          {/* Post-conversion analysis checkbox */}
          <div className="flex items-center gap-3 pt-2 border-t border-zinc-800/60">
            <input
              type="checkbox"
              id="analyze"
              checked={analyzeEnabled}
              onChange={e => { setAnalyzeEnabled(e.target.checked); if (!e.target.checked) setAnalysis(null); }}
              className="w-4 h-4 rounded border-zinc-600 bg-zinc-800 text-cyan-500 focus:ring-cyan-500/30"
            />
            <label htmlFor="analyze" className="text-[12px] font-mono text-zinc-300 cursor-pointer select-none">
              Post-Conversion Analysis
            </label>
            <span className="text-[10px] font-mono text-zinc-500">
              Compare speaker embeddings to verify voice conversion quality
            </span>
          </div>
        </section>

        {/* Input audio */}
        <section className="flex flex-col gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-900/50">
          <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
            Input Audio
          </label>

          {/* File picker */}
          <label className="flex items-center gap-3 cursor-pointer">
            <div className="flex-1 px-3 py-2 rounded-lg border border-dashed border-zinc-700
                            bg-zinc-900 text-[12px] font-mono text-zinc-400
                            hover:border-zinc-500 hover:text-zinc-300 transition-colors">
              {inputFile ? inputFile.name : 'Click to select audio file…'}
            </div>
            <input
              type="file"
              accept="audio/*"
              className="sr-only"
              onChange={handleFileChange}
            />
            {inputFile && (
              <button
                onClick={() => { setInputFile(null); setOutputUrl(null); setJobStatus('idle'); }}
                className="px-2 py-1 text-[11px] font-mono text-zinc-500 hover:text-zinc-300"
              >✕</button>
            )}
          </label>

          {/* Input waveform */}
          {inputFile && (
            <PlaybackWaveform
              file={inputFile}
              duration={inputDuration}
              color="#22d3ee"
              label="INPUT"
            />
          )}
        </section>

        {/* Convert button + progress */}
        <section className="flex flex-col gap-3">
          <div className="flex items-center gap-3">
            {jobStatus !== 'running' ? (
              <button
                onClick={handleConvert}
                disabled={!canConvert}
                className="flex-1 py-3 rounded-lg font-mono text-[13px] font-medium tracking-wider uppercase
                           transition-all disabled:opacity-30 disabled:cursor-not-allowed
                           bg-cyan-900/40 border border-cyan-600/40 text-cyan-300
                           hover:bg-cyan-900/60 hover:border-cyan-500/60 enabled:hover:text-cyan-200"
              >
                Convert
              </button>
            ) : (
              <button
                onClick={handleCancel}
                className="flex-1 py-3 rounded-lg font-mono text-[13px] font-medium tracking-wider uppercase
                           bg-red-900/30 border border-red-700/40 text-red-400
                           hover:bg-red-900/50 transition-all"
              >
                Cancel
              </button>
            )}
          </div>

          {/* Progress bar */}
          {(jobStatus === 'running' || jobStatus === 'done') && (
            <div className="flex flex-col gap-1">
              <div className="w-full h-2 rounded-full bg-zinc-800 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    jobStatus === 'done' ? 'bg-emerald-500' : 'bg-cyan-500'
                  }`}
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] font-mono text-zinc-500">
                <span>{jobStatus === 'done' ? '✓ Done' : `Processing… ${progress}%`}</span>
                <span>{progress}%</span>
              </div>
            </div>
          )}

          {jobStatus === 'error' && (
            <div className="px-3 py-2 rounded-lg bg-red-950/40 border border-red-800/40 text-red-400
                            text-[11px] font-mono">
              {jobError}
            </div>
          )}
        </section>

        {/* Output audio */}
        {outputUrl && (
          <section className="flex flex-col gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-900/50">
            <div className="flex items-center justify-between">
              <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
                Output Audio
              </label>
              <button
                onClick={handleDownload}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-[11px] font-mono
                           bg-emerald-900/30 border-emerald-700/40 text-emerald-400
                           hover:bg-emerald-900/50 hover:border-emerald-600/50 transition-colors"
              >
                <span>↓</span>
                <span>Download {outputFilename || 'output'}</span>
              </button>
            </div>

            <PlaybackWaveform
              url={outputUrl}
              duration={outputDuration}
              color="#a78bfa"
              label="OUTPUT"
            />
          </section>
        )}

        {/* Post-conversion analysis results */}
        {analyzeEnabled && outputUrl && (
          <section className="flex flex-col gap-3 p-4 rounded-xl border border-zinc-800 bg-zinc-900/50">
            <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
              Voice Conversion Analysis
            </label>
            <VoiceAnalysisPanel
              analysis={analysis}
              loading={analysisLoading}
              profileName={selectedProfile?.name ?? 'Profile'}
            />
          </section>
        )}

        {/* Tips */}
        <TipsPanel tips={[
          {
            icon: '🎛️',
            title: 'Start with defaults',
            body: 'pitch=0, index_rate=0.50, protect=0.33 match the realtime defaults. If the voice character sounds off, raise index_rate toward 1.0. If consonants sound mangled, lower protect toward 0.',
          },
          {
            icon: '⏱️',
            title: 'First ~2 s may sound slightly degraded',
            body: 'The rolling context buffer starts cold (silence), so feature and pitch estimation improves as it fills. This is identical behaviour to realtime mode.',
          },
          {
            icon: '📏',
            title: 'Use files of at least 10 s for meaningful results',
            body: 'Very short clips (< 5 s) give the model insufficient audio context. 10–30 s clips are ideal for diagnosing cloning quality without long wait times.',
          },
          {
            icon: '🎤',
            title: 'Record in the same conditions as training audio',
            body: 'Microphone type, room acoustics, and recording level all affect how well the voice model transfers. Mismatched conditions are the most common cause of poor output.',
          },
          {
            icon: '🔊',
            title: 'Normalise your input',
            body: 'Audio that is too quiet or too loud stresses the model. Aim for peaks around −3 dBFS before converting.',
          },
        ]} />

      </div>
    </div>
  );
}
