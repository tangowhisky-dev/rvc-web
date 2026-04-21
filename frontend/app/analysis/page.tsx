'use client';

import { Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer,
  ScatterChart, Scatter, ReferenceLine, ZAxis,
  LineChart, Line, Legend,
} from 'recharts';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AnalysisResult {
  ab_similarity: number;
  profile_a_similarity: number | null;
  profile_b_similarity: number | null;
  improvement: number | null;
  improvement_pct: number | null;
  quality_a: string;
  quality_b: string;
  quality_profile_a: string | null;
  quality_profile_b: string | null;
  summary: string;
  emb_a: number[];
  emb_b: number[];
  emb_profile: number[] | null;
  duration_a: number;
  duration_b: number;
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

function fmtDur(s: number): string {
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}m ${sec}s`;
}

// ---------------------------------------------------------------------------
// PlaybackWaveform — identical to offline page
// ---------------------------------------------------------------------------

interface PlaybackWaveformProps {
  url: string;        // always a server URL here
  duration: number;
  color?: string;
}

function PlaybackWaveform({ url, duration, color = '#22d3ee' }: PlaybackWaveformProps) {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const audioRef     = useRef<HTMLAudioElement | null>(null);
  const rafRef       = useRef<number>(0);

  const [peaks, setPeaks]             = useState<Float32Array | null>(null);
  const [loading, setLoading]         = useState(false);
  const [playing, setPlaying]         = useState(false);
  const [playheadSec, setPlayheadSec] = useState(0);
  const [cursorSec, setCursorSec]     = useState(0);

  useEffect(() => {
    if (!url) return;
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = url;
      setPlaying(false); setPlayheadSec(0); setCursorSec(0);
    }
  }, [url]);

  useEffect(() => {
    if (!url) return;
    setLoading(true); setPeaks(null);
    let cancelled = false;
    const BINS = 1200;

    function buildPeaks(ch: Float32Array): Float32Array {
      const spb = Math.max(1, Math.floor(ch.length / BINS));
      const out = new Float32Array(BINS);
      for (let i = 0; i < BINS; i++) {
        let max = 0;
        const off = i * spb;
        for (let j = 0; j < spb; j++) { const v = Math.abs(ch[off + j] ?? 0); if (v > max) max = v; }
        out[i] = max;
      }
      return out;
    }

    fetch(url)
      .then(r => r.arrayBuffer())
      .then(async ab => {
        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 8000 });
        const buf = await audioCtx.decodeAudioData(ab);
        audioCtx.close();
        if (!cancelled) { setPeaks(buildPeaks(buf.getChannelData(0))); setLoading(false); }
      })
      .catch(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [url]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#09090b'; ctx.fillRect(0, 0, W, H);

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

    if (duration > 0) {
      const cx = (cursorSec / duration) * W;
      ctx.strokeStyle = 'rgba(148,163,184,0.5)'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
      ctx.setLineDash([]);
    }

    if (duration > 0 && playing) {
      const px = (playheadSec / duration) * W;
      ctx.strokeStyle = '#fbbf24'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
    }

    if (duration > 0) {
      ctx.font = '9px monospace'; ctx.fillStyle = '#52525b';
      for (const f of [0, 0.25, 0.5, 0.75, 1]) {
        const x = f * W;
        ctx.fillText(fmtDuration(f * duration), Math.min(x + 2, W - 28), H - 3);
      }
    }
  }, [peaks, playheadSec, cursorSec, duration, playing, color]);

  function getOrCreateAudio(): HTMLAudioElement {
    if (!audioRef.current) {
      const a = new Audio(url);
      a.onended = () => { setPlaying(false); cancelAnimationFrame(rafRef.current); };
      audioRef.current = a;
    }
    return audioRef.current;
  }

  function tickPlayhead() {
    const a = audioRef.current;
    if (!a || a.paused) return;
    setPlayheadSec(a.currentTime);
    rafRef.current = requestAnimationFrame(tickPlayhead);
  }

  function togglePlay() {
    const a = getOrCreateAudio();
    if (playing) {
      a.pause(); setPlaying(false); cancelAnimationFrame(rafRef.current);
    } else {
      a.currentTime = cursorSec;
      a.play().then(() => { setPlaying(true); rafRef.current = requestAnimationFrame(tickPlayhead); }).catch(() => {});
    }
  }

  useEffect(() => () => { audioRef.current?.pause(); cancelAnimationFrame(rafRef.current); }, []);

  const xToSec = useCallback((clientX: number) => {
    const rect = containerRef.current!.getBoundingClientRect();
    return Math.max(0, Math.min(duration, ((clientX - rect.left) / rect.width) * duration));
  }, [duration]);

  const onCanvasClick = useCallback((e: React.MouseEvent) => {
    const sec = xToSec(e.clientX);
    setCursorSec(sec);
    if (audioRef.current && playing) audioRef.current.currentTime = sec;
  }, [xToSec, playing]);

  return (
    <div className="flex flex-col gap-2">
      <div ref={containerRef} className="relative w-full select-none">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/80 rounded-lg z-10">
            <span className="text-[11px] font-mono text-zinc-400 animate-pulse">Decoding waveform…</span>
          </div>
        )}
        <canvas
          ref={canvasRef} width={1200} height={80}
          className="w-full h-20 rounded-lg cursor-crosshair"
          style={{ imageRendering: 'pixelated' }}
          onClick={onCanvasClick}
        />
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={togglePlay}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-[11px] font-mono
                      transition-colors shrink-0 ${
            playing
              ? 'bg-amber-900/40 border-amber-600/40 text-amber-300'
              : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-600'
          }`}
        >
          {playing ? <><span>■</span><span>Stop</span></> : <><span>▶</span><span>Play from {fmtDuration(cursorSec)}</span></>}
        </button>
        <span className="text-[10px] font-mono text-zinc-600">Click waveform to set playback position</span>
        {duration > 0 && (
          <span className="ml-auto text-[10px] font-mono text-zinc-500 shrink-0">
            {fmtDuration(playing ? playheadSec : cursorSec)} / {fmtDuration(duration)}
          </span>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SimilarityGauge
// ---------------------------------------------------------------------------

function SimilarityGauge({ value, label, sub }: { value: number; label: string; sub?: string }) {
  const pct = Math.round(value * 100);
  const color = value > 0.85 ? '#22c55e' : value > 0.65 ? '#f59e0b' : '#ef4444';
  const dashArray = 2 * Math.PI * 40;
  const dashOffset = dashArray * (1 - value);
  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="96" height="96" viewBox="0 0 96 96">
        <circle cx="48" cy="48" r="40" fill="none" stroke="#27272a" strokeWidth="8" />
        <circle cx="48" cy="48" r="40" fill="none" stroke={color} strokeWidth="8"
          strokeDasharray={dashArray} strokeDashoffset={dashOffset}
          strokeLinecap="round" transform="rotate(-90 48 48)"
          style={{ transition: 'stroke-dashoffset 0.6s ease' }} />
        <text x="48" y="52" textAnchor="middle" fill={color}
          fontSize="18" fontFamily="monospace" fontWeight="bold">{pct}%</text>
      </svg>
      <span className="text-[11px] font-mono text-zinc-300">{label}</span>
      {sub && <span className="text-[10px] font-mono text-zinc-500">{sub}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// EmbeddingCharts
// ---------------------------------------------------------------------------

function EmbeddingCharts({ result }: { result: AnalysisResult }) {
  const dimCount = result.emb_a.length;

  const barData = [
    { name: 'A ↔ B (direct)', value: Math.round(result.ab_similarity * 100), fill: '#22d3ee' },
  ];

  const diffData = result.emb_a.map((val, i) => ({ dim: i + 1, diff: val - result.emb_b[i] }));

  const allVals = [...result.emb_a, ...result.emb_b];
  const scatterMin = Math.min(...allVals) - 0.05;
  const scatterMax = Math.max(...allVals) + 0.05;
  const scatterRange = scatterMax - scatterMin;

  const scatterData = result.emb_a.map((val, i) => ({
    x: val, y: result.emb_b[i], dim: i + 1, diff: Math.abs(val - result.emb_b[i]),
  }));

  const diffToColor = (diff: number) => {
    const t = Math.min(diff / (scatterRange * 0.3), 1);
    const r = Math.round(34 + t * (239 - 34));
    const g = Math.round(197 + t * (68 - 197));
    const b = Math.round(94 + t * (68 - 94));
    return `rgb(${r},${g},${b})`;
  };

  const ColoredDot = ({ cx, cy, payload }: any) => (
    <circle cx={cx} cy={cy} r={3.5} fill={diffToColor(payload.diff)} fillOpacity={0.8} />
  );

  const lineData = result.emb_a.map((val, i) => ({ dim: i + 1, A: val, B: result.emb_b[i] }));

  const tooltip = {
    contentStyle: {
      color: '#e5e7eb', backgroundColor: '#18181b',
      border: '1px solid #3f3f46', borderRadius: '6px',
      fontSize: '11px', fontFamily: 'monospace',
    },
  };

  return (
    <div className="flex flex-col gap-8">
      <div>
        <label className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-3 block">Speaker Similarity</label>
        <ResponsiveContainer width="100%" height={80}>
          <BarChart data={barData} layout="vertical" margin={{ left: 10, right: 30, top: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10, fill: '#71717a' }} tickFormatter={v => `${v}%`} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: '#a1a1aa' }} width={120} />
            <RechartsTooltip {...tooltip} formatter={(v: any) => `${v}%`} />
            <Bar dataKey="value" fill="#22d3ee" radius={[0, 4, 4, 0]} barSize={22} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <label className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-1 block">Embedding Difference (A − B)</label>
        <p className="text-[10px] font-mono text-zinc-600 mb-3">Above zero = A higher, below = B higher. Near zero = similar voices.</p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={diffData} margin={{ top: 8, right: 20, bottom: 28, left: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis dataKey="dim" type="number" domain={[1, dimCount]} tick={{ fontSize: 9, fill: '#71717a' }}
              label={{ value: 'Dimension', position: 'bottom', offset: 0, fontSize: 10, fill: '#a1a1aa' }} />
            <YAxis tick={{ fontSize: 9, fill: '#71717a' }} tickFormatter={(v: number) => v.toFixed(2)}
              label={{ value: 'Difference', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 10, fill: '#a1a1aa' }} />
            <RechartsTooltip {...tooltip} formatter={(v: any) => typeof v === 'number' ? v.toFixed(4) : v} />
            <Bar dataKey="diff" fill="#22d3ee" fillOpacity={0.65} barSize={2} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <label className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-1 block">Scatter — A vs B</label>
        <p className="text-[10px] font-mono text-zinc-600 mb-3">Each dot = one of {dimCount} ECAPA-TDNN dimensions. On diagonal = identical. Green = similar, red = different.</p>
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 40, left: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis type="number" dataKey="x" name="A" domain={[scatterMin, scatterMax]}
              tick={{ fontSize: 9, fill: '#71717a' }} tickFormatter={(v: number) => v.toFixed(2)}
              label={{ value: 'File A', position: 'bottom', offset: 0, fontSize: 10, fill: '#a1a1aa' }} />
            <YAxis type="number" dataKey="y" name="B" domain={[scatterMin, scatterMax]}
              tick={{ fontSize: 9, fill: '#71717a' }} tickFormatter={(v: number) => v.toFixed(2)}
              label={{ value: 'File B', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 10, fill: '#a1a1aa' }} />
            <ZAxis range={[16, 16]} />
            <RechartsTooltip {...tooltip} cursor={{ strokeDasharray: '3 3' }}
              formatter={(v: any, name: any) => [typeof v === 'number' ? v.toFixed(4) : v, name === 'x' ? 'A' : 'B']} />
            <ReferenceLine
              segment={[{ x: scatterMin, y: scatterMin }, { x: scatterMax, y: scatterMax }]}
              stroke="#484a40" strokeWidth={1.5} strokeDasharray="6 4" />
            <Scatter data={scatterData} shape={<ColoredDot />} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div>
        <label className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-1 block">Embedding Values — All {dimCount} Dimensions</label>
        <p className="text-[10px] font-mono text-zinc-600 mb-3">Overlapping lines = similar speaker characteristics.</p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={lineData} margin={{ top: 28, right: 20, bottom: 28, left: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis dataKey="dim" type="number" domain={[1, dimCount]} tick={{ fontSize: 9, fill: '#71717a' }}
              label={{ value: 'Dimension', position: 'bottom', offset: 0, fontSize: 10, fill: '#a1a1aa' }} />
            <YAxis tick={{ fontSize: 9, fill: '#71717a' }}
              label={{ value: 'Value', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 10, fill: '#a1a1aa' }} />
            <RechartsTooltip {...tooltip} />
            <Legend verticalAlign="top" height={22} wrapperStyle={{ fontSize: '10px', fontFamily: 'monospace', color: '#a1a1aa' }} />
            <Line type="monotone" dataKey="A" stroke="#22d3ee" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="B" stroke="#a78bfa" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

export default function AnalysisPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-64 text-[12px] font-mono text-zinc-500">Loading…</div>
    }>
      <AnalysisPageInner />
    </Suspense>
  );
}

function AnalysisPageInner() {
  const searchParams = useSearchParams();
  const fromOffline = searchParams.get('from') === 'offline';
  const jobId       = searchParams.get('job_id');
  const offlineProfileId = searchParams.get('profile_id');

  // Server-side refs — used for waveform URLs and analysis calls
  const [refA, setRefA] = useState<string | null>(null);  // analysis upload ref
  const [refB, setRefB] = useState<string | null>(null);
  const [labelA, setLabelA] = useState(
    fromOffline ? 'File A — Reference (Training Audio)' : 'File A — Source / Input'
  );
  const [labelB, setLabelB] = useState(
    fromOffline ? 'File B — Converted Output' : 'File B — Converted / Output'
  );

  // Duration state (loaded from Audio element after URL is set)
  const [durA, setDurA] = useState(0);
  const [durB, setDurB] = useState(0);

  // Upload state — tracks per-slot upload progress
  const [uploadingA, setUploadingA] = useState(false);
  const [uploadingB, setUploadingB] = useState(false);

  const [running, setRunning] = useState(false);
  const [result, setResult]   = useState<AnalysisResult | null>(null);
  const [error, setError]     = useState<string | null>(null);

  // Waveform URLs — constructed from refs or job_id
  const urlA = refA
    ? `${API}/api/analysis/audio/${refA}?slot=a`
    : (jobId && fromOffline ? `${API}/api/offline/reference/${jobId}` : null);

  const urlB = refB
    ? `${API}/api/analysis/audio/${refB}?slot=b`
    : (jobId ? `${API}/api/offline/result/${jobId}` : null);

  // Load duration once we have a URL
  useEffect(() => {
    if (!urlA) return;
    const a = new Audio(urlA);
    a.addEventListener('loadedmetadata', () => setDurA(a.duration));
  }, [urlA]);

  useEffect(() => {
    if (!urlB) return;
    const a = new Audio(urlB);
    a.addEventListener('loadedmetadata', () => setDurB(a.duration));
  }, [urlB]);

  // Auto-run when arriving from offline page with a job_id
  useEffect(() => {
    if (fromOffline && jobId && !result && !running) {
      runFromJob(jobId);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function uploadFile(file: File, slot: 'a' | 'b') {
    const setUploading = slot === 'a' ? setUploadingA : setUploadingB;
    setUploading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('slot', slot);
      const res = await fetch(`${API}/api/analysis/upload`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`Upload failed: HTTP ${res.status}`);
      const data = await res.json();
      if (slot === 'a') { setRefA(data.ref_id); setLabelA(file.name); }
      else              { setRefB(data.ref_id); setLabelB(file.name); }
    } catch (e: any) {
      setError(e.message ?? 'Upload failed');
    } finally {
      setUploading(false);
    }
  }

  async function runFromJob(jid: string) {
    setRunning(true); setError(null); setResult(null);
    try {
      const res = await fetch(`${API}/api/offline/analyze/${jid}`, { method: 'POST' });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d?.detail ?? `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (e: any) {
      setError(e.message ?? 'Analysis failed');
    } finally {
      setRunning(false);
    }
  }

  async function runFromRefs() {
    if (!refA || !refB) return;
    setRunning(true); setError(null); setResult(null);
    try {
      const form = new FormData();
      form.append('ref_a', refA);
      form.append('ref_b', refB);
      const res = await fetch(`${API}/api/analysis/compare`, { method: 'POST', body: form });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d?.detail ?? `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (e: any) {
      setError(e.message ?? 'Analysis failed');
    } finally {
      setRunning(false);
    }
  }

  const canRun = !running && (
    (fromOffline && !!jobId) ||
    (!!refA && !!refB)
  );

  const isOfflineMode = fromOffline && !!jobId;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="max-w-3xl mx-auto px-6 py-8 flex flex-col gap-8">

        {/* Header */}
        <div>
          <h1 className="text-xl font-mono font-semibold text-zinc-100">Voice Analysis</h1>
          <p className="mt-1 text-[12px] font-mono text-zinc-500">
            Compare two audio files using ECAPA-TDNN speaker embeddings (192-dim)
          </p>
        </div>

        {/* Audio Files */}
        <section className="flex flex-col gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-900/50">
          <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
            Audio Files
          </label>

          {isOfflineMode ? (
            /* Offline mode — files already on server, just show waveforms */
            <div className="flex flex-col gap-5">
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">{labelA}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">A</span>
                  {durA > 0 && <span className="ml-auto text-[10px] font-mono text-zinc-500">{fmtDur(durA)}</span>}
                </div>
                {urlA && <PlaybackWaveform url={urlA} duration={durA} color="#22d3ee" />}
              </div>
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">{labelB}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-violet-500/10 border border-violet-500/30 text-violet-400">B</span>
                  {durB > 0 && <span className="ml-auto text-[10px] font-mono text-zinc-500">{fmtDur(durB)}</span>}
                </div>
                {urlB && <PlaybackWaveform url={urlB} duration={durB} color="#a78bfa" />}
              </div>
            </div>
          ) : (
            /* Direct upload mode */
            <div className="flex flex-col gap-5">
              {/* File A */}
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">{refA ? labelA : 'File A — Source / Input'}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">A</span>
                  {durA > 0 && <span className="ml-auto text-[10px] font-mono text-zinc-500">{fmtDur(durA)}</span>}
                </div>
                {urlA ? (
                  <>
                    <PlaybackWaveform url={urlA} duration={durA} color="#22d3ee" />
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono text-zinc-600 truncate">{labelA}</span>
                      <button onClick={() => { setRefA(null); setDurA(0); setResult(null); }}
                        className="text-[10px] font-mono text-zinc-600 hover:text-zinc-400 transition-colors shrink-0">
                        × clear
                      </button>
                    </div>
                  </>
                ) : (
                  <label className={`flex flex-col items-center justify-center gap-2 py-8 rounded-lg
                    border border-dashed transition-colors cursor-pointer
                    ${uploadingA ? 'border-cyan-600/40 bg-cyan-900/10' : 'border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800/30'}`}>
                    {uploadingA ? (
                      <>
                        <span className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                        <span className="text-[11px] font-mono text-cyan-400">Uploading…</span>
                      </>
                    ) : (
                      <>
                        <span className="text-xl text-zinc-600">↑</span>
                        <span className="text-[11px] font-mono text-zinc-500">Drop or click to upload</span>
                        <span className="text-[10px] font-mono text-zinc-600">WAV · MP3 · FLAC · M4A</span>
                      </>
                    )}
                    <input type="file" accept="audio/*" className="hidden"
                      onChange={e => { const f = e.target.files?.[0]; if (f) uploadFile(f, 'a'); e.target.value = ''; }} />
                  </label>
                )}
              </div>

              {/* File B */}
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">{refB ? labelB : 'File B — Converted / Output'}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-violet-500/10 border border-violet-500/30 text-violet-400">B</span>
                  {durB > 0 && <span className="ml-auto text-[10px] font-mono text-zinc-500">{fmtDur(durB)}</span>}
                </div>
                {urlB ? (
                  <>
                    <PlaybackWaveform url={urlB} duration={durB} color="#a78bfa" />
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono text-zinc-600 truncate">{labelB}</span>
                      <button onClick={() => { setRefB(null); setDurB(0); setResult(null); }}
                        className="text-[10px] font-mono text-zinc-600 hover:text-zinc-400 transition-colors shrink-0">
                        × clear
                      </button>
                    </div>
                  </>
                ) : (
                  <label className={`flex flex-col items-center justify-center gap-2 py-8 rounded-lg
                    border border-dashed transition-colors cursor-pointer
                    ${uploadingB ? 'border-violet-600/40 bg-violet-900/10' : 'border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800/30'}`}>
                    {uploadingB ? (
                      <>
                        <span className="w-5 h-5 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                        <span className="text-[11px] font-mono text-violet-400">Uploading…</span>
                      </>
                    ) : (
                      <>
                        <span className="text-xl text-zinc-600">↑</span>
                        <span className="text-[11px] font-mono text-zinc-500">Drop or click to upload</span>
                        <span className="text-[10px] font-mono text-zinc-600">WAV · MP3 · FLAC · M4A</span>
                      </>
                    )}
                    <input type="file" accept="audio/*" className="hidden"
                      onChange={e => { const f = e.target.files?.[0]; if (f) uploadFile(f, 'b'); e.target.value = ''; }} />
                  </label>
                )}
              </div>
            </div>
          )}
        </section>

        {/* Run button */}
        <div className="flex items-center gap-4">
          <button
            onClick={isOfflineMode ? () => runFromJob(jobId!) : runFromRefs}
            disabled={!canRun}
            className="flex-1 py-3 rounded-lg font-mono text-[13px] font-medium tracking-wider uppercase
                       transition-all disabled:opacity-30 disabled:cursor-not-allowed
                       bg-cyan-900/40 border border-cyan-600/40 text-cyan-300
                       hover:bg-cyan-900/60 hover:border-cyan-500/60 enabled:hover:text-cyan-200"
          >
            {running ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-3.5 h-3.5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                Analyzing…
              </span>
            ) : 'Run Analysis'}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="p-3 rounded-lg bg-red-900/30 border border-red-800 text-[11px] font-mono text-red-300">
            ❌ {error}
          </div>
        )}

        {/* Running */}
        {running && (
          <div className="flex items-center justify-center gap-3 py-8 text-[12px] font-mono text-zinc-400">
            <div className="w-5 h-5 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
            Extracting ECAPA-TDNN embeddings — 5–30s depending on audio length…
          </div>
        )}

        {/* Results */}
        {result && !running && (
          <div className="flex flex-col gap-6">
            <section className="flex flex-col gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-900/50">
              <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">Result</label>
              <p className="text-[12px] font-mono text-zinc-200 leading-relaxed">{result.summary}</p>
              <div className="flex flex-wrap gap-8 justify-center pt-2">
                <SimilarityGauge value={result.ab_similarity} label="A ↔ B" sub="Direct comparison" />
              </div>
              <div className="flex gap-4 text-[10px] font-mono text-zinc-600 pt-1">
                <span>A: {fmtDur(result.duration_a)} · {result.emb_a.length} dims</span>
                <span>B: {fmtDur(result.duration_b)}</span>
              </div>
            </section>

            <section className="flex flex-col gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-900/50">
              <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">Embedding Analysis</label>
              <EmbeddingCharts result={result} />
            </section>
          </div>
        )}

        {/* Empty state */}
        {!result && !running && !error && !isOfflineMode && (
          <div className="flex flex-col items-center justify-center gap-3 py-16 text-[11px] font-mono text-zinc-600">
            <span className="text-3xl">🎙</span>
            <span>Upload two audio files to compare speaker embeddings</span>
          </div>
        )}

      </div>
    </div>
  );
}
