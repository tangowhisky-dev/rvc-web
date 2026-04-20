'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { TipsPanel } from '../TipsPanel';
import { ProfilePicker } from '../ProfilePicker';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Profile {
  id: string;
  name: string;
  status: string;
  model_path: string | null;
  index_path: string | null;
  total_epochs_trained: number;
  embedder: string;
  vocoder: string;
  pipeline?: string;
  best_model_path?: string | null;
  best_epoch?: number | null;
  best_avg_gen_loss?: number | null;
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

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function OfflinePage() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [profileId, setProfileId] = useState<string | null>(null);
  const [useBest, setUseBest] = useState(false);

  // Input audio
  const [inputFile, setInputFile]     = useState<File | null>(null);
  const [inputDuration, setInputDuration] = useState(0);

  // Inference params
  const [pitch, setPitch]               = useState(0);
  const [indexRate, setIndexRate]       = useState(0.50);
  const [protect, setProtect]           = useState(0.33);
  const [noiseReduction, setNoiseReduction] = useState(false);
  const [solaCrossfadeMs, setSolaCrossfadeMs] = useState(20);
  // Beatrice 2 params
  const [pitchShift, setPitchShift]           = useState(0);
  const [formantShift, setFormantShift]       = useState(0);

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

  // Post-conversion analysis — navigate to analysis page with context
  const router = useRouter();
  const outputFilePathRef = useRef<string | null>(null);

  // Load profiles — use shared /api/profiles, filter to inference-ready only
  useEffect(() => {
    fetch(`${API}/api/profiles`)
      .then(r => r.json())
      .then((data: Profile[]) => {
        const inferReady = data.filter((p: Profile) => p.status === 'trained');
        setProfiles(inferReady);
        const first = inferReady.find((p: Profile) => p.model_path && p.total_epochs_trained > 0);
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

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setInputFile(f);
    setOutputUrl(null);
    setJobStatus('idle');
    setJobError(null);
    setProgress(0);
    setJobId(null);
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
    form.append('use_best', String(useBest));
    form.append('noise_reduction', String(noiseReduction));
    form.append('sola_crossfade_ms', String(solaCrossfadeMs));
    // Beatrice 2 params (ignored by RVC backend)
    form.append('pitch_shift_semitones', String(pitchShift));
    form.append('formant_shift_semitones', String(formantShift));
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

  function goToAnalysis() {
    if (!jobId) return;
    router.push(`/analysis?from=offline&job_id=${jobId}`);
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
              pipeline: p.pipeline,
              best_model_path: p.best_model_path,
              best_epoch: p.best_epoch,
              best_avg_gen_loss: p.best_avg_gen_loss,
            }))}
            selectedId={profileId}
            onChange={setProfileId}
            emptyMessage="No trained profiles found."
          />
          {/* Use best variant checkbox */}
          <label className="flex items-center gap-2 cursor-pointer select-none mt-1">
            <input
              type="checkbox"
              checked={useBest}
              disabled={!profiles.find(p => p.id === profileId)?.best_model_path || jobStatus === 'running'}
              onChange={(e) => setUseBest(e.target.checked)}
              className="w-4 h-4 rounded border border-zinc-600 bg-zinc-900 accent-amber-500 focus:ring-2 focus:ring-amber-500/50 disabled:opacity-40"
            />
            <span className="text-[11px] font-mono text-zinc-300">
              Use best variant
            </span>
          </label>
        </section>

        {/* Inference params */}
        <section className="flex flex-col gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-900/50">
          <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
            Inference Parameters
          </label>
          {selectedProfile?.pipeline === 'beatrice2' ? (
            <div className="grid grid-cols-2 gap-6">
              <ParamSlider
                label={`Pitch Shift (${pitchShift > 0 ? '+' : ''}${pitchShift} st)`}
                value={pitchShift} min={-12} max={12} step={0.5}
                onChange={setPitchShift} hint="Semitone pitch shift"
              />
              <ParamSlider
                label={`Formant Shift (${formantShift > 0 ? '+' : ''}${formantShift} st)`}
                value={formantShift} min={-3} max={3} step={0.25}
                onChange={setFormantShift} hint="Formant shift (vocal tract)"
              />
            </div>
          ) : (
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
          )}

          {/* Noise Reduction + SOLA Crossfade — side by side */}
          <div className="flex gap-2 items-stretch pt-2 border-t border-zinc-800/60">
            {/* Noise Reduction toggle */}
            <div className="flex-1 flex items-center justify-between px-1 py-2 rounded-lg bg-zinc-900/60 border border-zinc-800">
              <div className="flex flex-col gap-0.5">
                <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">Noise Reduction</span>
                <span className="text-[11px] text-zinc-500">
                  {noiseReduction
                    ? 'RNNoise active — mic, room & fan noise suppressed before conversion'
                    : 'Disabled — raw audio passed to model as-is'}
                </span>
              </div>
              <button
                onClick={() => setNoiseReduction(v => !v)}
                className={`ml-3 shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                  noiseReduction ? 'bg-cyan-500' : 'bg-zinc-700'
                }`}
                aria-label="Toggle noise reduction"
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
                    noiseReduction ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            {/* SOLA Crossfade slider */}
            <div className="flex-1 flex flex-col gap-1.5 px-1 py-2 rounded-lg bg-zinc-900/60 border border-zinc-800">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">SOLA Crossfade</span>
                <span className="text-[11px] font-mono text-zinc-300 tabular-nums">
                  {solaCrossfadeMs === 0 ? 'off' : `${solaCrossfadeMs} ms`}
                </span>
              </div>
              <input
                type="range" min={0} max={50} step={10}
                value={solaCrossfadeMs}
                disabled={jobStatus === 'running'}
                onChange={(e) => setSolaCrossfadeMs(Number(e.target.value))}
                className="w-full h-1 accent-cyan-500 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              />
              <span className="text-[10px] text-zinc-500">
                {solaCrossfadeMs === 0
                  ? 'Disabled — raw block boundary, may click'
                  : `${solaCrossfadeMs}ms overlap-add window. Longer = smoother joins, slightly more latency`}
              </span>
            </div>
          </div>

          {/* Post-conversion analysis toggle */}
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

        {/* Run Analysis button — appears after conversion completes */}
        {outputUrl && jobStatus === 'done' && (
          <div className="flex justify-end">
            <button
              onClick={goToAnalysis}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-[12px] font-mono font-semibold
                bg-violet-700/40 border border-violet-600/50 text-violet-300
                hover:bg-violet-700/60 hover:border-violet-500/70 hover:text-violet-200 transition-all"
            >
              <span>Run Analysis</span>
              <span className="text-[14px]">→</span>
            </button>
          </div>
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
