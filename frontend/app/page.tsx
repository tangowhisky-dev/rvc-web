'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AudioFile {
  id: string;
  profile_id: string;
  filename: string;
  file_path: string;
  duration: number | null;
  is_cleaned: boolean;
  created_at: string;
}

interface Profile {
  id: string;
  name: string;
  status: string;
  created_at: string;
  sample_path: string;
  profile_dir: string | null;
  model_path: string | null;
  checkpoint_path: string | null;
  index_path: string | null;
  total_epochs_trained: number;
  needs_retraining: boolean;
  embedder: string;
  vocoder: string;
  pipeline: string;
  best_model_path: string | null;
  best_epoch: number | null;
  best_avg_gen_loss: number | null;
  audio_files: AudioFile[];
  has_speaker_f0: boolean;
  speaker_mean_f0: number | null;
  speaker_std_f0?: number | null;
  speaker_p5_f0?: number | null;
  speaker_p25_f0?: number | null;
  speaker_p50_f0?: number | null;
  speaker_p75_f0?: number | null;
  speaker_p95_f0?: number | null;
  speaker_vel_std?: number | null;
  speaker_voiced_rate?: number | null;
}

interface HealthStatus {
  profile_id: string;
  audio_ok: boolean;
  model_ok: boolean;
  index_ok: boolean;
  can_train: boolean;
  can_infer: boolean;
  errors: string[];
}

interface RealtimeStatus { active: boolean; }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtDuration(sec: number | null | undefined): string {
  if (sec == null) return '--';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function fmtSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function StatusBadge({ status }: { status: string }) {
  const cls: Record<string, string> = {
    untrained: 'bg-zinc-700 text-zinc-300',
    training:  'bg-amber-900/50 text-amber-300 animate-pulse',
    trained:   'bg-cyan-900/50 text-cyan-300',
    failed:    'bg-red-900/50 text-red-300',
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider ${cls[status] ?? 'bg-zinc-700 text-zinc-400'}`}>
      {status}
    </span>
  );
}

function HealthPill({ ok, label, loading }: { ok: boolean; label: string; loading: boolean }) {
  if (loading) return null;
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono border ${
      ok
        ? 'bg-emerald-950/40 border-emerald-800/50 text-emerald-400'
        : 'bg-red-950/40 border-red-800/50 text-red-400'
    }`}>
      {ok ? '✓' : '✗'} {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Waveform Canvas + dual-handle segment selector + preview playback
// ---------------------------------------------------------------------------

const MIN_SEG_SEC = 10 * 60;  // 10 min
const MAX_SEG_SEC = 30 * 60;  // 30 min

interface WaveformProps {
  file: File;
  duration: number;
  startSec: number;
  endSec: number;
  onRangeChange: (start: number, end: number) => void;
}

function WaveformViewer({ file, duration, startSec, endSec, onRangeChange }: WaveformProps) {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const audioRef     = useRef<HTMLAudioElement | null>(null);
  const rafRef       = useRef<number>(0);

  const [peaks, setPeaks]       = useState<Float32Array | null>(null);
  const [loading, setLoading]   = useState(true);
  const [playing, setPlaying]   = useState(false);
  const [playheadSec, setPlayheadSec] = useState<number | null>(null);

  const dragging        = useRef<'start' | 'end' | 'body' | null>(null);
  const dragOriginPx    = useRef(0);
  const dragOriginStart = useRef(0);
  const dragOriginEnd   = useRef(0);

  // Build object URL for the file once; revoke on unmount or file change
  const objectUrlRef = useRef<string | null>(null);
  useEffect(() => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = URL.createObjectURL(file);
    // Reset audio element when file changes
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = objectUrlRef.current;
      setPlaying(false);
      setPlayheadSec(null);
    }
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    };
  }, [file]);

  // Decode audio + build peak envelope at 8 kHz
  useEffect(() => {
    setLoading(true);
    setPeaks(null);
    let cancelled = false;

    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 8000 });
        const buf = await audioCtx.decodeAudioData(e.target!.result as ArrayBuffer);
        audioCtx.close();
        if (cancelled) return;

        const channel = buf.getChannelData(0);
        const BINS = 1200;
        const spb   = Math.max(1, Math.floor(channel.length / BINS));
        const out   = new Float32Array(BINS);
        for (let i = 0; i < BINS; i++) {
          let max = 0;
          const off = i * spb;
          for (let j = 0; j < spb; j++) {
            const v = Math.abs(channel[off + j] ?? 0);
            if (v > max) max = v;
          }
          out[i] = max;
        }
        if (!cancelled) { setPeaks(out); setLoading(false); }
      } catch { if (!cancelled) setLoading(false); }
    };
    reader.readAsArrayBuffer(file);
    return () => { cancelled = true; };
  }, [file]);

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !peaks) return;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    ctx.fillStyle = '#09090b';
    ctx.fillRect(0, 0, W, H);

    const sx = (startSec / duration) * W;
    const ex = (endSec   / duration) * W;

    // Dim outside selection
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(0,  0, sx, H);
    ctx.fillRect(ex, 0, W - ex, H);

    // Selection tint
    ctx.fillStyle = 'rgba(8,145,178,0.10)';
    ctx.fillRect(sx, 0, ex - sx, H);

    // Peaks
    const mid = H / 2;
    for (let i = 0; i < peaks.length; i++) {
      const x    = (i / peaks.length) * W;
      const h    = peaks[i] * mid * 0.88;
      const inSel = x >= sx && x <= ex;
      ctx.strokeStyle = inSel ? '#22d3ee' : '#3f3f46';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, mid - h);
      ctx.lineTo(x, mid + h);
      ctx.stroke();
    }

    // Selection handles
    const drawHandle = (x: number) => {
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      ctx.fillStyle = '#06b6d4';
      ctx.beginPath(); ctx.arc(x, H / 2, 6, 0, Math.PI * 2); ctx.fill();
    };
    drawHandle(sx);
    drawHandle(ex);

    // Playhead
    if (playheadSec != null) {
      const px = (playheadSec / duration) * W;
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
    }

    // Duration label
    const segDur = endSec - startSec;
    ctx.font = 'bold 11px monospace';
    ctx.fillStyle = '#22d3ee';
    const label = `${(segDur / 60).toFixed(1)} min`;
    ctx.fillText(label, sx + (ex - sx) / 2 - 18, H / 2 + 4);

    // Edge time stamps
    ctx.font = '10px monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(fmtDuration(startSec), Math.max(4, sx + 4),   12);
    ctx.fillText(fmtDuration(endSec),   Math.min(W - 44, ex - 44), 12);
  }, [peaks, startSec, endSec, duration, playheadSec]);

  // Playback controls
  function getOrCreateAudio(): HTMLAudioElement {
    if (!audioRef.current) {
      const a = new Audio();
      a.src = objectUrlRef.current ?? '';
      a.onended = () => { setPlaying(false); setPlayheadSec(null); cancelAnimationFrame(rafRef.current); };
      audioRef.current = a;
    }
    return audioRef.current;
  }

  function tickPlayhead() {
    const a = audioRef.current;
    if (!a || a.paused) return;
    setPlayheadSec(a.currentTime);
    // Stop at endSec
    if (a.currentTime >= endSec) {
      a.pause();
      setPlaying(false);
      setPlayheadSec(null);
      return;
    }
    rafRef.current = requestAnimationFrame(tickPlayhead);
  }

  function togglePlay() {
    const a = getOrCreateAudio();
    if (playing) {
      a.pause();
      setPlaying(false);
      cancelAnimationFrame(rafRef.current);
    } else {
      a.currentTime = startSec;
      a.play().then(() => {
        setPlaying(true);
        rafRef.current = requestAnimationFrame(tickPlayhead);
      }).catch(() => {});
    }
  }

  // Stop on unmount
  useEffect(() => () => {
    audioRef.current?.pause();
    cancelAnimationFrame(rafRef.current);
  }, []);

  // Pointer drag for segment selection
  const xToSec = useCallback((clientX: number) => {
    const rect = containerRef.current!.getBoundingClientRect();
    return Math.max(0, Math.min(duration, ((clientX - rect.left) / rect.width) * duration));
  }, [duration]);

  const THRESH = 10;

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    const rect = containerRef.current!.getBoundingClientRect();
    const W  = rect.width;
    const px = e.clientX - rect.left;
    const sx = (startSec / duration) * W;
    const ex = (endSec   / duration) * W;

    if (Math.abs(px - sx) <= THRESH) {
      dragging.current = 'start';
    } else if (Math.abs(px - ex) <= THRESH) {
      dragging.current = 'end';
    } else if (px > sx && px < ex) {
      dragging.current = 'body';
      dragOriginPx.current    = px;
      dragOriginStart.current = startSec;
      dragOriginEnd.current   = endSec;
    } else {
      return;
    }
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    e.preventDefault();
  }, [startSec, endSec, duration]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return;
    e.preventDefault();
    const sec = xToSec(e.clientX);

    if (dragging.current === 'start') {
      const ns = Math.min(sec, endSec - MIN_SEG_SEC);
      const ne = Math.min(ns + MAX_SEG_SEC, duration, endSec);
      onRangeChange(Math.max(0, ns), Math.max(ns + MIN_SEG_SEC, ne));
    } else if (dragging.current === 'end') {
      const ne = Math.max(sec, startSec + MIN_SEG_SEC);
      onRangeChange(startSec, Math.min(ne, startSec + MAX_SEG_SEC, duration));
    } else {
      const rect = containerRef.current!.getBoundingClientRect();
      const delta = ((e.clientX - rect.left) - dragOriginPx.current) / rect.width * duration;
      const segLen = dragOriginEnd.current - dragOriginStart.current;
      let ns = dragOriginStart.current + delta;
      let ne = dragOriginEnd.current + delta;
      if (ns < 0) { ne = segLen; ns = 0; }
      if (ne > duration) { ns = duration - segLen; ne = duration; }
      onRangeChange(ns, ne);
    }
  }, [startSec, endSec, duration, onRangeChange, xToSec]);

  const onPointerUp = useCallback(() => { dragging.current = null; }, []);

  const segLen   = endSec - startSec;
  const segValid = segLen >= MIN_SEG_SEC && segLen <= MAX_SEG_SEC;

  return (
    <div className="flex flex-col gap-2">
      <div ref={containerRef} className="relative w-full select-none touch-none">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/80 rounded-lg z-10">
            <span className="text-[11px] font-mono text-zinc-400 animate-pulse">Decoding waveform…</span>
          </div>
        )}
        <canvas
          ref={canvasRef}
          width={1200} height={80}
          className="w-full h-20 rounded-lg cursor-col-resize"
          style={{ imageRendering: 'pixelated' }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        />
        {/* Time ruler */}
        <div className="flex justify-between text-[9px] font-mono text-zinc-700 mt-0.5 px-0.5 pointer-events-none">
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <span key={f}>{fmtDuration(duration * f)}</span>
          ))}
        </div>
      </div>

      {/* Controls row: play preview + validity + segment times */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={togglePlay}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-[11px] font-mono
                      transition-colors shrink-0 ${
            playing
              ? 'bg-amber-900/40 border-amber-600/40 text-amber-300'
              : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-600'
          }`}
        >
          {playing ? (
            <><span>■</span><span>Stop</span></>
          ) : (
            <><span>▶</span><span>Preview from {fmtDuration(startSec)}</span></>
          )}
        </button>

        <span className="text-[10px] font-mono text-zinc-600">
          Drag handles or body to select segment
        </span>

        <span className={`ml-auto text-[10px] font-mono font-semibold shrink-0 ${
          segValid ? 'text-emerald-400' : 'text-amber-400'
        }`}>
          {fmtDuration(startSec)} – {fmtDuration(endSec)}
          {' '}({(segLen / 60).toFixed(1)} min
          {!segValid ? ' · needs 10–15 min' : ''})
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// AudioFilePicker — file input + waveform + clip submit
// Reused by both new-profile creation and add-audio-to-profile flows.
// ---------------------------------------------------------------------------

interface AudioFilePickerProps {
  /** If provided, shows a "Profile Name" field and creates a new profile. */
  newProfile?: boolean;
  /** For existing profiles, called with (profileId, file, segStart, segEnd) */
  onSubmit: (args: { name?: string; file: File; segStart: number; segEnd: number; embedder?: string; vocoder?: string; pipeline?: string }) => Promise<void>;
  onCancel: () => void;
  submitLabel?: string;
  showNameField?: boolean;
  /** Show embedder + vocoder + engine selectors (new-profile creation only) */
  showEmbedder?: boolean;
  /** Whether CUDA is available (for Beatrice 2 gating) */
  hasCuda?: boolean;
}

const EMBEDDER_OPTIONS: { value: string; label: string; description: string }[] = [
  { value: 'spin-v2',   label: 'SPIN-v2',        description: 'Best quality · recommended' },
  { value: 'spin',      label: 'SPIN',            description: 'Good quality · faster extraction' },
  { value: 'contentvec',label: 'ContentVec',      description: 'Reliable baseline · widely tested' },
  { value: 'hubert',    label: 'HuBERT (legacy)', description: 'Original fairseq model · fallback only' },
];

const VOCODER_OPTIONS: { value: string; label: string; description: string }[] = [
  { value: 'HiFi-GAN',  label: 'HiFi-GAN',   description: 'Fast · proven · widely supported' },
  { value: 'RefineGAN', label: 'RefineGAN',   description: 'Better speech clarity · slower training' },
];

function AudioFilePicker({ onSubmit, onCancel, submitLabel, showNameField, showEmbedder, hasCuda }: AudioFilePickerProps) {
  const [nameInput, setNameInput]           = useState('');
  const [pipeline, setPipeline]             = useState('rvc');
  const [embedder, setEmbedder]             = useState('spin-v2');
  const [vocoder, setVocoder]               = useState('HiFi-GAN');
  const [file, setFile]                     = useState<File | null>(null);
  const [fileDuration, setFileDuration]     = useState<number | null>(null);
  const [durationError, setDurationError]   = useState<string | null>(null);
  const [uploading, setUploading]           = useState(false);
  const [error, setError]                   = useState<string | null>(null);
  const [startSec, setStartSec]             = useState(0);
  const [endSec, setEndSec]                 = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  async function onFileChange(f: File | null) {
    setFile(f);
    setFileDuration(null);
    setDurationError(null);
    setStartSec(0);
    setEndSec(0);
    if (!f) return;

    if (f.size > 200 * 1024 * 1024) {
      setDurationError('File exceeds 200 MB limit');
      return;
    }

    const url = URL.createObjectURL(f);
    try {
      const audio = new Audio();
      await new Promise<void>((resolve, reject) => {
        audio.src = url;
        audio.preload = 'metadata';
        audio.onloadedmetadata = () => resolve();
        audio.onerror = () => reject(new Error('Could not read audio metadata'));
        setTimeout(() => reject(new Error('Metadata timeout')), 8000);
      });
      const dur = audio.duration;
      URL.revokeObjectURL(url);

      if (!isFinite(dur) || dur <= 0) { setDurationError('Could not determine duration'); return; }
      if (dur > 60 * 60)              { setDurationError(`${(dur / 60).toFixed(1)} min — max is 60 min`); return; }

      setFileDuration(dur);
      const defaultEnd   = Math.min(dur, MAX_SEG_SEC);
      const defaultStart = Math.max(0, defaultEnd - MIN_SEG_SEC);
      setStartSec(defaultStart);
      setEndSec(defaultEnd);
    } catch (err) {
      URL.revokeObjectURL(url);
      setDurationError(`Check failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  const segLen   = endSec - startSec;
  const segValid = segLen >= MIN_SEG_SEC && segLen <= MAX_SEG_SEC;
  const nameOk   = !showNameField || nameInput.trim().length > 0;
  const canSubmit = nameOk && !!file && !durationError && segValid && !uploading;

  async function handleSubmit(ev: React.FormEvent) {
    ev.preventDefault();
    if (!canSubmit || !file) return;
    setUploading(true);
    setError(null);
    try {
      await onSubmit({ name: showNameField ? nameInput.trim() : undefined, file, segStart: startSec, segEnd: endSec, embedder: showEmbedder ? embedder : undefined, vocoder: showEmbedder ? vocoder : undefined, pipeline: showEmbedder ? pipeline : undefined });
      // Reset on success
      setNameInput('');
      setFile(null);
      setFileDuration(null);
      setStartSec(0);
      setEndSec(0);
      if (fileRef.current) fileRef.current.value = '';
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div className={`grid gap-4 ${showNameField ? 'grid-cols-2' : 'grid-cols-1'}`}>
        {showNameField && (
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">Profile Name</label>
            <input
              type="text" value={nameInput} placeholder="e.g. My Voice"
              onChange={(e) => setNameInput(e.target.value)}
              className="bg-zinc-950 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                         font-mono text-zinc-200 placeholder:text-zinc-600
                         focus:outline-none focus:border-cyan-600 transition-colors"
            />
          </div>
        )}
        <div className="flex flex-col gap-1.5">
          <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
            Audio File
            <span className="ml-2 font-normal text-zinc-600 normal-case tracking-normal">
              max 60 min · select 10–30 min segment below
            </span>
          </label>
          <input
            ref={fileRef}
            type="file"
            accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a"
            onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
            className="block text-[13px] font-mono text-zinc-300
                       file:mr-3 file:py-1.5 file:px-3 file:rounded file:border file:border-zinc-700
                       file:bg-zinc-800 file:text-zinc-300 file:text-[12px] file:cursor-pointer
                       hover:file:bg-zinc-700 file:transition-colors cursor-pointer"
          />
          {file && !durationError && (
            <span className="text-[11px] font-mono text-zinc-500">
              {file.name} · {fmtSize(file.size)}
              {fileDuration != null && ` · ${fmtDuration(fileDuration)} total`}
            </span>
          )}
          {durationError && (
            <span className="text-[11px] font-mono text-red-400">{durationError}</span>
          )}
        </div>
      </div>

      {/* Embedder selector — shown only for new profile creation */}
      {showEmbedder && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
              Engine
            </label>
            <span className="text-[10px] font-mono text-amber-500/80 bg-amber-950/30 border border-amber-800/40 px-1.5 py-0.5 rounded">
              locked at creation
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setPipeline('rvc')}
              className={`text-left px-3 py-2 rounded-lg border text-[12px] font-mono transition-colors ${
                pipeline === 'rvc'
                  ? 'bg-indigo-900/30 border-indigo-600/50 text-indigo-300'
                  : 'bg-zinc-900 border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300'
              }`}
            >
              <div className="font-semibold">◈ RVC</div>
              <div className="text-[10px] opacity-70 mt-0.5">Works on MPS · CPU · CUDA · epoch training</div>
            </button>
            <button
              type="button"
              onClick={() => { if (hasCuda) setPipeline('beatrice2'); }}
              disabled={!hasCuda}
              className={`text-left px-3 py-2 rounded-lg border text-[12px] font-mono transition-colors ${
                pipeline === 'beatrice2'
                  ? 'bg-amber-900/30 border-amber-600/50 text-amber-300'
                  : hasCuda
                    ? 'bg-zinc-900 border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300'
                    : 'bg-zinc-900/50 border-zinc-800 text-zinc-600 cursor-not-allowed'
              }`}
            >
              <div className="font-semibold">◈ Beatrice 2</div>
              <div className="text-[10px] opacity-70 mt-0.5">
                {hasCuda ? 'Source-filter vocoder · CUDA · step training' : 'CUDA GPU required — not available'}
              </div>
            </button>
          </div>
        </div>
      )}

      {/* Embedder + vocoder selectors — only for RVC profiles */}
      {showEmbedder && pipeline === 'rvc' && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
              Feature Embedder
            </label>
            <span className="text-[10px] font-mono text-amber-500/80 bg-amber-950/30 border border-amber-800/40 px-1.5 py-0.5 rounded">
              locked after first training run
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {EMBEDDER_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setEmbedder(opt.value)}
                className={`text-left px-3 py-2 rounded-lg border text-[12px] font-mono transition-colors ${
                  embedder === opt.value
                    ? 'bg-cyan-900/30 border-cyan-600/50 text-cyan-300'
                    : 'bg-zinc-900 border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300'
                }`}
              >
                <div className="font-semibold">{opt.label}</div>
                <div className="text-[10px] opacity-70 mt-0.5">{opt.description}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Vocoder selector — shown only for new RVC profile creation */}
      {showEmbedder && pipeline === 'rvc' && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
              Vocoder
            </label>
            <span className="text-[10px] font-mono text-amber-500/80 bg-amber-950/30 border border-amber-800/40 px-1.5 py-0.5 rounded">
              locked after first training run
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {VOCODER_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setVocoder(opt.value)}
                className={`text-left px-3 py-2 rounded-lg border text-[12px] font-mono transition-colors ${
                  vocoder === opt.value
                    ? 'bg-violet-900/30 border-violet-600/50 text-violet-300'
                    : 'bg-zinc-900 border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300'
                }`}
              >
                <div className="font-semibold">{opt.label}</div>
                <div className="text-[10px] opacity-70 mt-0.5">{opt.description}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {file && fileDuration != null && !durationError && (
        <WaveformViewer
          file={file}
          duration={fileDuration}
          startSec={startSec}
          endSec={endSec}
          onRangeChange={(s, e) => { setStartSec(s); setEndSec(e); }}
        />
      )}

      {error && (
        <div className="rounded border border-red-800/60 bg-red-950/30 px-3 py-2 text-[12px] font-mono text-red-300">
          {error}
        </div>
      )}

      <div className="flex items-center gap-3">
        <button type="submit" disabled={!canSubmit}
          className="py-2 px-5 rounded-lg font-mono text-[12px] font-medium tracking-wider uppercase
                     transition-all bg-cyan-900/40 border border-cyan-600/40 text-cyan-300
                     hover:bg-cyan-800/40 hover:border-cyan-500/60
                     disabled:opacity-30 disabled:cursor-not-allowed">
          {uploading ? '⟳ Uploading…' : (submitLabel ?? '↑ Upload & Clip')}
        </button>
        <button type="button" onClick={onCancel}
          className="py-2 px-4 rounded-lg font-mono text-[12px] text-zinc-500
                     hover:text-zinc-300 transition-colors">
          Cancel
        </button>
      </div>
    </form>
  );
}

// ---------------------------------------------------------------------------
// UploadPanel — new profile creation (wraps AudioFilePicker)
// ---------------------------------------------------------------------------

interface UploadPanelProps {
  onUploaded: () => void;
  onCancel: () => void;
}

function UploadPanel({ onUploaded, onCancel }: UploadPanelProps) {
  const [hasCuda, setHasCuda] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/training/hardware`)
      .then(r => r.json())
      .then(d => setHasCuda(Boolean(d.cuda_available)))
      .catch(() => {});
  }, []);

  async function handleSubmit({ name, file, segStart, segEnd, embedder, vocoder, pipeline }: { name?: string; file: File; segStart: number; segEnd: number; embedder?: string; vocoder?: string; pipeline?: string }) {
    const form = new FormData();
    form.append('name', name!);
    form.append('file', file);
    form.append('seg_start', String(segStart));
    form.append('seg_end',   String(segEnd));
    if (pipeline) form.append('pipeline', pipeline);
    if (embedder) form.append('embedder', embedder);
    if (vocoder)  form.append('vocoder',  vocoder);

    const res = await fetch(`${API}/api/profiles`, { method: 'POST', body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(body.detail ?? `HTTP ${res.status}`);
    }
    onUploaded();
  }

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
      <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500 mb-4">New Profile</h2>
      <AudioFilePicker
        showNameField
        showEmbedder
        hasCuda={hasCuda}
        onSubmit={handleSubmit}
        onCancel={onCancel}
        submitLabel="↑ Create Profile"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// AudioFileRow — one row in the per-profile audio list
// ---------------------------------------------------------------------------

interface AudioFileRowProps {
  profileId: string;
  audioFile: AudioFile;
  onDeleted: () => void;
  onCleaned: () => void;
}

function AudioFileRow({ profileId, audioFile, onDeleted, onCleaned }: AudioFileRowProps) {
  const [playing, setPlaying]     = useState(false);
  const [cleaning, setCleaning]   = useState(false);
  const [cleanMsg, setCleanMsg]   = useState<string | null>(null);
  const [cleanOk, setCleanOk]     = useState<boolean | null>(null);
  const [deleting, setDeleting]   = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  function togglePlay() {
    if (!audioRef.current) {
      const a = new Audio(`${API}/api/profiles/${profileId}/audio/${audioFile.id}`);
      a.onended = () => setPlaying(false);
      audioRef.current = a;
    }
    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
    } else {
      audioRef.current.play().then(() => setPlaying(true)).catch(() => {});
    }
  }

  // Stop on unmount
  useEffect(() => () => { audioRef.current?.pause(); }, []);

  async function handleClean() {
    setCleaning(true);
    setCleanMsg(null);
    setCleanOk(null);
    try {
      const res = await fetch(`${API}/api/profiles/${profileId}/audio/${audioFile.id}/clean`, {
        method: 'POST',
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setCleanOk(data.ok);
      setCleanMsg(data.ok
        ? `✓ Done — ${(data.duration_sec / 60).toFixed(1)} min cleaned`
        : `✗ ${data.message}`
      );
      if (data.ok) { onCleaned(); audioRef.current = null; }
    } catch (err) {
      setCleanOk(false);
      setCleanMsg(`✗ ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCleaning(false);
    }
  }

  async function handleDelete() {
    if (!window.confirm(`Delete audio file "${audioFile.filename}"?`)) return;
    setDeleting(true);
    audioRef.current?.pause();
    try {
      const res = await fetch(`${API}/api/profiles/${profileId}/audio/${audioFile.id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      onDeleted();
    } catch (err) {
      setDeleting(false);
      alert(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="flex items-center gap-3 py-2 px-3 rounded-lg bg-zinc-950/40 border border-zinc-800/60">
      {/* File info */}
      <div className="flex-1 min-w-0 flex items-center gap-2">
        <span className="text-[12px] font-mono text-zinc-200 truncate">{audioFile.filename}</span>
        {audioFile.duration != null && (
          <span className="text-[11px] font-mono text-zinc-500 shrink-0">
            {fmtDuration(audioFile.duration)}
          </span>
        )}
        {audioFile.is_cleaned ? (
          <span className="text-[10px] font-mono text-emerald-500 shrink-0">✓ clean</span>
        ) : (
          <span className="text-[10px] font-mono text-zinc-600 shrink-0">raw</span>
        )}
      </div>

      {/* Clean result message */}
      {cleanMsg && (
        <span className={`text-[10px] font-mono shrink-0 ${cleanOk ? 'text-emerald-400' : 'text-red-400'}`}>
          {cleanMsg}
        </span>
      )}

      {/* Actions */}
      <div className="flex items-center gap-1.5 shrink-0">
        <button
          onClick={togglePlay}
          title="Play"
          className={`px-2 py-1 rounded text-[11px] font-mono border transition-colors ${
            playing
              ? 'bg-cyan-900/40 border-cyan-600/50 text-cyan-300'
              : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200'
          }`}
        >
          {playing ? '■' : '▶'}
        </button>

        <button
          onClick={handleClean}
          disabled={cleaning}
          title={audioFile.is_cleaned ? 'Re-run noise removal' : 'Run noise removal (overwrites file)'}
          className="px-2 py-1 rounded text-[11px] font-mono border transition-colors
                     bg-indigo-950/40 border-indigo-800/50 text-indigo-400
                     hover:bg-indigo-900/40 hover:text-indigo-300
                     disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {cleaning ? (
            <svg className="animate-spin w-3 h-3" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
            </svg>
          ) : (audioFile.is_cleaned ? '⟳ re-clean' : '⬡ clean')}
        </button>

        <button
          onClick={handleDelete}
          disabled={deleting}
          title="Delete audio file"
          className="px-2 py-1 rounded text-[11px] font-mono border transition-colors
                     text-red-400 border-red-900/50 bg-red-950/20
                     hover:bg-red-900/30 hover:border-red-800/60
                     disabled:opacity-40"
        >
          {deleting ? '…' : '✕ Remove'}
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// F0Button — compute / recalculate speaker F0 prior stats for a profile
// ---------------------------------------------------------------------------
function F0Button({ profileId, profile, onRefresh }: {
  profileId: string;
  profile: Profile;
  onRefresh: () => void;
}) {
  const [computing, setComputing] = useState(false);

  async function handleCompute() {
    setComputing(true);
    try {
      const res = await fetch(`${API}/api/offline/speaker_f0/${profileId}/compute`, { method: 'POST' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        alert(`F0 compute failed: ${body.detail ?? res.statusText}`);
        return;
      }
      onRefresh();
    } catch (err) {
      alert(String(err));
    } finally {
      setComputing(false);
    }
  }

  const hasFull = profile.speaker_p5_f0 != null;
  const tip = profile.has_speaker_f0
    ? hasFull
      ? `μ=${profile.speaker_mean_f0?.toFixed(1)}Hz  σ=${profile.speaker_std_f0?.toFixed(3)}  range=[${profile.speaker_p5_f0?.toFixed(0)}–${profile.speaker_p95_f0?.toFixed(0)} Hz]  vel=${profile.speaker_vel_std?.toFixed(4)}  voiced=${((profile.speaker_voiced_rate ?? 0) * 100).toFixed(0)}%`
      : `Recalculate — current: ${profile.speaker_mean_f0?.toFixed(1)} Hz (legacy, missing percentiles)`
    : 'Compute F0 prior statistics for auto pitch-shift at inference';

  return (
    <button
      onClick={handleCompute}
      disabled={computing}
      title={tip}
      className="px-3 py-1.5 rounded-md text-[11px] font-mono uppercase tracking-wider
                 border transition-colors disabled:opacity-40
                 text-cyan-400 border-cyan-900/50 bg-cyan-950/20
                 hover:bg-cyan-900/30 hover:border-cyan-800/60">
      {computing ? '⟳' : profile.has_speaker_f0
        ? hasFull
          ? `F0 ${profile.speaker_mean_f0?.toFixed(0)}Hz ✓`
          : `F0 ${profile.speaker_mean_f0?.toFixed(0)}Hz ↺`
        : 'Calc F0'}
    </button>
  );
}

// ProfileCard
// ---------------------------------------------------------------------------

interface ProfileCardProps {
  profile: Profile;
  onDeleted: () => void;
  onRefresh: () => void;
}

function ProfileCard({ profile, onDeleted, onRefresh }: ProfileCardProps) {
  const [deleting, setDeleting]           = useState(false);
  const [exporting, setExporting]         = useState(false);
  const [showAddAudio, setShowAddAudio]   = useState(false);
  const [editField, setEditField]         = useState<'embedder' | 'vocoder' | null>(null);
  const [updating, setUpdating]           = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Health check state — loaded async on mount and when status changes
  const [health, setHealth]             = useState<HealthStatus | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setHealthLoading(true);
    fetch(`${API}/api/profiles/${profile.id}/health`)
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then((h: HealthStatus) => { if (!cancelled) { setHealth(h); setHealthLoading(false); } })
      .catch(() => { if (!cancelled) setHealthLoading(false); });
    return () => { cancelled = true; };
  }, [profile.id, profile.status, profile.audio_files.length]);

  // Close edit menu on outside click
  useEffect(() => {
    if (!editField) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setEditField(null);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [editField]);

  // Keyboard: Escape closes menu
  useEffect(() => {
    if (!editField) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setEditField(null);
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [editField]);

  async function handleUpdateField(field: 'embedder' | 'vocoder', value: string) {
    setUpdating(true);
    try {
      const res = await fetch(`${API}/api/profiles/${profile.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [field]: value }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      onRefresh();
    } catch (err) {
      alert(err instanceof Error ? err.message : String(err));
    } finally {
      setUpdating(false);
      setEditField(null);
    }
  }

  async function handleDelete() {
    if (!window.confirm(`Delete profile "${profile.name}" and all its files? This cannot be undone.`)) return;
    setDeleting(true);
    try {
      const res = await fetch(`${API}/api/profiles/${profile.id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      onDeleted();
    } catch (err) {
      setDeleting(false);
      alert(err instanceof Error ? err.message : String(err));
    }
  }

  function handleExport() {
    setExporting(true);
    // Trigger browser download via a temporary anchor — no fetch() needed since
    // the endpoint streams a file directly.
    const a = document.createElement('a');
    a.href = `${API}/api/profiles/${profile.id}/export`;
    a.download = '';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Give the browser a moment to initiate the download, then reset state.
    setTimeout(() => setExporting(false), 1500);
  }

  async function handleAddAudio({ file, segStart, segEnd }: { name?: string; file: File; segStart: number; segEnd: number }) {
    const form = new FormData();
    form.append('file', file);
    form.append('seg_start', String(segStart));
    form.append('seg_end',   String(segEnd));

    const res = await fetch(`${API}/api/profiles/${profile.id}/audio`, { method: 'POST', body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(body.detail ?? `HTTP ${res.status}`);
    }
    setShowAddAudio(false);
    onRefresh();
  }

  const hasErrors = health && health.errors.length > 0;
  const totalDuration = profile.audio_files.reduce((s, af) => s + (af.duration ?? 0), 0);

  return (
    <div className={`rounded-xl border overflow-hidden ${
      hasErrors
        ? 'border-red-800/60 bg-zinc-900/60'
        : 'border-zinc-800 bg-zinc-900/60'
    }`}>

      {/* ── Header ───────────────────────────────────────────────────── */}
      <div className="px-5 py-4 flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1.5 min-w-0">

          {/* Name + status row */}
          <div className="flex items-center gap-2.5 flex-wrap">
            <span className="text-[15px] font-mono font-medium text-zinc-100 truncate">
              {profile.name}
            </span>
            <StatusBadge status={profile.status} />

            {/* Health pills */}
            {!healthLoading && health && (
              <div className="flex items-center gap-1.5">
                <HealthPill ok={health.can_train} label="train" loading={false} />
                <HealthPill ok={health.can_infer} label="infer" loading={false} />
              </div>
            )}
            {healthLoading && (
              <span className="text-[10px] font-mono text-zinc-600 animate-pulse">checking…</span>
            )}

            {/* Re-training recommended tag */}
            {profile.needs_retraining && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono
                               bg-amber-950/50 border border-amber-700/50 text-amber-400">
                ⚠ re-training recommended
              </span>
            )}
          </div>

          {/* Meta row */}
          <div className="flex flex-col gap-2">
            {/* Full profile ID */}
            <span className="text-[10px] font-mono text-zinc-600 truncate" title={profile.id}>
              {profile.id}
            </span>
            {/* Stats + badges row */}
            <div className="flex items-center gap-2 flex-wrap">
              {profile.audio_files.length > 0 && (
                <span className="text-[11px] font-mono text-zinc-400">
                  {profile.audio_files.length} file{profile.audio_files.length !== 1 ? 's' : ''} · ⏱ {fmtDuration(totalDuration)}
                </span>
              )}
              {profile.total_epochs_trained > 0 && (
                <span className="text-[11px] font-mono text-cyan-600">{profile.total_epochs_trained} epochs</span>
              )}
              {/* Checkpoint badge — best vs final */}
              {profile.best_model_path && profile.best_epoch !== null && (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono
                               bg-amber-950/40 border border-amber-800/40 text-amber-400"
                      title="Best checkpoint by avg generator loss">
                  ⭐ best (epoch {profile.best_epoch}, {profile.best_avg_gen_loss?.toFixed(3) ?? ''})
                </span>
              )}
              {/* Embedder + vocoder badges — RVC only */}
              {profile.pipeline !== 'beatrice2' && (<>
              {/* Embedder badge — clickable if not locked */}
              <span
                className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono
                           bg-indigo-950/40 border border-indigo-800/40 text-indigo-400
                           ${profile.total_epochs_trained === 0 ? 'cursor-pointer hover:border-indigo-600/60 hover:bg-indigo-950/60' : 'cursor-not-allowed opacity-60'}`}
                title={profile.total_epochs_trained > 0 ? 'Locked after training' : 'Click to change embedder'}
                onClick={() => profile.total_epochs_trained === 0 && setEditField(editField === 'embedder' ? null : 'embedder')}
              >
                ◈ {profile.embedder || 'spin-v2'}
                {profile.total_epochs_trained > 0 && (
                  <span className="text-indigo-600" title="Locked after first training run">🔒</span>
                )}
              </span>
              {/* Vocoder badge — clickable if not locked */}
              <span
                className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono
                           bg-violet-950/40 border border-violet-800/40 text-violet-400
                           ${profile.total_epochs_trained === 0 ? 'cursor-pointer hover:border-violet-600/60 hover:bg-violet-950/60' : 'cursor-not-allowed opacity-60'}`}
                title={profile.total_epochs_trained > 0 ? 'Locked after training' : 'Click to change vocoder'}
                onClick={() => profile.total_epochs_trained === 0 && setEditField(editField === 'vocoder' ? null : 'vocoder')}
              >
                ◈ {profile.vocoder || 'HiFi-GAN'}
                {profile.total_epochs_trained > 0 && (
                  <span className="text-violet-600" title="Locked after first training run">🔒</span>
                )}
              </span>
              </>)}
              {/* Beatrice 2 engine badge */}
              {profile.pipeline === 'beatrice2' && (
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono
                               bg-emerald-950/40 border border-emerald-800/40 text-emerald-400">
                  ◈ Beatrice 2
                </span>
              )}
            </div>
            {/* Edit dropdown */}
            {editField && (
              <div
                ref={menuRef}
                className="relative z-50 w-48 rounded-lg border border-zinc-700 bg-zinc-800 shadow-xl overflow-hidden"
              >
                <div className="px-2 py-1.5 text-[9px] font-mono uppercase tracking-widest text-zinc-500 border-b border-zinc-700/60">
                  {editField}
                </div>
                {(editField === 'embedder' ? EMBEDDER_OPTIONS : VOCODER_OPTIONS).map(opt => (
                  <button
                    key={opt.value}
                    type="button"
                    disabled={updating}
                    onClick={() => handleUpdateField(editField, opt.value)}
                    className={`w-full text-left px-3 py-1.5 text-[11px] font-mono transition-colors
                                ${opt.value === (editField === 'embedder' ? profile.embedder : profile.vocoder)
                                  ? 'text-cyan-300 bg-zinc-700/60'
                                  : 'text-zinc-300 hover:bg-zinc-700/40'}
                                disabled:opacity-40`}
                  >
                    {opt.label}
                    {opt.value === (editField === 'embedder' ? profile.embedder : profile.vocoder) && (
                      <span className="ml-2 text-cyan-400">✓</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Delete button */}
        <div className="shrink-0 flex items-center gap-2">
          <button
            onClick={handleExport}
            disabled={exporting}
            title="Export profile as .zip"
            className="px-3 py-1.5 rounded-md text-[11px] font-mono uppercase tracking-wider
                       text-zinc-400 border border-zinc-700/60 bg-zinc-800/40
                       hover:bg-zinc-700/50 hover:text-zinc-200 transition-colors
                       disabled:opacity-40">
            {exporting ? '⟳' : '↑ Export'}
          </button>
          <F0Button profileId={profile.id} profile={profile} onRefresh={onRefresh} />
          <button onClick={handleDelete} disabled={deleting}
            className="px-3 py-1.5 rounded-md text-[11px] font-mono uppercase tracking-wider
                       text-red-400 border border-red-900/50 bg-red-950/20
                       hover:bg-red-900/30 hover:border-red-800/60 transition-colors
                       disabled:opacity-40">
            {deleting ? '…' : 'Delete'}
          </button>
        </div>
      </div>

      {/* ── F0 prior stats strip ─────────────────────────────────────── */}
      {profile.has_speaker_f0 && profile.speaker_p5_f0 != null && (
        <div className="border-t border-zinc-700/40 bg-zinc-900/30 px-5 py-2 flex flex-wrap gap-x-4 gap-y-0.5 text-[10px] font-mono text-zinc-400">
          <span title="Geometric mean F0">μ {profile.speaker_mean_f0?.toFixed(1)} Hz</span>
          <span title="Log-space std (distribution width)">σ {profile.speaker_std_f0?.toFixed(3)}</span>
          <span title="Modal pitch (median)">P50 {profile.speaker_p50_f0?.toFixed(1)} Hz</span>
          <span title="Speaking range (P5–P95)">range [{profile.speaker_p5_f0?.toFixed(0)}–{profile.speaker_p95_f0?.toFixed(0)} Hz]</span>
          <span title="Velocity std (intonation dynamics)">vel {profile.speaker_vel_std?.toFixed(4)}</span>
          <span title="Voiced frame fraction">voiced {((profile.speaker_voiced_rate ?? 0) * 100).toFixed(0)}%</span>
        </div>
      )}

      {/* ── Health error strip ────────────────────────────────────────── */}
      {health && health.errors.length > 0 && (
        <div className="border-t border-red-800/40 bg-red-950/20 px-5 py-2.5 flex flex-col gap-1">
          <span className="text-[10px] font-mono uppercase tracking-widest text-red-400 font-semibold">
            ⚠ Profile integrity issues
          </span>
          {health.errors.map((err, i) => (
            <span key={i} className="text-[11px] font-mono text-red-300/80">· {err}</span>
          ))}
        </div>
      )}

      {/* ── Audio files section ───────────────────────────────────────── */}
      <div className="border-t border-zinc-800 px-5 py-3 flex flex-col gap-2">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
            Audio Files
          </span>
          {!showAddAudio && (
            <button
              onClick={() => setShowAddAudio(true)}
              className="px-2.5 py-1 rounded text-[11px] font-mono border transition-colors
                         bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-600"
            >
              + Add Audio
            </button>
          )}
        </div>

        {profile.audio_files.length === 0 ? (
          <p className="text-[11px] font-mono text-zinc-600 py-1">No audio files — add one to enable training.</p>
        ) : (
          <div className="flex flex-col gap-1.5">
            {profile.audio_files.map((af) => (
              <AudioFileRow
                key={af.id}
                profileId={profile.id}
                audioFile={af}
                onDeleted={onRefresh}
                onCleaned={onRefresh}
              />
            ))}
          </div>
        )}

        {/* Inline Add Audio form */}
        {showAddAudio && (
          <div className="mt-2 rounded-lg border border-zinc-700 bg-zinc-900/60 p-4">
            <p className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-3">Add Audio File</p>
            <AudioFilePicker
              onSubmit={handleAddAudio}
              onCancel={() => setShowAddAudio(false)}
              submitLabel="↑ Upload & Clip"
            />
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ImportModal — picks a .zip, handles name collision, calls /api/profiles/import
// ---------------------------------------------------------------------------

interface ImportModalProps {
  onImported: () => void;
  onCancel: () => void;
}

function ImportModal({ onImported, onCancel }: ImportModalProps) {
  const [file, setFile]               = useState<File | null>(null);
  const [nameOverride, setNameOverride] = useState('');
  const [collisionName, setCollisionName] = useState<string | null>(null);
  const [suggested, setSuggested]     = useState('');
  const [importing, setImporting]     = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setImporting(true);
    setError(null);

    const form = new FormData();
    form.append('file', file);
    if (nameOverride.trim()) form.append('name_override', nameOverride.trim());

    try {
      const res = await fetch(`${API}/api/profiles/import`, { method: 'POST', body: form });

      if (res.status === 409) {
        const body = await res.json();
        const detail = body.detail ?? body;
        setCollisionName(detail.name ?? (nameOverride || file.name));
        setSuggested(detail.suggested ?? '');
        setNameOverride(detail.suggested ?? '');
        setError(`A profile named "${detail.name}" already exists. Rename it below or choose a different name.`);
        setImporting(false);
        return;
      }

      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        const detail = body.detail;
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
      }

      const result = await res.json();
      if (result.warnings?.length) {
        // Non-fatal — show but still proceed
        console.warn('import warnings:', result.warnings);
      }
      onImported();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setImporting(false);
    }
  }

  return (
    <div className="rounded-xl border border-zinc-700 bg-zinc-900/80 p-5 flex flex-col gap-4">
      <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">Import Profile</h2>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        {/* File picker */}
        <div className="flex flex-col gap-1.5">
          <label className="text-[11px] font-mono text-zinc-400">
            Profile zip file
          </label>
          <div className="flex items-center gap-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".zip,application/zip"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0] ?? null;
                setFile(f);
                setError(null);
                setCollisionName(null);
                if (f && !nameOverride) {
                  // Pre-fill name from filename stem
                  const stem = f.name.replace(/\.rvc-profile\.zip$|\.zip$/, '').replace(/_/g, ' ');
                  setNameOverride('');  // let manifest name take priority
                }
              }}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-3 py-1.5 rounded-md text-[11px] font-mono border transition-colors
                         bg-zinc-800 border-zinc-700 text-zinc-300 hover:bg-zinc-700/60">
              Choose .zip
            </button>
            {file && (
              <span className="text-[11px] font-mono text-zinc-400 truncate max-w-[200px]" title={file.name}>
                {file.name} <span className="text-zinc-600">({(file.size / 1024 / 1024).toFixed(0)} MB)</span>
              </span>
            )}
          </div>
        </div>

        {/* Name override — shown when collision is detected or user wants to rename */}
        {(collisionName !== null || nameOverride) && (
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-mono text-zinc-400">
              {collisionName ? 'Rename profile' : 'Profile name (optional)'}
            </label>
            <input
              type="text"
              value={nameOverride}
              onChange={(e) => { setNameOverride(e.target.value); setError(null); }}
              placeholder={suggested || 'Enter a new name…'}
              className="w-full rounded-md bg-zinc-800 border border-zinc-700 px-3 py-1.5
                         text-[12px] font-mono text-zinc-200 placeholder-zinc-600
                         focus:outline-none focus:border-cyan-700"
            />
          </div>
        )}

        {/* Manual rename link when no collision yet */}
        {collisionName === null && !nameOverride && (
          <button
            type="button"
            onClick={() => setNameOverride(' ')}
            className="text-[11px] font-mono text-zinc-600 hover:text-zinc-400 text-left">
            + rename on import (optional)
          </button>
        )}

        {error && (
          <p className="text-[11px] font-mono text-amber-400 rounded bg-amber-950/30 border border-amber-800/40 px-3 py-2">
            {error}
          </p>
        )}

        <div className="flex items-center gap-3 pt-1">
          <button
            type="submit"
            disabled={!file || importing}
            className="px-4 py-1.5 rounded-lg text-[12px] font-mono uppercase tracking-wide border transition-all
                       bg-cyan-900/40 border-cyan-600/40 text-cyan-300 hover:bg-cyan-800/40
                       disabled:opacity-40 disabled:cursor-not-allowed">
            {importing ? '⟳ Importing…' : '↓ Import'}
          </button>
          <button
            type="button"
            onClick={onCancel}
            className="px-3 py-1.5 rounded-lg text-[12px] font-mono border
                       bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors">
            Cancel
          </button>
          <span className="text-[10px] font-mono text-zinc-600 ml-1">
            Creates a new profile — original is unchanged
          </span>
        </div>
      </form>
    </div>
  );
}


// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function LibraryPage() {
  const [profiles, setProfiles]       = useState<Profile[]>([]);
  const [loading, setLoading]         = useState(true);
  const [sessionActive, setSessionActive] = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [showUpload, setShowUpload]   = useState(false);
  const [showImport, setShowImport]   = useState(false);

  async function fetchProfiles() {
    try {
      const res = await fetch(`${API}/api/profiles`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setProfiles(await res.json());
      setError(null);
    } catch (err) {
      setError(
        `Cannot reach backend (${err instanceof Error ? err.message : String(err)}). ` +
        'Start the backend: conda run -n rvc uvicorn backend.app.main:app --reload'
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    fetchProfiles();
    const tid = setInterval(async () => {
      if (cancelled) return;
      try {
        const r = await fetch(`${API}/api/realtime/status`);
        if (r.ok) {
          const d: RealtimeStatus = await r.json();
          if (!cancelled) setSessionActive(d.active);
        }
      } catch { /* non-fatal */ }
    }, 3000);
    return () => { cancelled = true; clearInterval(tid); };
  }, []);

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      {sessionActive && (
        <div className="bg-amber-900/40 border-b border-amber-700/50 px-6 py-2 text-[12px] font-mono text-amber-300 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse inline-block" />
          Realtime voice conversion session is active
        </div>
      )}

      <div className="max-w-4xl mx-auto px-6 py-8 flex flex-col gap-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-mono font-semibold text-zinc-100">Voice Profiles</h1>
            <p className="text-[13px] text-zinc-500 mt-1">
              Each profile holds audio files for one voice. Add files, clean noise, then train.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-[11px] font-mono text-zinc-600">
              {profiles.length} profile{profiles.length !== 1 ? 's' : ''}
            </span>
            <button
              onClick={() => { setShowImport((v) => !v); setShowUpload(false); }}
              className={`px-3 py-1.5 rounded-lg text-[12px] font-mono uppercase tracking-wide border transition-all ${
                showImport
                  ? 'bg-zinc-800 border-zinc-700 text-zinc-300'
                  : 'bg-zinc-800/60 border-zinc-700 text-zinc-400 hover:bg-zinc-700/50 hover:text-zinc-200'
              }`}>
              {showImport ? '✕ Cancel' : '↓ Import'}
            </button>
            <button
              onClick={() => { setShowUpload((v) => !v); setShowImport(false); }}
              className={`px-3 py-1.5 rounded-lg text-[12px] font-mono uppercase tracking-wide border transition-all ${
                showUpload
                  ? 'bg-zinc-800 border-zinc-700 text-zinc-300'
                  : 'bg-cyan-900/40 border-cyan-600/40 text-cyan-300 hover:bg-cyan-800/40'
              }`}>
              {showUpload ? '✕ Cancel' : '+ New Profile'}
            </button>
          </div>
        </div>

        {error && (
          <div className="rounded-lg border border-red-800/60 bg-red-950/40 px-4 py-3 text-[13px] font-mono text-red-300">
            {error}
          </div>
        )}

        {showImport && (
          <section>
            <ImportModal
              onImported={() => { setShowImport(false); fetchProfiles(); }}
              onCancel={() => setShowImport(false)}
            />
          </section>
        )}

        {showUpload && (
          <section>
            <UploadPanel
              onUploaded={() => { setShowUpload(false); fetchProfiles(); }}
              onCancel={() => setShowUpload(false)}
            />
          </section>
        )}

        <section>
          {loading ? (
            <div className="text-center py-16 text-[13px] font-mono text-zinc-500">Loading…</div>
          ) : profiles.length === 0 ? (
            <div className="rounded-xl border border-dashed border-zinc-800 flex items-center justify-center py-16">
              <p className="text-[13px] font-mono text-zinc-500">
                No profiles yet — click <span className="text-cyan-400">+ New Profile</span> to add one.
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              {profiles.map((p) => (
                <ProfileCard key={p.id} profile={p} onDeleted={fetchProfiles} onRefresh={fetchProfiles} />
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
