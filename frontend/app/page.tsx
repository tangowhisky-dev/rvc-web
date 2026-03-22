'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

const API = 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Profile {
  id: string;
  name: string;
  status: string;
  created_at: string;
  sample_path: string;
  audio_duration: number | null;
  preprocessed_path: string | null;
  model_path: string | null;
  index_path: string | null;
  profile_dir: string | null;
}

interface HealthStatus {
  profile_id: string;
  audio_ok: boolean;
  preprocessed_ok: boolean;
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
const MAX_SEG_SEC = 15 * 60;  // 15 min

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
      // Start from startSec
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
        {/* Play/pause preview from startSec */}
        <button
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
// Upload panel
// ---------------------------------------------------------------------------

interface UploadPanelProps {
  onUploaded: () => void;
}

function UploadPanel({ onUploaded }: UploadPanelProps) {
  const [nameInput, setNameInput]   = useState('');
  const [file, setFile]             = useState<File | null>(null);
  const [fileDuration, setFileDuration] = useState<number | null>(null);
  const [durationError, setDurationError] = useState<string | null>(null);
  const [uploading, setUploading]   = useState(false);
  const [error, setError]           = useState<string | null>(null);

  const [startSec, setStartSec] = useState(0);
  const [endSec,   setEndSec]   = useState(0);

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
      if (dur > 30 * 60)              { setDurationError(`${(dur / 60).toFixed(1)} min — max is 30 min`); return; }

      setFileDuration(dur);
      // Default: last 15 min (or whole file if shorter), anchored at start if < 15 min
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

  async function handleUpload(ev: React.FormEvent) {
    ev.preventDefault();
    if (!nameInput.trim() || !file || !fileDuration || durationError || !segValid) return;

    setUploading(true);
    setError(null);

    const form = new FormData();
    form.append('name', nameInput.trim());
    form.append('file', file);
    form.append('seg_start', String(startSec));
    form.append('seg_end',   String(endSec));

    try {
      const res = await fetch(`${API}/api/profiles`, { method: 'POST', body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      setNameInput('');
      setFile(null);
      setFileDuration(null);
      setStartSec(0); setEndSec(0);
      if (fileRef.current) fileRef.current.value = '';
      onUploaded();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }

  const canSubmit = nameInput.trim().length > 0 && !!file && !durationError && segValid && !uploading;

  return (
    <form onSubmit={handleUpload}
      className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5 flex flex-col gap-5">

      <div className="grid grid-cols-2 gap-4">
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
        <div className="flex flex-col gap-1.5">
          <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
            Audio Sample
            <span className="ml-2 font-normal text-zinc-600 normal-case tracking-normal">
              max 30 min · select 10–15 min segment below
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

      {/* Waveform — canvas drag handles are the only segment selection surface */}
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

      <button type="submit" disabled={!canSubmit}
        className="py-2.5 rounded-lg font-mono text-[13px] font-medium tracking-wider uppercase
                   transition-all bg-cyan-900/40 border border-cyan-600/40 text-cyan-300
                   hover:bg-cyan-800/40 hover:border-cyan-500/60
                   disabled:opacity-30 disabled:cursor-not-allowed">
        {uploading ? '⟳  Uploading…' : '↑  Upload & Clip to Selected Segment'}
      </button>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Profile card
// ---------------------------------------------------------------------------

interface ProfileCardProps {
  profile: Profile;
  onDeleted: () => void;
  onRefresh: () => void;
}

function ProfileCard({ profile, onDeleted, onRefresh }: ProfileCardProps) {
  const [preprocessing, setPreprocessing] = useState(false);
  const [preprocessMsg, setPreprocessMsg] = useState<string | null>(null);
  const [preprocessOk, setPreprocessOk]   = useState<boolean | null>(null);
  const [deleting, setDeleting]           = useState(false);

  const [playingOrig,  setPlayingOrig]  = useState(false);
  const [playingClean, setPlayingClean] = useState(false);
  const origAudioRef  = useRef<HTMLAudioElement | null>(null);
  const cleanAudioRef = useRef<HTMLAudioElement | null>(null);

  // Health check state — loaded async on mount
  const [health, setHealth]           = useState<HealthStatus | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setHealthLoading(true);
    fetch(`${API}/api/profiles/${profile.id}/health`)
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then((h: HealthStatus) => { if (!cancelled) { setHealth(h); setHealthLoading(false); } })
      .catch(() => { if (!cancelled) setHealthLoading(false); });
    return () => { cancelled = true; };
  }, [profile.id, profile.status]);

  function stopAllAudio() {
    origAudioRef.current?.pause();
    cleanAudioRef.current?.pause();
    setPlayingOrig(false);
    setPlayingClean(false);
  }

  function toggleOrig() {
    if (!origAudioRef.current) {
      const a = new Audio(`${API}/api/profiles/${profile.id}/audio`);
      a.onended = () => setPlayingOrig(false);
      origAudioRef.current = a;
    }
    if (playingOrig) {
      origAudioRef.current.pause();
      setPlayingOrig(false);
    } else {
      stopAllAudio();
      origAudioRef.current.play().then(() => setPlayingOrig(true)).catch(() => {});
    }
  }

  function toggleClean() {
    if (!cleanAudioRef.current) {
      const a = new Audio(`${API}/api/profiles/${profile.id}/audio/clean`);
      a.onended = () => setPlayingClean(false);
      cleanAudioRef.current = a;
    }
    if (playingClean) {
      cleanAudioRef.current.pause();
      setPlayingClean(false);
    } else {
      stopAllAudio();
      cleanAudioRef.current.src = `${API}/api/profiles/${profile.id}/audio/clean?t=${Date.now()}`;
      cleanAudioRef.current.play().then(() => setPlayingClean(true)).catch(() => {});
    }
  }

  async function handlePreprocess() {
    setPreprocessing(true);
    setPreprocessMsg(null);
    setPreprocessOk(null);
    try {
      const res = await fetch(`${API}/api/profiles/${profile.id}/preprocess`, {
        method: 'POST',
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setPreprocessOk(data.ok);
      setPreprocessMsg(
        data.ok
          ? `Done — ${(data.duration_sec / 60).toFixed(1)} min cleaned`
          : data.message
      );
      if (data.ok) { onRefresh(); cleanAudioRef.current = null; }
    } catch (err) {
      setPreprocessOk(false);
      setPreprocessMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setPreprocessing(false);
    }
  }

  async function handleDelete() {
    if (!window.confirm(`Delete profile "${profile.name}"? This cannot be undone.`)) return;
    setDeleting(true);
    stopAllAudio();
    try {
      const res = await fetch(`${API}/api/profiles/${profile.id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      onDeleted();
    } catch (err) {
      setDeleting(false);
      alert(err instanceof Error ? err.message : String(err));
    }
  }

  // Derived health display
  const hasErrors = health && health.errors.length > 0;
  const isCorrupted = health && !health.can_train && !health.can_infer && profile.status !== 'untrained';

  return (
    <div className={`rounded-xl border overflow-hidden ${
      hasErrors
        ? 'border-red-800/60 bg-zinc-900/60'
        : 'border-zinc-800 bg-zinc-900/60'
    }`}>
      {/* Header */}
      <div className="px-5 py-4 flex items-center justify-between gap-4">
        <div className="flex flex-col gap-1 min-w-0">
          <div className="flex items-center gap-3">
            <span className="text-[15px] font-mono font-medium text-zinc-100 truncate">
              {profile.name}
            </span>
            <StatusBadge status={profile.status} />

            {/* Health indicators */}
            {!healthLoading && health && (
              <div className="flex items-center gap-1.5">
                <HealthPill ok={health.can_train} label="train" loading={false} />
                <HealthPill ok={health.can_infer} label="infer" loading={false} />
              </div>
            )}
            {healthLoading && (
              <span className="text-[10px] font-mono text-zinc-600 animate-pulse">checking…</span>
            )}
          </div>
          <div className="flex items-center gap-3 text-[11px] font-mono text-zinc-500">
            <span>{profile.id.slice(0, 12)}…</span>
            {profile.audio_duration != null && (
              <span>⏱ {fmtDuration(profile.audio_duration)}</span>
            )}
            {profile.preprocessed_path && (
              <span className="text-emerald-500">✓ noise removed</span>
            )}
            {profile.profile_dir && (
              <span title={profile.profile_dir} className="text-zinc-600 max-w-[200px] truncate">
                📁 {profile.profile_dir.split('/').slice(-2).join('/')}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button onClick={toggleOrig} title="Play clipped sample"
            className={`px-2.5 py-1.5 rounded-md text-[11px] font-mono border transition-colors ${
              playingOrig
                ? 'bg-cyan-900/40 border-cyan-600/50 text-cyan-300'
                : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200'}`}>
            {playingOrig ? '■' : '▶'} orig
          </button>

          {profile.preprocessed_path && (
            <button onClick={toggleClean} title="Play noise-removed audio"
              className={`px-2.5 py-1.5 rounded-md text-[11px] font-mono border transition-colors ${
                playingClean
                  ? 'bg-emerald-900/40 border-emerald-600/50 text-emerald-300'
                  : 'bg-zinc-800 border-zinc-700 text-emerald-500 hover:text-emerald-300'}`}>
              {playingClean ? '■' : '▶'} clean
            </button>
          )}

          <button onClick={handleDelete} disabled={deleting}
            className="px-3 py-1.5 rounded-md text-[11px] font-mono uppercase tracking-wider
                       text-red-400 border border-red-900/50 bg-red-950/20
                       hover:bg-red-900/30 hover:border-red-800/60 transition-colors
                       disabled:opacity-40">
            Delete
          </button>
        </div>
      </div>

      {/* Health error strip — shown when files are missing */}
      {health && health.errors.length > 0 && (
        <div className="border-t border-red-800/40 bg-red-950/20 px-5 py-2.5 flex flex-col gap-1">
          <span className="text-[10px] font-mono uppercase tracking-widest text-red-400 font-semibold">
            ⚠ Profile integrity issues
          </span>
          {health.errors.map((err, i) => (
            <span key={i} className="text-[11px] font-mono text-red-300/80">
              · {err}
            </span>
          ))}
          {!health.can_train && (
            <span className="text-[10px] font-mono text-zinc-500 mt-0.5">
              Training unavailable — audio file missing.
              Re-upload audio to restore this profile.
            </span>
          )}
          {health.can_train && !health.can_infer && (
            <span className="text-[10px] font-mono text-zinc-500 mt-0.5">
              Inference unavailable — model or index missing. Run training to generate them.
            </span>
          )}
        </div>
      )}

      {/* Noise removal strip */}
      <div className="border-t border-zinc-800 px-5 py-3 flex items-center gap-4">
        <button
          onClick={handlePreprocess}
          disabled={preprocessing || (health !== null && !health.can_train)}
          title={health && !health.can_train ? 'Audio file missing — cannot preprocess' : undefined}
          className="px-4 py-2 rounded-lg font-mono text-[12px] font-medium uppercase tracking-wide
                     bg-indigo-900/40 border border-indigo-600/40 text-indigo-300
                     hover:bg-indigo-800/40 hover:border-indigo-500/60 transition-all
                     disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {preprocessing && (
            <svg className="animate-spin w-3 h-3 shrink-0" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
            </svg>
          )}
          {preprocessing ? 'Removing noise…' : profile.preprocessed_path ? '⬡ Re-run Noise Removal' : '⬡ Remove Noise'}
        </button>

        {preprocessMsg ? (
          <span className={`text-[11px] font-mono ${preprocessOk ? 'text-emerald-400' : 'text-red-400'}`}>
            {preprocessOk ? '✓ ' : '✗ '}{preprocessMsg}
          </span>
        ) : (
          <span className="text-[10px] font-mono text-zinc-600">
            Spectral noise gating on the clipped sample
          </span>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function LibraryPage() {
  const [profiles, setProfiles]     = useState<Profile[]>([]);
  const [loading, setLoading]       = useState(true);
  const [sessionActive, setSessionActive] = useState(false);
  const [error, setError]           = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);

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

  function onUploaded() {
    setShowUpload(false);
    fetchProfiles();
  }

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
              Upload a sample, select 10–15 min segment, upload to clip it, then optionally remove noise.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-[11px] font-mono text-zinc-600">
              {profiles.length} profile{profiles.length !== 1 ? 's' : ''}
            </span>
            <button
              onClick={() => setShowUpload((v) => !v)}
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

        {showUpload && (
          <section>
            <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500 mb-4">
              Upload New Profile
            </h2>
            <UploadPanel onUploaded={onUploaded} />
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
