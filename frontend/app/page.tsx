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
    training: 'bg-amber-900/50 text-amber-300 animate-pulse',
    trained: 'bg-cyan-900/50 text-cyan-300',
    failed: 'bg-red-900/50 text-red-300',
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider ${cls[status] ?? 'bg-zinc-700 text-zinc-400'}`}>
      {status}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Waveform Canvas + Segment Selector
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
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [peaks, setPeaks] = useState<Float32Array | null>(null);
  const [loading, setLoading] = useState(true);
  const dragging = useRef<'start' | 'end' | 'body' | null>(null);
  const dragStart = useRef(0);
  const dragStartSec = useRef(0);
  const dragEndSec = useRef(0);

  // Decode audio and compute peak waveform data
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
        const samplesPerBin = Math.max(1, Math.floor(channel.length / BINS));
        const out = new Float32Array(BINS);
        for (let i = 0; i < BINS; i++) {
          let max = 0;
          const off = i * samplesPerBin;
          for (let j = 0; j < samplesPerBin; j++) {
            const v = Math.abs(channel[off + j] || 0);
            if (v > max) max = v;
          }
          out[i] = max;
        }
        if (!cancelled) { setPeaks(out); setLoading(false); }
      } catch (err) {
        if (!cancelled) setLoading(false);
      }
    };
    reader.readAsArrayBuffer(file);
    return () => { cancelled = true; };
  }, [file]);

  // Draw waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !peaks) return;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = '#09090b';
    ctx.fillRect(0, 0, W, H);

    // Selection overlay
    const sx = (startSec / duration) * W;
    const ex = (endSec   / duration) * W;
    ctx.fillStyle = 'rgba(8, 145, 178, 0.12)';
    ctx.fillRect(sx, 0, ex - sx, H);

    // Out-of-selection dim
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(0, 0, sx, H);
    ctx.fillRect(ex, 0, W - ex, H);

    // Peaks
    const mid = H / 2;
    for (let i = 0; i < peaks.length; i++) {
      const x = (i / peaks.length) * W;
      const h = peaks[i] * mid * 0.9;
      const inSel = x >= sx && x <= ex;
      ctx.strokeStyle = inSel ? '#22d3ee' : '#3f3f46';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, mid - h);
      ctx.lineTo(x, mid + h);
      ctx.stroke();
    }

    // Selection handles (vertical lines + drag indicators)
    const drawHandle = (xPos: number, color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xPos, 0);
      ctx.lineTo(xPos, H);
      ctx.stroke();
      // diamond grip
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(xPos, H / 2, 6, 0, Math.PI * 2);
      ctx.fill();
    };
    drawHandle(sx, '#06b6d4');
    drawHandle(ex, '#06b6d4');

    // Time labels
    ctx.font = '10px monospace';
    ctx.fillStyle = '#94a3b8';
    const labels = [startSec, endSec];
    labels.forEach((t, idx) => {
      const lx = idx === 0 ? sx + 6 : ex - 42;
      ctx.fillText(fmtDuration(t), Math.max(4, Math.min(W - 44, lx)), 14);
    });

    // Duration label in centre of selection
    const segDur = endSec - startSec;
    const label = `${(segDur / 60).toFixed(1)} min`;
    ctx.font = 'bold 11px monospace';
    ctx.fillStyle = '#22d3ee';
    const cx = sx + (ex - sx) / 2;
    ctx.fillText(label, cx - 18, H / 2 + 4);

  }, [peaks, startSec, endSec, duration]);

  // Pointer event handling for drag
  const xToSec = (clientX: number) => {
    const rect = containerRef.current!.getBoundingClientRect();
    return Math.max(0, Math.min(duration, ((clientX - rect.left) / rect.width) * duration));
  };

  const HANDLE_THRESH_PX = 10;

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (!canvasRef.current) return;
    const rect = containerRef.current!.getBoundingClientRect();
    const W = rect.width;
    const px = e.clientX - rect.left;
    const sx = (startSec / duration) * W;
    const ex = (endSec   / duration) * W;

    if (Math.abs(px - sx) < HANDLE_THRESH_PX) {
      dragging.current = 'start';
    } else if (Math.abs(px - ex) < HANDLE_THRESH_PX) {
      dragging.current = 'end';
    } else if (px > sx && px < ex) {
      dragging.current = 'body';
      dragStart.current = px;
      dragStartSec.current = startSec;
      dragEndSec.current = endSec;
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
      const newStart = Math.min(sec, endSec - MIN_SEG_SEC);
      const newEnd   = Math.min(newStart + MAX_SEG_SEC, duration);
      const clampedEnd = Math.max(newStart + MIN_SEG_SEC, Math.min(endSec, newEnd));
      onRangeChange(Math.max(0, newStart), clampedEnd);

    } else if (dragging.current === 'end') {
      const newEnd   = Math.max(sec, startSec + MIN_SEG_SEC);
      const clamped  = Math.min(newEnd, startSec + MAX_SEG_SEC, duration);
      onRangeChange(startSec, Math.max(startSec + MIN_SEG_SEC, clamped));

    } else if (dragging.current === 'body') {
      const rect = containerRef.current!.getBoundingClientRect();
      const delta = ((e.clientX - rect.left) - dragStart.current) / rect.width * duration;
      const segLen = dragEndSec.current - dragStartSec.current;
      let ns = dragStartSec.current + delta;
      let ne = dragEndSec.current + delta;
      if (ns < 0) { ne = segLen; ns = 0; }
      if (ne > duration) { ns = duration - segLen; ne = duration; }
      onRangeChange(ns, ne);
    }
  }, [startSec, endSec, duration, onRangeChange]);

  const onPointerUp = useCallback(() => {
    dragging.current = null;
  }, []);

  return (
    <div ref={containerRef} className="relative w-full select-none touch-none">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/80 rounded-lg z-10">
          <span className="text-[11px] font-mono text-zinc-400 animate-pulse">Decoding waveform…</span>
        </div>
      )}
      <canvas
        ref={canvasRef}
        width={1200}
        height={80}
        className="w-full h-20 rounded-lg cursor-col-resize"
        style={{ imageRendering: 'pixelated' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      />
      <div className="flex justify-between text-[10px] font-mono text-zinc-600 mt-1 px-0.5">
        <span>0:00</span>
        <span>{fmtDuration(duration / 4)}</span>
        <span>{fmtDuration(duration / 2)}</span>
        <span>{fmtDuration(duration * 3 / 4)}</span>
        <span>{fmtDuration(duration)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload + preprocessing panel (shown inline when a file is selected)
// ---------------------------------------------------------------------------

interface UploadPanelProps {
  onUploaded: () => void;
}

function UploadPanel({ onUploaded }: UploadPanelProps) {
  const [nameInput, setNameInput] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [fileDuration, setFileDuration] = useState<number | null>(null);
  const [durationError, setDurationError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [startSec, setStartSec] = useState(0);
  const [endSec, setEndSec] = useState(0);

  const fileRef = useRef<HTMLInputElement>(null);

  // When a file is selected, decode its duration client-side
  async function onFileChange(f: File | null) {
    setFile(f);
    setFileDuration(null);
    setDurationError(null);
    setStartSec(0);
    setEndSec(0);

    if (!f) return;

    // Check 200 MB hard cap before any decoding
    if (f.size > 200 * 1024 * 1024) {
      setDurationError('File exceeds 200 MB limit');
      return;
    }

    // Decode audio duration client-side via Web Audio API
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

      if (!isFinite(dur) || dur <= 0) {
        setDurationError('Could not determine audio duration');
        return;
      }
      if (dur > 30 * 60) {
        setDurationError(`File is ${(dur / 60).toFixed(1)} min — max is 30 min`);
        return;
      }

      setFileDuration(dur);
      // Default selection: first 10 min (or whole file if < 10 min)
      const defaultEnd = Math.min(dur, MAX_SEG_SEC);
      const defaultStart = Math.max(0, defaultEnd - MIN_SEG_SEC);
      setStartSec(defaultStart);
      setEndSec(defaultEnd);
    } catch (err) {
      URL.revokeObjectURL(url);
      setDurationError(`Duration check failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  function onRangeChange(s: number, e: number) {
    setStartSec(s);
    setEndSec(e);
  }

  const segLen = endSec - startSec;
  const segValid = segLen >= MIN_SEG_SEC && segLen <= MAX_SEG_SEC;

  async function handleUpload(ev: React.FormEvent) {
    ev.preventDefault();
    if (!nameInput.trim() || !file || !fileDuration || durationError) return;

    setUploading(true);
    setError(null);

    const form = new FormData();
    form.append('name', nameInput.trim());
    form.append('file', file);

    try {
      const res = await fetch(`${API}/api/profiles`, { method: 'POST', body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      setNameInput('');
      setFile(null);
      setFileDuration(null);
      setStartSec(0);
      setEndSec(0);
      if (fileRef.current) fileRef.current.value = '';
      onUploaded();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }

  const canSubmit = nameInput.trim().length > 0 && !!file && !durationError && !uploading;

  return (
    <form onSubmit={handleUpload}
      className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5 flex flex-col gap-5">

      {/* Name + file row */}
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
            <span className="ml-2 font-normal text-zinc-600 normal-case tracking-normal">max 30 min</span>
          </label>
          <input
            ref={fileRef} type="file"
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
              {fileDuration != null && ` · ${fmtDuration(fileDuration)}`}
            </span>
          )}
          {durationError && (
            <span className="text-[11px] font-mono text-red-400">{durationError}</span>
          )}
        </div>
      </div>

      {/* Waveform + segment selector */}
      {file && fileDuration != null && !durationError && (
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
              Select Training Segment
              <span className="ml-2 normal-case tracking-normal text-zinc-600">
                10–15 min · drag handles or body to adjust
              </span>
            </span>
            <span className={`text-[11px] font-mono font-semibold ${
              segValid ? 'text-emerald-400' : 'text-amber-400'
            }`}>
              {fmtDuration(startSec)} – {fmtDuration(endSec)}
              {' '}({(segLen / 60).toFixed(1)} min{!segValid ? ' — needs 10–15 min' : ''})
            </span>
          </div>

          <WaveformViewer
            file={file}
            duration={fileDuration}
            startSec={startSec}
            endSec={endSec}
            onRangeChange={onRangeChange}
          />

          {/* Fine-tune sliders */}
          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-mono text-zinc-500">Start</label>
              <input type="range" min={0} max={fileDuration}
                step={1} value={startSec}
                onChange={(e) => {
                  const s = Number(e.target.value);
                  const maxStart = Math.min(fileDuration - MIN_SEG_SEC, endSec - MIN_SEG_SEC);
                  const ns = Math.max(0, Math.min(s, maxStart));
                  const ne = Math.min(endSec, ns + MAX_SEG_SEC, fileDuration);
                  const clampedEnd = Math.max(ns + MIN_SEG_SEC, ne);
                  onRangeChange(ns, Math.min(clampedEnd, fileDuration));
                }}
                className="accent-cyan-500"
              />
              <span className="text-[10px] font-mono text-zinc-500">{fmtDuration(startSec)}</span>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-mono text-zinc-500">End</label>
              <input type="range" min={0} max={fileDuration}
                step={1} value={endSec}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  const minEnd = startSec + MIN_SEG_SEC;
                  const maxEnd = Math.min(startSec + MAX_SEG_SEC, fileDuration);
                  onRangeChange(startSec, Math.max(minEnd, Math.min(v, maxEnd)));
                }}
                className="accent-cyan-500"
              />
              <span className="text-[10px] font-mono text-zinc-500">{fmtDuration(endSec)}</span>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded border border-red-800/60 bg-red-950/30 px-3 py-2 text-[12px] font-mono text-red-300">
          {error}
        </div>
      )}

      <button type="submit" disabled={!canSubmit}
        className="py-2.5 rounded-lg font-mono text-[13px] font-medium tracking-wider uppercase transition-all
                   bg-cyan-900/40 border border-cyan-600/40 text-cyan-300
                   hover:bg-cyan-800/40 hover:border-cyan-500/60
                   disabled:opacity-30 disabled:cursor-not-allowed">
        {uploading ? '⟳  Uploading…' : '↑  Upload Profile'}
      </button>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Profile card with preprocessing controls
// ---------------------------------------------------------------------------

interface ProfileCardProps {
  profile: Profile;
  onDeleted: () => void;
  onRefresh: () => void;
}

function ProfileCard({ profile, onDeleted, onRefresh }: ProfileCardProps) {
  const [preprocessing, setPreprocessing] = useState(false);
  const [preprocessMsg, setPreprocessMsg] = useState<string | null>(null);
  const [preprocessOk, setPreprocessOk] = useState<boolean | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Audio players
  const [playingOrig, setPlayingOrig] = useState(false);
  const [playingClean, setPlayingClean] = useState(false);
  const origAudioRef  = useRef<HTMLAudioElement | null>(null);
  const cleanAudioRef = useRef<HTMLAudioElement | null>(null);

  // Segment state for preprocessing — default to entire file if duration known
  const dur = profile.audio_duration ?? 0;
  const defaultEnd = Math.min(dur, MAX_SEG_SEC);
  const defaultStart = Math.max(0, defaultEnd - MIN_SEG_SEC);
  const [segStart, setSegStart] = useState(defaultStart);
  const [segEnd, setSegEnd]     = useState(defaultEnd);

  const segLen = segEnd - segStart;
  const segValid = dur > 0 && segLen >= MIN_SEG_SEC && segLen <= MAX_SEG_SEC;

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
      // Force reload to get fresh file after preprocessing
      cleanAudioRef.current.src = `${API}/api/profiles/${profile.id}/audio/clean?t=${Date.now()}`;
      cleanAudioRef.current.play().then(() => setPlayingClean(true)).catch(() => {});
    }
  }

  async function handlePreprocess() {
    if (!segValid) return;
    setPreprocessing(true);
    setPreprocessMsg(null);
    setPreprocessOk(null);

    try {
      const res = await fetch(`${API}/api/profiles/${profile.id}/preprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start_sec: segStart, end_sec: segEnd }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      setPreprocessOk(data.ok);
      setPreprocessMsg(data.ok ? `Noise removal complete — ${(data.duration_sec / 60).toFixed(1)} min` : data.message);
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

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 overflow-hidden">
      {/* Header row */}
      <div className="px-5 py-4 flex items-center justify-between gap-4">
        <div className="flex flex-col gap-1 min-w-0">
          <div className="flex items-center gap-3">
            <span className="text-[15px] font-mono font-medium text-zinc-100 truncate">{profile.name}</span>
            <StatusBadge status={profile.status} />
          </div>
          <div className="flex items-center gap-3 text-[11px] font-mono text-zinc-500">
            <span>{profile.id.slice(0, 12)}…</span>
            {profile.audio_duration != null && (
              <span>⏱ {fmtDuration(profile.audio_duration)}</span>
            )}
            {profile.preprocessed_path && (
              <span className="text-emerald-500">✓ cleaned</span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Play original */}
          <button onClick={toggleOrig}
            title="Play original"
            className={`px-2.5 py-1.5 rounded-md text-[11px] font-mono border transition-colors
              ${playingOrig
                ? 'bg-cyan-900/40 border-cyan-600/50 text-cyan-300'
                : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200'}`}>
            {playingOrig ? '■' : '▶'} orig
          </button>

          {/* Play cleaned — only if preprocessed */}
          {profile.preprocessed_path && (
            <button onClick={toggleClean}
              title="Play cleaned audio"
              className={`px-2.5 py-1.5 rounded-md text-[11px] font-mono border transition-colors
                ${playingClean
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

      {/* Preprocessing section */}
      {dur > 0 && (
        <div className="border-t border-zinc-800 px-5 py-4 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
              Noise Removal
              <span className="ml-2 normal-case tracking-normal text-zinc-600">
                select 10–15 min training segment
              </span>
            </span>
            <span className={`text-[10px] font-mono ${segValid ? 'text-emerald-400' : 'text-amber-400'}`}>
              {fmtDuration(segStart)} – {fmtDuration(segEnd)} ({(segLen / 60).toFixed(1)} min)
            </span>
          </div>

          {/* Segment sliders (compact, no waveform needed after upload) */}
          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-mono text-zinc-600">Start</label>
              <input type="range" min={0} max={dur} step={5} value={segStart}
                onChange={(e) => {
                  const s = Number(e.target.value);
                  const ns = Math.min(s, dur - MIN_SEG_SEC);
                  const ne = Math.min(segEnd, ns + MAX_SEG_SEC, dur);
                  setSegStart(ns); setSegEnd(Math.max(ns + MIN_SEG_SEC, ne));
                }}
                className="accent-cyan-500 h-1.5" />
              <span className="text-[10px] font-mono text-zinc-600">{fmtDuration(segStart)}</span>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-mono text-zinc-600">End</label>
              <input type="range" min={0} max={dur} step={5} value={segEnd}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  const ne = Math.max(segStart + MIN_SEG_SEC, Math.min(v, segStart + MAX_SEG_SEC, dur));
                  setSegEnd(ne);
                }}
                className="accent-cyan-500 h-1.5" />
              <span className="text-[10px] font-mono text-zinc-600">{fmtDuration(segEnd)}</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button onClick={handlePreprocess}
              disabled={preprocessing || !segValid}
              className="px-4 py-2 rounded-lg font-mono text-[12px] font-medium uppercase tracking-wide
                         bg-indigo-900/40 border border-indigo-600/40 text-indigo-300
                         hover:bg-indigo-800/40 hover:border-indigo-500/60 transition-all
                         disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-2">
              {preprocessing && (
                <svg className="animate-spin w-3 h-3" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
              )}
              {preprocessing ? 'Removing noise…' : '⬡ Remove Noise'}
            </button>

            {preprocessMsg && (
              <span className={`text-[11px] font-mono ${preprocessOk ? 'text-emerald-400' : 'text-red-400'}`}>
                {preprocessOk ? '✓ ' : '✗ '}{preprocessMsg}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function LibraryPage() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);
  const [sessionActive, setSessionActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  async function fetchProfiles() {
    try {
      const res = await fetch(`${API}/api/profiles`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Profile[] = await res.json();
      setProfiles(data);
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
    const interval = setInterval(async () => {
      if (cancelled) return;
      try {
        const res = await fetch(`${API}/api/realtime/status`);
        if (res.ok) {
          const d: RealtimeStatus = await res.json();
          if (!cancelled) setSessionActive(d.active);
        }
      } catch { /* non-fatal */ }
    }, 3000);
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  function onUploaded() {
    setShowUpload(false);
    fetchProfiles();
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      {sessionActive && (
        <div className="bg-amber-900/40 border-b border-amber-700/50 px-6 py-2 text-[12px] font-mono text-amber-300 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse inline-block"/>
          Realtime voice conversion session is active
        </div>
      )}

      <div className="max-w-4xl mx-auto px-6 py-8 flex flex-col gap-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-mono font-semibold text-zinc-100">Voice Profiles</h1>
            <p className="text-[13px] text-zinc-500 mt-1">
              Upload voice samples, select training segments, and remove background noise.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-[11px] font-mono text-zinc-600">
              {profiles.length} profile{profiles.length !== 1 ? 's' : ''}
            </span>
            <button onClick={() => setShowUpload((v) => !v)}
              className={`px-3 py-1.5 rounded-lg text-[12px] font-mono uppercase tracking-wide border transition-all
                ${showUpload
                  ? 'bg-zinc-800 border-zinc-700 text-zinc-300'
                  : 'bg-cyan-900/40 border-cyan-600/40 text-cyan-300 hover:bg-cyan-800/40'}`}>
              {showUpload ? '✕ Cancel' : '+ New Profile'}
            </button>
          </div>
        </div>

        {error && (
          <div className="rounded-lg border border-red-800/60 bg-red-950/40 px-4 py-3 text-[13px] font-mono text-red-300">
            {error}
          </div>
        )}

        {/* Upload panel */}
        {showUpload && (
          <section>
            <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500 mb-4">
              Upload New Profile
            </h2>
            <UploadPanel onUploaded={onUploaded} />
          </section>
        )}

        {/* Profile list */}
        <section>
          {loading ? (
            <div className="text-center py-16 text-[13px] font-mono text-zinc-500">Loading…</div>
          ) : profiles.length === 0 ? (
            <div className="rounded-xl border border-dashed border-zinc-800 flex items-center justify-center py-16">
              <p className="text-[13px] font-mono text-zinc-500">
                No voice profiles yet — click <span className="text-cyan-400">+ New Profile</span> to add one.
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              {profiles.map((p) => (
                <ProfileCard
                  key={p.id}
                  profile={p}
                  onDeleted={fetchProfiles}
                  onRefresh={fetchProfiles}
                />
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
