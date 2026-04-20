'use client';

import { useEffect, useRef, useState } from 'react';

// ---------------------------------------------------------------------------
// ProfilePicker — custom dropdown with library-style badges on each row.
// Each item shows: name · epoch count · indigo embedder badge · violet vocoder badge
// ---------------------------------------------------------------------------

export interface PickerProfile {
  id: string;
  name: string;
  status?: string;
  total_epochs_trained: number;
  needs_retraining?: boolean;
  embedder?: string;
  vocoder?: string;
  pipeline?: string;
}

interface ProfilePickerProps {
  profiles: PickerProfile[];
  selectedId: string | null;
  onChange: (id: string) => void;
  disabled?: boolean;
  emptyMessage?: string;
}

export function ProfilePicker({
  profiles,
  selectedId,
  onChange,
  disabled = false,
  emptyMessage = 'No profiles found',
}: ProfilePickerProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  // Keyboard: Escape closes
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [open]);

  if (profiles.length === 0) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-2.5
                      text-[12px] font-mono text-zinc-500">
        {emptyMessage}
      </div>
    );
  }

  const selected = profiles.find(p => p.id === (selectedId ?? '')) ?? null;

  function select(id: string) {
    onChange(id);
    setOpen(false);
  }

  return (
    <div ref={ref} className="relative w-full">
      {/* Trigger */}
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between gap-3
                   bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2
                   hover:border-zinc-600 focus:outline-none focus:border-cyan-600
                   disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {selected ? (
          <ProfileRow profile={selected} />
        ) : (
          <span className="text-[12px] font-mono text-zinc-500">Select a profile…</span>
        )}
        <svg className={`w-3.5 h-3.5 text-zinc-500 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`}
             fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown list */}
      {open && (
        <div className="absolute z-50 mt-1 w-full max-h-72 overflow-y-auto
                        rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl">
          {profiles.map(p => (
            <button
              key={p.id}
              type="button"
              onClick={() => select(p.id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 text-left
                          hover:bg-zinc-800 transition-colors
                          ${p.id === selectedId ? 'bg-zinc-800/60' : ''}
                          border-b border-zinc-800/60 last:border-b-0`}
            >
              <ProfileRow profile={p} selected={p.id === selectedId} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single row — same badge style as Library page
// ---------------------------------------------------------------------------
function ProfileRow({ profile, selected }: { profile: PickerProfile; selected?: boolean }) {
  const emb     = profile.embedder ?? 'spin-v2';
  const voc     = profile.vocoder  ?? 'HiFi-GAN';
  const pip     = profile.pipeline ?? 'rvc';
  const isB2    = pip === 'beatrice2';
  const locked  = profile.total_epochs_trained > 0;
  const epochs  = locked ? `${profile.total_epochs_trained} epochs` : 'untrained';

  const statusIcon = profile.status === 'trained'  ? ' ✓'
                   : profile.status === 'training' ? ' ⟳'
                   : profile.status === 'failed'   ? ' ✗'
                   : '';

  return (
    <div className="flex items-center gap-2 flex-wrap min-w-0">
      {/* Name */}
      <span className={`text-[12px] font-mono font-medium truncate max-w-[140px]
                        ${selected ? 'text-cyan-300' : 'text-zinc-200'}`}>
        {profile.name}{statusIcon}
        {profile.needs_retraining && <span className="ml-1 text-amber-400">⚠</span>}
      </span>

      {/* Epoch / step count */}
      <span className={`text-[11px] font-mono tabular-nums shrink-0
                        ${locked ? 'text-cyan-600' : 'text-zinc-600'}`}>
        {epochs}
      </span>

      {/* Pipeline badge */}
      {isB2 ? (
        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono shrink-0
                         bg-amber-950/40 border border-amber-800/40 text-amber-400">
          ◈ Beatrice 2
        </span>
      ) : (
        <>
          {/* Embedder badge */}
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono shrink-0
                           bg-indigo-950/40 border border-indigo-800/40 text-indigo-400">
            ◈ {emb}
            {locked && <span className="text-indigo-600" title="Locked after first training run">🔒</span>}
          </span>

          {/* Vocoder badge */}
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono shrink-0
                           bg-violet-950/40 border border-violet-800/40 text-violet-400">
            ◈ {voc}
            {locked && <span className="text-violet-600" title="Locked after first training run">🔒</span>}
          </span>
        </>
      )}
    </div>
  );
}
