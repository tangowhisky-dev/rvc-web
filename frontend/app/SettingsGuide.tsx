'use client';

// ---------------------------------------------------------------------------
// SettingsGuide — detailed per-parameter explanations displayed as a
// reference section below the sliders. Each entry covers: what the parameter
// does, how it affects output, and guidance on when to raise / lower it.
// ---------------------------------------------------------------------------

interface Setting {
  name: string;
  range: string;
  default: string;
  summary: string;
  details: string;
  raise: string;
  lower: string;
  badge?: string;     // optional colour key: 'cyan' | 'violet' | 'amber' | 'emerald' | 'red'
}

const BADGE_CLASSES: Record<string, string> = {
  cyan:    'bg-cyan-900/30 text-cyan-400 border-cyan-700/40',
  violet:  'bg-violet-900/30 text-violet-400 border-violet-700/40',
  amber:   'bg-amber-900/30 text-amber-400 border-amber-700/40',
  emerald: 'bg-emerald-900/30 text-emerald-400 border-emerald-700/40',
  red:     'bg-red-900/30 text-red-400 border-red-700/40',
};

export function SettingsGuide({ settings }: { settings: Setting[] }) {
  return (
    <div className="flex flex-col gap-0 rounded-xl border border-zinc-800 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2.5 bg-zinc-900/60 border-b border-zinc-800 flex items-center gap-2">
        <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">
          Parameter Reference
        </span>
      </div>

      {/* Entries */}
      {settings.map((s, i) => (
        <div
          key={i}
          className={`px-5 py-4 flex flex-col gap-2 ${
            i < settings.length - 1 ? 'border-b border-zinc-800/60' : ''
          }`}
        >
          {/* Name row */}
          <div className="flex items-baseline gap-3 flex-wrap">
            <span className="text-[13px] font-mono font-semibold text-zinc-200">{s.name}</span>
            <span className="text-[10px] font-mono text-zinc-600 tabular-nums">{s.range}</span>
            <span className="text-[10px] font-mono text-zinc-600">default {s.default}</span>
            {s.badge && (
              <span className={`ml-auto text-[9px] font-mono uppercase tracking-widest px-2 py-0.5
                               rounded border ${BADGE_CLASSES[s.badge] ?? BADGE_CLASSES.cyan}`}>
                {s.badge}
              </span>
            )}
          </div>

          {/* Summary */}
          <p className="text-[12px] font-mono text-zinc-300 leading-relaxed">{s.summary}</p>

          {/* Details */}
          <p className="text-[11px] font-mono text-zinc-500 leading-relaxed">{s.details}</p>

          {/* Raise / Lower guidance */}
          <div className="grid grid-cols-2 gap-3 mt-1">
            <div className="flex flex-col gap-1 px-3 py-2 rounded-lg bg-emerald-950/20 border border-emerald-900/30">
              <span className="text-[9px] font-mono uppercase tracking-widest text-emerald-600">Raise when</span>
              <span className="text-[11px] font-mono text-zinc-400 leading-relaxed">{s.raise}</span>
            </div>
            <div className="flex flex-col gap-1 px-3 py-2 rounded-lg bg-red-950/20 border border-red-900/30">
              <span className="text-[9px] font-mono uppercase tracking-widest text-red-600">Lower when</span>
              <span className="text-[11px] font-mono text-zinc-400 leading-relaxed">{s.lower}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// LossGuide — explains each training loss signal
// ---------------------------------------------------------------------------

interface LossEntry {
  key: string;
  color: string;
  name: string;
  what: string;
  healthy: string;
  warning: string;
}

export function LossGuide({ losses }: { losses: LossEntry[] }) {
  return (
    <div className="flex flex-col gap-0 rounded-xl border border-zinc-800 overflow-hidden">
      <div className="px-4 py-2.5 bg-zinc-900/60 border-b border-zinc-800">
        <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">
          Loss Reference — what each signal means
        </span>
      </div>

      {losses.map((l, i) => (
        <div
          key={i}
          className={`px-5 py-4 flex flex-col gap-2 ${
            i < losses.length - 1 ? 'border-b border-zinc-800/60' : ''
          }`}
        >
          <div className="flex items-center gap-3">
            <span
              className="w-2.5 h-2.5 rounded-sm shrink-0"
              style={{ backgroundColor: l.color }}
            />
            <span className="text-[13px] font-mono font-semibold text-zinc-200">{l.name}</span>
            <code className="text-[10px] font-mono text-zinc-600">{l.key}</code>
          </div>

          <p className="text-[12px] font-mono text-zinc-300 leading-relaxed">{l.what}</p>

          <div className="grid grid-cols-2 gap-3 mt-1">
            <div className="flex flex-col gap-1 px-3 py-2 rounded-lg bg-emerald-950/20 border border-emerald-900/30">
              <span className="text-[9px] font-mono uppercase tracking-widest text-emerald-600">Healthy trend</span>
              <span className="text-[11px] font-mono text-zinc-400 leading-relaxed">{l.healthy}</span>
            </div>
            <div className="flex flex-col gap-1 px-3 py-2 rounded-lg bg-red-950/20 border border-red-900/30">
              <span className="text-[9px] font-mono uppercase tracking-widest text-red-600">Watch out for</span>
              <span className="text-[11px] font-mono text-zinc-400 leading-relaxed">{l.warning}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
