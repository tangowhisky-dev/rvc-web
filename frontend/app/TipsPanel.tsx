'use client';

interface Tip {
  icon: string;
  title: string;
  body: string;
}

interface TipsPanelProps {
  tips: Tip[];
}

export function TipsPanel({ tips }: TipsPanelProps) {
  return (
    <aside className="rounded-xl border border-zinc-800 bg-zinc-900/30 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-zinc-800 flex items-center gap-2">
        <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">Tips</span>
      </div>
      <div className="divide-y divide-zinc-800/60">
        {tips.map((tip, i) => (
          <div key={i} className="px-4 py-3 flex gap-3">
            <span className="text-base shrink-0 mt-px">{tip.icon}</span>
            <div className="flex flex-col gap-0.5">
              <span className="text-[11px] font-mono font-semibold text-zinc-300">{tip.title}</span>
              <span className="text-[11px] font-mono text-zinc-500 leading-relaxed">{tip.body}</span>
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
