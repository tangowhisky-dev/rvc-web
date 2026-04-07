'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const TABS = [
  { label: 'Library', href: '/' },
  { label: 'Training', href: '/training' },
  { label: 'Realtime', href: '/realtime' },
  { label: 'Offline', href: '/offline' },
  { label: 'Analysis', href: '/analysis' },
  { label: 'Setup', href: '/setup' },
];

export function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="bg-zinc-900/80 border-b border-zinc-800 px-6 flex items-center gap-1 sticky top-0 z-50 backdrop-blur-sm">
      {/* Brand */}
      <span className="text-[11px] font-mono font-medium tracking-[0.15em] uppercase text-zinc-400 mr-4 py-3">
        RVC <span className="text-cyan-400">Studio</span>
      </span>

      {/* Tabs */}
      {TABS.map(({ label, href }) => {
        const isActive =
          href === '/' ? pathname === '/' : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            className={`px-4 py-3 text-[13px] font-mono transition-colors border-b-2 ${
              isActive
                ? 'text-cyan-400 border-cyan-500'
                : 'text-zinc-400 hover:text-zinc-200 border-transparent'
            }`}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
