'use client';

import { useEffect, useRef, useState } from 'react';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

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
}

interface RealtimeStatus {
  active: boolean;
}

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: string }) {
  const classes: Record<string, string> = {
    untrained: 'bg-zinc-700 text-zinc-300',
    training: 'bg-amber-900/50 text-amber-300 animate-pulse',
    trained: 'bg-cyan-900/50 text-cyan-300',
    failed: 'bg-red-900/50 text-red-300',
  };
  const cls = classes[status] ?? 'bg-zinc-700 text-zinc-400';
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono uppercase tracking-wider ${cls}`}
    >
      {status}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Main Page Component
// ---------------------------------------------------------------------------

export default function LibraryPage() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);
  const [nameInput, setNameInput] = useState('');
  const [fileInput, setFileInput] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileRef = useRef<HTMLInputElement>(null);

  // ---------------------------------------------------------------------------
  // Fetch profiles
  // ---------------------------------------------------------------------------

  async function fetchProfiles() {
    try {
      const res = await fetch(`${API}/api/profiles`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Profile[] = await res.json();
      setProfiles(data);
    } catch (err) {
      setError(
        `Cannot reach backend (${err instanceof Error ? err.message : String(err)}). ` +
          'Start the backend: conda run -n rvc uvicorn backend.app.main:app --reload'
      );
    } finally {
      setLoading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Poll realtime status every 3 seconds
  // ---------------------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    fetchProfiles();

    const interval = setInterval(async () => {
      if (cancelled) return;
      try {
        const res = await fetch(`${API}/api/realtime/status`);
        if (!res.ok) return;
        const data: RealtimeStatus = await res.json();
        if (!cancelled) setSessionActive(data.active);
      } catch (_) {
        // Non-fatal — session banner just won't show
      }
    }, 3000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------------------------------------------------------------------------
  // Upload handler
  // ---------------------------------------------------------------------------

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!nameInput.trim() || !fileInput) return;

    setUploading(true);
    setError(null);

    const form = new FormData();
    form.append('name', nameInput.trim());
    form.append('file', fileInput);

    try {
      const res = await fetch(`${API}/api/profiles`, {
        method: 'POST',
        // No Content-Type header — browser sets multipart/form-data with boundary
        body: form,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      setNameInput('');
      setFileInput(null);
      if (fileRef.current) fileRef.current.value = '';
      await fetchProfiles();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Delete handler
  // ---------------------------------------------------------------------------

  async function handleDelete(id: string, name: string) {
    if (!window.confirm(`Delete profile "${name}"? This cannot be undone.`)) return;

    try {
      const res = await fetch(`${API}/api/profiles/${id}`, { method: 'DELETE' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      await fetchProfiles();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  // ---------------------------------------------------------------------------
  // Derived state
  // ---------------------------------------------------------------------------

  const canSubmit = nameInput.trim().length > 0 && fileInput !== null && !uploading;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      {/* Session active banner */}
      {sessionActive && (
        <div className="bg-amber-900/40 border-b border-amber-700/50 px-6 py-2 text-[12px] font-mono text-amber-300 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse inline-block" />
          Realtime voice conversion session is active
        </div>
      )}

      <div className="max-w-4xl mx-auto px-6 py-8 flex flex-col gap-8">
        {/* Page header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-mono font-semibold text-zinc-100">
              Voice Profiles
            </h1>
            <p className="text-[13px] text-zinc-500 mt-1">
              Upload audio samples to create profiles for training and realtime conversion.
            </p>
          </div>
          <span className="text-[11px] font-mono text-zinc-600">
            {profiles.length} profile{profiles.length !== 1 ? 's' : ''}
          </span>
        </div>

        {/* Error banner */}
        {error && (
          <div className="rounded-lg border border-red-800/60 bg-red-950/40 px-4 py-3 text-[13px] font-mono text-red-300">
            {error}
          </div>
        )}

        {/* Profile list */}
        {loading ? (
          <div className="text-center py-16 text-[13px] font-mono text-zinc-500">
            Loading…
          </div>
        ) : profiles.length === 0 ? (
          <div className="rounded-xl border border-dashed border-zinc-800 flex items-center justify-center py-16 px-8">
            <p className="text-[13px] font-mono text-zinc-500 text-center">
              No voice profiles yet — upload one below.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {profiles.map((profile) => (
              <div
                key={profile.id}
                className="rounded-xl border border-zinc-800 bg-zinc-900/60 px-5 py-4 flex items-center justify-between gap-4"
              >
                <div className="flex flex-col gap-1 min-w-0">
                  <div className="flex items-center gap-3">
                    <span className="text-[15px] font-mono font-medium text-zinc-100 truncate">
                      {profile.name}
                    </span>
                    <StatusBadge status={profile.status} />
                  </div>
                  <span className="text-[11px] font-mono text-zinc-600 truncate">
                    {profile.id}
                  </span>
                </div>

                <button
                  onClick={() => handleDelete(profile.id, profile.name)}
                  className="flex-shrink-0 px-3 py-1.5 rounded-md text-[11px] font-mono uppercase
                             tracking-wider text-red-400 border border-red-900/50 bg-red-950/20
                             hover:bg-red-900/30 hover:border-red-800/60 transition-colors"
                >
                  Delete
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Upload form */}
        <section>
          <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500 mb-4">
            Upload New Profile
          </h2>
          <form
            onSubmit={handleUpload}
            className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5 flex flex-col gap-4"
          >
            {/* Name field */}
            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
                Profile Name
              </label>
              <input
                type="text"
                value={nameInput}
                onChange={(e) => setNameInput(e.target.value)}
                placeholder="e.g. My Voice"
                className="bg-zinc-950 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                           font-mono text-zinc-200 placeholder:text-zinc-600
                           focus:outline-none focus:border-cyan-600 transition-colors"
              />
            </div>

            {/* File field */}
            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
                Audio Sample
              </label>
              <input
                ref={fileRef}
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a"
                onChange={(e) => setFileInput(e.target.files?.[0] ?? null)}
                className="block text-[13px] font-mono text-zinc-300
                           file:mr-3 file:py-1.5 file:px-3
                           file:rounded file:border file:border-zinc-700
                           file:bg-zinc-800 file:text-zinc-300 file:text-[12px]
                           file:cursor-pointer hover:file:bg-zinc-700
                           file:transition-colors cursor-pointer"
              />
              {fileInput && (
                <span className="text-[11px] font-mono text-zinc-500">
                  {fileInput.name} ({(fileInput.size / 1024).toFixed(0)} KB)
                </span>
              )}
            </div>

            <button
              type="submit"
              disabled={!canSubmit}
              className="mt-1 py-2.5 rounded-lg font-mono text-[13px] font-medium
                         tracking-wider uppercase transition-all
                         bg-cyan-900/40 border border-cyan-600/40 text-cyan-300
                         hover:bg-cyan-800/40 hover:border-cyan-500/60
                         disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {uploading ? '⟳  Uploading…' : '↑  Upload Profile'}
            </button>
          </form>
        </section>
      </div>
    </main>
  );
}
