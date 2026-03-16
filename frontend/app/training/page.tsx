'use client';

import { useEffect, useRef, useState } from 'react';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Profile {
  id: string;
  name: string;
  status: string;
}

interface TrainingMsg {
  type: 'log' | 'phase' | 'done' | 'error' | 'keepalive' | 'epoch' | 'epoch_done' | 'index_done';
  message?: string;
  phase?: string;
  elapsed_s?: number;
  epoch?: number;
}

type JobState = 'idle' | 'running' | 'done' | 'failed';

// ---------------------------------------------------------------------------
// Phase definitions
// ---------------------------------------------------------------------------

const PHASES = ['preprocess', 'extract_f0', 'extract_feature', 'train', 'index', 'done'] as const;

const PHASE_LABELS: Record<string, string> = {
  preprocess: 'Preprocess',
  extract_f0: 'F0 Extract',
  extract_feature: 'Features',
  train: 'Train',
  index: 'Index',
  done: 'Done',
};

// ---------------------------------------------------------------------------
// Phase Progress Bar
// ---------------------------------------------------------------------------

function PhaseBar({ currentPhase, jobDone }: { currentPhase: string | null; jobDone: boolean }) {
  if (currentPhase === null) return null;

  const activePhase = jobDone ? 'done' : currentPhase;
  const currentIdx = PHASES.indexOf(activePhase as (typeof PHASES)[number]);

  return (
    <div className="flex gap-1.5 items-center">
      {PHASES.map((phase, idx) => {
        let pillClass: string;
        if (phase === activePhase && phase === 'done') {
          // Done — green
          pillClass = 'bg-emerald-700 text-emerald-100 shadow-[0_0_8px_rgba(16,185,129,0.4)]';
        } else if (phase === activePhase) {
          // Active non-done phase — cyan
          pillClass = 'bg-cyan-600 text-white shadow-[0_0_8px_rgba(8,145,178,0.5)]';
        } else if (idx < (currentIdx === -1 ? 0 : currentIdx)) {
          pillClass = 'bg-zinc-700 text-zinc-400';
        } else {
          pillClass = 'bg-zinc-900 text-zinc-600 border border-zinc-800';
        }
        return (
          <div
            key={phase}
            className={`flex-1 py-1.5 px-2 rounded text-center text-[10px] font-mono font-medium
                        uppercase tracking-wider transition-all duration-300 ${pillClass}`}
          >
            {PHASE_LABELS[phase]}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page Component
// ---------------------------------------------------------------------------

export default function TrainingPage() {
  // Profiles
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [profilesError, setProfilesError] = useState<string | null>(null);

  // Training config
  const [epochs, setEpochs] = useState<number>(20);

  // Training state
  const [logLines, setLogLines] = useState<string[]>([]);
  const [currentPhase, setCurrentPhase] = useState<string | null>(null);
  const [jobState, setJobState] = useState<JobState>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const jobStateRef = useRef<JobState>('idle');

  // Keep jobStateRef in sync (needed inside WS callbacks to avoid stale closure)
  useEffect(() => {
    jobStateRef.current = jobState;
  }, [jobState]);

  // ---------------------------------------------------------------------------
  // Load profiles on mount
  // ---------------------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    async function fetchProfiles() {
      try {
        const res = await fetch(`${API}/api/profiles`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: Profile[] = await res.json();
        if (cancelled) return;
        setProfiles(data);
        if (data.length > 0) setSelectedId(data[0].id);
      } catch (err) {
        if (!cancelled) {
          setProfilesError(
            `Cannot reach backend (${err instanceof Error ? err.message : String(err)}). ` +
              'Start the backend: conda run -n rvc uvicorn backend.app.main:app --reload'
          );
        }
      }
    }

    fetchProfiles();
    return () => {
      cancelled = true;
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Auto-scroll log area when lines change
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  // ---------------------------------------------------------------------------
  // Cleanup WS on unmount
  // ---------------------------------------------------------------------------

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  function appendLog(line: string) {
    setLogLines((prev) => {
      const next = [...prev, line];
      return next.length > 500 ? next.slice(next.length - 500) : next;
    });
  }

  function closeWs() {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Start Training
  // ---------------------------------------------------------------------------

  async function handleStart() {
    if (!selectedId || jobState === 'running') return;

    setErrorMsg(null);
    setLogLines([]);
    setCurrentPhase(null);

    try {
      // POST to start the job — WS connects AFTER this returns
      const res = await fetch(`${API}/api/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: selectedId, epochs }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }

      // Job accepted — transition to running state
      setJobState('running');
      jobStateRef.current = 'running';

      // Connect WebSocket after job is confirmed
      const ws = new WebSocket(`${WS_BASE}/ws/training/${selectedId}`);
      wsRef.current = ws;

      ws.onmessage = (event: MessageEvent) => {
        try {
          const msg = JSON.parse(event.data as string) as TrainingMsg;

          // Always update phase bar if message carries a phase
          if (msg.phase && msg.type !== 'done') {
            setCurrentPhase(msg.phase);
          }

          if (msg.type === 'log') {
            if (msg.message) appendLog(msg.message);

          } else if (msg.type === 'phase') {
            if (msg.message) appendLog(`[${msg.phase ?? ''}] ${msg.message}`);

          } else if (msg.type === 'epoch') {
            // New epoch started — show as prominent log entry
            if (msg.message) appendLog(`▶ ${msg.message}`);

          } else if (msg.type === 'epoch_done') {
            // Epoch completed
            if (msg.message) appendLog(`✓ ${msg.message}`);

          } else if (msg.type === 'index_done') {
            // FAISS index built — transition phase bar to done
            if (msg.message) appendLog(`✓ ${msg.message}`);

          } else if (msg.type === 'keepalive') {
            // Replace the last keepalive line rather than appending a new one
            if (msg.message) {
              setLogLines((prev) => {
                const last = prev[prev.length - 1] ?? '';
                if (last.startsWith('Training in progress…')) {
                  return [...prev.slice(0, -1), msg.message!];
                }
                return [...prev, msg.message!];
              });
            }

          } else if (msg.type === 'done') {
            setCurrentPhase('done');
            setJobState('done');
            jobStateRef.current = 'done';
            if (msg.message) appendLog(`✓ ${msg.message}`);
            closeWs();

          } else if (msg.type === 'error') {
            setJobState('failed');
            jobStateRef.current = 'failed';
            const errText = msg.message ?? 'Unknown error';
            setErrorMsg(errText);
            appendLog(`ERROR: ${errText}`);
            closeWs();
          }
        } catch (_) {
          // Malformed JSON — ignore
        }
      };

      ws.onerror = () => {
        appendLog('(WebSocket error)');
        if (jobStateRef.current === 'running') {
          setJobState('failed');
          jobStateRef.current = 'failed';
          setErrorMsg('WebSocket error — check backend logs.');
        }
        closeWs();
      };

      ws.onclose = () => {
        // Only treat unexpected close as an error — not after a clean done/failed
        if (jobStateRef.current === 'running') {
          appendLog('(connection closed)');
          setJobState('idle');
          jobStateRef.current = 'idle';
        }
        // If done or failed: keep the state — don't reset to idle
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(msg);
      setJobState('failed');
      jobStateRef.current = 'failed';
    }
  }

  // ---------------------------------------------------------------------------
  // Cancel Training
  // ---------------------------------------------------------------------------

  async function handleCancel() {
    if (!selectedId) return;

    try {
      await fetch(`${API}/api/training/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: selectedId }),
      });
    } catch (_) {
      // Best-effort
    }

    closeWs();
    setJobState('idle');
    jobStateRef.current = 'idle';
    appendLog('(training cancelled)');
  }

  // ---------------------------------------------------------------------------
  // Derived UI state
  // ---------------------------------------------------------------------------

  const isRunning = jobState === 'running';
  // Show Start Training only when fully idle or when previous run is done/failed
  // (not while running — which includes the index phase)
  const canStart = !!selectedId && jobState !== 'running';

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center">
            {/* Neural/training icon */}
            <svg
              width="12"
              height="12"
              viewBox="0 0 12 12"
              fill="none"
              className="text-cyan-400"
            >
              <circle cx="2" cy="6" r="1.2" fill="currentColor" />
              <circle cx="6" cy="2" r="1.2" fill="currentColor" />
              <circle cx="6" cy="10" r="1.2" fill="currentColor" />
              <circle cx="10" cy="6" r="1.2" fill="currentColor" />
              <line x1="3.2" y1="6" x2="4.8" y2="2.8" stroke="currentColor" strokeWidth="0.8" />
              <line x1="3.2" y1="6" x2="4.8" y2="9.2" stroke="currentColor" strokeWidth="0.8" />
              <line x1="7.2" y1="2.8" x2="8.8" y2="6" stroke="currentColor" strokeWidth="0.8" />
              <line x1="7.2" y1="9.2" x2="8.8" y2="6" stroke="currentColor" strokeWidth="0.8" />
            </svg>
          </div>
          <h1 className="text-sm font-mono font-medium tracking-wide">
            RVC <span className="text-cyan-400">Training</span>
          </h1>
        </div>

        {/* Job status badge */}
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full transition-colors ${
              isRunning
                ? 'bg-cyan-400 shadow-[0_0_6px_rgba(34,211,238,0.6)] animate-pulse'
                : jobState === 'done'
                ? 'bg-emerald-400'
                : jobState === 'failed'
                ? 'bg-red-400'
                : 'bg-zinc-600'
            }`}
          />
          <span className="text-[11px] font-mono uppercase tracking-widest text-zinc-400">
            {isRunning ? 'training' : jobState === 'done' ? 'done' : jobState === 'failed' ? 'failed' : 'idle'}
          </span>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-6 py-8 flex flex-col gap-8">
        {/* Error banner */}
        {(profilesError || (jobState === 'failed' && errorMsg)) && (
          <div className="rounded-lg border border-red-800/60 bg-red-950/40 px-4 py-3 text-[13px] font-mono text-red-300">
            {profilesError || errorMsg}
          </div>
        )}

        {/* Configuration */}
        <section>
          <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500 mb-4">
            Training Configuration
          </h2>
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5 flex flex-col gap-5">
            <div className="grid grid-cols-2 gap-4">
              {/* Profile selector */}
              <div className="flex flex-col gap-2">
                <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                  <span className="text-cyan-400">◈</span> Voice Profile
                </label>
                {profiles.length === 0 ? (
                  <div className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px] font-mono text-zinc-500">
                    {profilesError ? 'Backend unreachable' : 'No profiles — upload one in Library'}
                  </div>
                ) : (
                  <select
                    value={selectedId}
                    disabled={isRunning}
                    onChange={(e) => setSelectedId(e.target.value)}
                    className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                               font-mono text-zinc-200 focus:outline-none focus:border-cyan-600
                               disabled:opacity-40 disabled:cursor-not-allowed
                               hover:border-zinc-600 transition-colors"
                  >
                    {profiles.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}{' '}
                        {p.status === 'trained'
                          ? '✓'
                          : p.status === 'training'
                          ? '⟳'
                          : p.status === 'failed'
                          ? '✗'
                          : ''}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {/* Epoch count */}
              <div className="flex flex-col gap-2">
                <label className="text-[11px] font-mono uppercase tracking-widest text-zinc-400 flex items-center gap-2">
                  <span className="text-cyan-400">⟳</span> Epochs
                </label>
                <input
                  type="number"
                  value={epochs}
                  min={1}
                  max={200}
                  disabled={isRunning}
                  onChange={(e) => setEpochs(Math.max(1, Math.min(200, Number(e.target.value))))}
                  className="bg-zinc-900 border border-zinc-700 rounded-md px-3 py-2 text-[13px]
                             font-mono text-zinc-200 focus:outline-none focus:border-cyan-600
                             disabled:opacity-40 disabled:cursor-not-allowed
                             hover:border-zinc-600 transition-colors"
                />
              </div>
            </div>

            {/* Action buttons */}
            <div className="flex items-center gap-3 pt-2 border-t border-zinc-800">
              <button
                onClick={handleStart}
                disabled={!canStart}
                className="flex-1 py-3 rounded-lg font-mono text-[13px] font-medium
                           tracking-wider uppercase transition-all
                           bg-cyan-900/40 border border-cyan-600/40 text-cyan-300
                           hover:bg-cyan-800/40 hover:border-cyan-500/60
                           disabled:opacity-30 disabled:cursor-not-allowed"
              >
                {jobState === 'done' ? '↺ Train Again' : '▶ Start Training'}
              </button>

              {isRunning && (
                <button
                  onClick={handleCancel}
                  className="flex-1 py-3 rounded-lg font-mono text-[13px] font-medium
                             tracking-wider uppercase transition-all
                             bg-red-900/40 border border-red-700/40 text-red-300
                             hover:bg-red-800/40 hover:border-red-600/60"
                >
                  ◼ Cancel
                </button>
              )}
            </div>
          </div>
        </section>

        {/* Phase progress bar — only when a phase is active */}
        {currentPhase !== null && (
          <section>
            <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500 mb-3">
              Progress
            </h2>
            <PhaseBar currentPhase={currentPhase} jobDone={jobState === 'done'} />
          </section>
        )}

        {/* Log area */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">
              Training Log
            </h2>
            {logLines.length > 0 && (
              <span className="text-[10px] font-mono text-zinc-600">
                {logLines.length} line{logLines.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
          <div
            ref={logRef}
            className="overflow-y-auto max-h-80 bg-zinc-950 rounded-lg p-3 font-mono text-[11px]
                       border border-zinc-800"
          >
            {logLines.length === 0 ? (
              <div className="text-zinc-600 select-none">
                {isRunning ? 'Waiting for log output…' : 'No training output yet. Start a job to see logs.'}
              </div>
            ) : (
              logLines.map((line, i) => (
                <div
                  key={i}
                  className={`leading-relaxed whitespace-pre-wrap break-all ${
                    line.startsWith('ERROR:')
                      ? 'text-red-400'
                      : line.startsWith('(')
                      ? 'text-zinc-500 italic'
                      : 'text-zinc-300'
                  }`}
                >
                  {line}
                </div>
              ))
            )}
          </div>
        </section>

        {/* Footer */}
        <footer className="text-[11px] font-mono text-zinc-600 space-y-1 pb-4">
          <p>
            Training data must be uploaded as a voice sample in the Library tab first.
          </p>
          <p>
            Backend:{' '}
            <code className="text-zinc-500">
              conda run -n rvc uvicorn backend.app.main:app --reload
            </code>
          </p>
        </footer>
      </div>
    </main>
  );
}
