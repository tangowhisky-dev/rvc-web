export const metadata = {
  title: "BlackHole Setup Guide — RVC Studio",
};

const sections = [
  {
    num: 1,
    title: "Install BlackHole 2ch",
    steps: [
      "Download the latest BlackHole 2ch installer from the Existential Audio GitHub repository (github.com/ExistentialAudio/BlackHole).",
      "Run the downloaded .pkg installer and follow the prompts.",
      "After installation, restart your Mac (or at minimum log out and back in) so macOS registers the new audio device.",
    ],
  },
  {
    num: 2,
    title: "Open Audio MIDI Setup",
    steps: [
      "Open Finder → Applications → Utilities → Audio MIDI Setup.",
      "Click the + button in the bottom-left corner and choose Create Multi-Output Device.",
      "In the device list on the right, check both BlackHole 2ch and your speakers (e.g. MacBook Pro Speakers or your external audio interface).",
      "Right-click (or Control-click) the speaker entry and select Use This Device For Sound Output to make it the master clock source.",
    ],
  },
  {
    num: 3,
    title: "Set System Output to Multi-Output Device",
    steps: [
      "Open System Settings → Sound → Output.",
      "Select the Multi-Output Device you just created.",
      "This routes system audio to both your physical speakers and BlackHole 2ch simultaneously, so you can monitor audio while it is being routed.",
    ],
  },
  {
    num: 4,
    title: "Start RVC and Route Output to BlackHole",
    steps: [
      "Run ./scripts/start.sh from the rvc-web project directory, or start the backend and frontend manually.",
      "Navigate to the Realtime tab in RVC Studio.",
      "Set the Output Device to BlackHole 2ch. BlackHole 2ch is typically device id=1 on a default install.",
      "Start the RVC audio engine — converted voice audio will be written to the BlackHole 2ch virtual device.",
    ],
  },
  {
    num: 5,
    title: "Configure Your Target App's Input",
    steps: [
      "Open your DAW or communications app (e.g. Discord, Zoom, GarageBand, OBS).",
      "In that app's audio/microphone settings, set the input device to BlackHole 2ch.",
      "The app now receives the RVC-converted voice as its microphone input.",
      "Test by speaking — the converted voice should appear as the active input in the target app.",
    ],
  },
];

export default function SetupPage() {
  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-10 max-w-3xl mx-auto">
      {/* Header */}
      <div className="mb-10">
        <h1 className="text-2xl font-mono font-semibold text-zinc-100 mb-2">
          BlackHole Setup Guide
        </h1>
        <p className="text-zinc-400 text-sm leading-relaxed">
          BlackHole 2ch is a virtual audio device that lets RVC Studio route
          converted voice audio to any app on your Mac. Follow the five steps
          below to get everything wired up.
        </p>
      </div>

      {/* Sections */}
      <div className="space-y-8">
        {sections.map(({ num, title, steps }) => (
          <section key={num} className="bg-zinc-900 rounded-xl border border-zinc-800 p-6">
            <h2 className="text-base font-mono font-semibold text-cyan-400 mb-4">
              {num}. {title}
            </h2>
            <ol className="space-y-2 list-none">
              {steps.map((step, i) => (
                <li key={i} className="flex gap-3 text-sm text-zinc-300 leading-relaxed">
                  <span className="text-zinc-600 font-mono select-none shrink-0">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          </section>
        ))}
      </div>

      {/* Tip block */}
      <div className="mt-8 bg-cyan-950/40 border border-cyan-800/50 rounded-xl p-5">
        <p className="text-sm font-mono text-cyan-300 font-semibold mb-1">Tip — Monitor BlackHole output</p>
        <p className="text-sm text-zinc-300 leading-relaxed">
          To hear what BlackHole is receiving in real time, add BlackHole 2ch to the
          Multi-Output Device in Audio MIDI Setup and route your physical speakers there
          too. Audio will play through both the virtual device and your speakers
          simultaneously.
        </p>
      </div>

      {/* Quick reference */}
      <div className="mt-6 bg-zinc-900 border border-zinc-800 rounded-xl p-5">
        <p className="text-xs font-mono text-zinc-500 mb-3 uppercase tracking-wider">Quick Reference</p>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-zinc-500 text-xs font-mono">BlackHole device id</span>
            <p className="text-zinc-200 font-mono">1 (default install)</p>
          </div>
          <div>
            <span className="text-zinc-500 text-xs font-mono">Backend port</span>
            <p className="text-zinc-200 font-mono">8000</p>
          </div>
          <div>
            <span className="text-zinc-500 text-xs font-mono">Frontend port</span>
            <p className="text-zinc-200 font-mono">3000</p>
          </div>
          <div>
            <span className="text-zinc-500 text-xs font-mono">Health check</span>
            <p className="text-zinc-200 font-mono">curl localhost:8000/health</p>
          </div>
        </div>
      </div>
    </main>
  );
}
