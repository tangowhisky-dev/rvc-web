# Requirements

This file is the explicit capability and coverage contract for the project.

## Active

### R001 — Voice Sample Library
- Class: primary-user-loop
- Status: validated
- Description: User can upload audio files and name them as voice profiles. Profiles are persisted in SQLite with metadata (name, created_at, file path, training status). User can list, select, and delete profiles.
- Why it matters: All other features depend on a named, persistent voice profile.
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: M001/S02, M001/S04
- Validation: S01 — 10 passing contract tests + live curl verification (upload, list, get, delete) with real SQLite writes and disk I/O
- Notes: Audio stored in `data/samples/<profile_id>/` relative to the backend root.

### R002 — Training / Fine-tuning Pipeline
- Class: primary-user-loop
- Status: active
- Description: User can select a voice profile and trigger a full RVC fine-tuning run (preprocess → feature extract → train). Training runs against a copy of the base pretrained v2 model — the base is never modified. One fine-tuned copy exists at a time; running training again overwrites it. Training logs stream in real time via WebSocket. User can see current phase and cancel.
- Why it matters: The fine-tuned model is what produces convincing voice cloning.
- Source: user
- Primary owning slice: M001/S02
- Supporting slices: M001/S04
- Validation: unmapped
- Notes: Uses `infer/modules/train/preprocess.py`, `extract_feature_print.py`, `train.py` as subprocesses launched from FastAPI. macOS / MPS means training is slow — realistic expectation is 20–60 min for a short sample.

### R003 — Realtime Voice Conversion
- Class: primary-user-loop
- Status: active
- Description: User can start a realtime voice conversion session that reads audio from a selected input device (mic), converts it using the selected fine-tuned model via `rtrvc.py`, and writes output to a selected output device (BlackHole 2ch). User controls pitch (semitones), index rate, and protect. Session can be started and stopped from the browser.
- Why it matters: This is the core end-to-end user outcome of the whole system.
- Source: user
- Primary owning slice: M001/S03
- Supporting slices: M001/S04
- Validation: unmapped
- Notes: Audio loop lives entirely in Python (sounddevice). Browser only controls start/stop/params. Block size targeting 150–300ms latency on Apple Silicon MPS.

### R004 — Live Waveform Visualization
- Class: primary-user-loop
- Status: active
- Description: While a realtime session is active, the browser displays live waveforms for both the input and output audio streams. Rendering is offloaded to a Web Worker with OffscreenCanvas so the main thread stays responsive.
- Why it matters: Provides visual feedback that the system is working and audio is flowing.
- Source: user
- Primary owning slice: M001/S03
- Supporting slices: none
- Validation: unmapped
- Notes: Backend samples audio data and sends it to the frontend via WebSocket as compact float arrays. The Web Worker renders into a canvas element.

### R005 — Audio Device Selection
- Class: integration
- Status: active
- Description: User can select the input device (microphone) and output device from a list of available system audio devices. BlackHole 2ch should be prominent. Selection is persisted across sessions.
- Why it matters: Without correct device selection the audio routing to BlackHole won't work.
- Source: user
- Primary owning slice: M001/S03
- Supporting slices: none
- Validation: unmapped
- Notes: Device list retrieved from `sounddevice.query_devices()` via backend API. MacBook Pro Microphone (id 2) and BlackHole 2ch (id 1) are the expected defaults.

### R006 — Modular Audio Processing Chain
- Class: quality-attribute
- Status: active
- Description: The realtime audio pipeline is structured as a composable chain of processing stages (e.g. preprocess → RVC infer → postprocess) so that future effects (noise removal, EQ, compression, clipping control) can be inserted without rewriting the core loop.
- Why it matters: User explicitly requested future effects support. Architectural debt here would make M002 painful.
- Source: user
- Primary owning slice: M001/S03
- Supporting slices: none
- Validation: unmapped
- Notes: Design as a list of `AudioProcessor` objects each with a `process(audio: np.ndarray) -> np.ndarray` interface.

### R007 — Start/Stop Service Scripts
- Class: operability
- Status: active
- Description: `scripts/start.sh` starts both services (FastAPI backend + Next.js frontend) and `scripts/stop.sh` stops them cleanly. Scripts handle conda env activation for the backend.
- Why it matters: Without scripts, the user must manually start services each time.
- Source: user
- Primary owning slice: M001/S04
- Supporting slices: none
- Validation: unmapped
- Notes: Backend runs under `conda run -n rvc`. PID files stored in `.pids/` for clean stop.

### R008 — BlackHole Setup Guide
- Class: operability
- Status: active
- Description: The app tells the user exactly which macOS Audio MIDI Setup and system settings to configure for BlackHole 2ch to work as the virtual output. This information is presented in the UI (not just docs).
- Why it matters: Without correct macOS routing, the converted voice never reaches other apps.
- Source: user
- Primary owning slice: M001/S04
- Supporting slices: none
- Validation: unmapped
- Notes: BlackHole device id=1 on this machine. The typical setup is: set RVC output → BlackHole 2ch, then set the target app's mic input → BlackHole 2ch.

## Deferred

### R009 — Audio Effects Chain (EQ, noise removal, compression, clipping)
- Class: differentiator
- Status: deferred
- Description: Real-time post-processing effects applied to the converted output stream: noise gate, de-click, EQ bands (bass/treble), loudness normalization, clipping control.
- Why it matters: Polishes the output quality significantly.
- Source: user
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: Deferred to M002. Architecture in M001/S03 must accommodate insertion points (R006).

### R010 — Multiple Fine-tuned Model Slots
- Class: core-capability
- Status: deferred
- Description: Maintain one trained model slot per voice profile rather than one global slot.
- Why it matters: Makes switching between trained profiles instant without re-training.
- Source: inferred
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: M001 maintains a single fine-tuned copy. Expand in M002.

## Out of Scope

### R011 — CUDA / Windows Support
- Class: constraint
- Status: out-of-scope
- Description: This system targets macOS + Apple Silicon (MPS). CUDA and Windows-specific code paths are not built or tested.
- Why it matters: Prevents scope creep in infrastructure.
- Source: user
- Primary owning slice: none
- Supporting slices: none
- Validation: n/a
- Notes: The upstream RVC repo supports CUDA; we leave those code paths intact but don't wire them.

### R012 — Multi-user / Authentication
- Class: anti-feature
- Status: out-of-scope
- Description: The app is single-user, local-only. No auth, no multi-tenancy.
- Why it matters: Prevents wasted effort on security infrastructure for a personal tool.
- Source: inferred
- Primary owning slice: none
- Supporting slices: none
- Validation: n/a
- Notes: If shared deployment is needed later, add auth then.

## Traceability

| ID | Class | Status | Primary owner | Supporting | Proof |
|---|---|---|---|---|---|
| R001 | primary-user-loop | validated | M001/S01 | S02, S04 | S01 contract tests + curl |
| R002 | primary-user-loop | active | M001/S02 | S04 | unmapped |
| R003 | primary-user-loop | active | M001/S03 | S04 | unmapped |
| R004 | primary-user-loop | active | M001/S03 | none | unmapped |
| R005 | integration | active | M001/S03 | none | unmapped |
| R006 | quality-attribute | active | M001/S03 | none | unmapped |
| R007 | operability | active | M001/S04 | none | unmapped |
| R008 | operability | active | M001/S04 | none | unmapped |
| R009 | differentiator | deferred | none | none | unmapped |
| R010 | core-capability | deferred | none | none | unmapped |
| R011 | constraint | out-of-scope | none | none | n/a |
| R012 | anti-feature | out-of-scope | none | none | n/a |

## Coverage Summary

- Active requirements: 8
- Mapped to slices: 8
- Validated: 1 (R001)
- Unmapped active requirements: 0
