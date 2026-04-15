# Beatrice 2 Integration Plan

**Status**: Planned — not yet implemented  
**Branch**: `RVC studio - beatrice 2`  
**Date**: 2026-04-15

---

## Overview

Integrate the Beatrice 2 voice conversion trainer as a second pipeline alongside the existing RVC pipeline. Users choose the engine at profile creation time; the choice is immutable (locked after the profile is created). All training, inference (offline and realtime), and UI flows are engine-aware.

---

## Engine Comparison

| Aspect | RVC | Beatrice 2 |
|---|---|---|
| Architecture | VITS encoder + HiFi-GAN/RefineGAN vocoder | ConvNeXt + D4C vocoder (source-filter model) |
| Training granularity | Epoch-based (1–1000 epochs) | Step-based (default 10,000 steps) |
| Checkpoint format | `G_latest.pth` (PyTorch .pth) | `checkpoint_latest.pt.gz` (gzip'd .pt) |
| Inference format | `model_infer.pth` | `paraphernalia_*/` directory (binary dumps) |
| Input SR | 16 kHz | 16 kHz |
| Output SR | 32 kHz | 24 kHz |
| CUDA required | No (MPS works, slower) | Yes — AMP + `torch.amp.autocast("cuda")` hardcoded |
| Speaker model | Single speaker per profile | Multi-speaker (1 speaker minimum) |
| Augmentation data | None | IR + noise files required |
| Embedding | Configurable (spin-v2 etc.) | Internal PhoneExtractor (not configurable) |
| F0 method | RMVPE / crepe / pm | Internal PitchEstimator (not configurable) |
| Export | `.pth` checkpoint + `.index` | `paraphernalia/` dir with 3 binary files + speaker embeddings + TOML |
| Progress tracking | Epoch number + loss values via stdout | Step number + loss values via tqdm/stdout |
| MOS evaluation | Not built in | Built in via UTMOS (torch.hub — requires internet on first run) |

---

## Milestone Breakdown (Implementation Order)

| # | Title | Description | Risk |
|---|---|---|---|
| M1 | Branch + file layout + asset paths + DB | Create branch, move files, resolve asset paths, DB migration | Low |
| M2 | Refactor `__main__.py` → logical modules | Split monolith into focused files, zero logic changes | Medium |
| M3 | Training wrapper + subprocess management | `BeatriceTrainingJob`, preprocess step, stdout parsing, DB writer | High |
| M4 | Inference engine (offline) | `BeatriceInferenceEngine`, offline convert integration | High |
| M5 | API contract updates | Profiles, training, offline routers; pipeline-aware endpoints | Low |
| M6 | Frontend: profile creation + training UI | Engine selector, step-based progress, beatrice2-specific params | Medium |
| M7 | Realtime inference integration | Realtime worker beatrice2 branch, formant shift param | Medium |
| M8 | Export / import Beatrice 2 format | Zip bundle with paraphernalia, manifest v2 | Low |

Pause between each milestone to avoid rate-limiting.

---

## Milestone 1 — Branch, File Layout, Asset Paths, DB

### 1.1 Create branch

```bash
git checkout -b "RVC studio - beatrice 2"
```

### 1.2 Move beatrice trainer to backend

Current location: `beatrice/trainer/beatrice_trainer/`  
New location: `backend/beatrice2/beatrice_trainer/`

After the Milestone 2 refactor (below) the trainer code will be split into sub-modules. Until then the original `__main__.py` is placed intact.

Additional files moved from `beatrice/`:
```
beatrice/preprocess.py        → backend/beatrice2/preprocess.py
beatrice/download_assets.py   → backend/beatrice2/download_assets.py
```

New wrapper files (empty stubs in M1, filled in M3/M4):
```
backend/beatrice2/__init__.py
backend/beatrice2/training.py        ← training wrapper (filled in M3)
backend/beatrice2/inference.py       ← inference adapter (filled in M4)
```

The `beatrice/` directory at the repo root is deleted after files are moved.

### 1.3 Asset locations

All Beatrice 2 pretrained assets live under `assets/beatrice2/`:

```
assets/
  beatrice2/
    pretrained/
      122_checkpoint_03000000.pt          ← PhoneExtractor weights (~16 MB)
      104_3_checkpoint_00300000.pt        ← PitchEstimator weights (~16 MB)
      151_checkpoint_libritts_r_200_02750000.pt.gz  ← pretrained base model (~140 MB)
    ir/
      0000.flac … (20-file subset or 1000 full)
    noise/
      *.flac (20-file subset or 1000 full)
    test/
      common_voice_ja_*.wav (8 evaluation utterances)
    images/
      noimage.png
```

**Not** inside `backend/beatrice2/assets/` — asset blobs stay in the top-level `assets/` tree consistent with RVC convention and `.gitignore` rules.

### 1.4 Config path resolution

`repo_root()` in `__main__.py` checks `BEATRICE_TRAINER_ROOT` env var first:
```python
def repo_root() -> Path:
    if "BEATRICE_TRAINER_ROOT" in os.environ:
        return Path(os.environ["BEATRICE_TRAINER_ROOT"])
    ...
```

Setting `BEATRICE_TRAINER_ROOT=/path/to/rvc-web` makes all relative asset paths resolve correctly:
```
phone_extractor_file  → assets/beatrice2/pretrained/122_checkpoint_03000000.pt
pitch_estimator_file  → assets/beatrice2/pretrained/104_3_checkpoint_00300000.pt
pretrained_file       → assets/beatrice2/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz
in_ir_wav_dir         → assets/beatrice2/ir
in_noise_wav_dir      → assets/beatrice2/noise
in_test_wav_dir       → assets/beatrice2/test
```

These overrides are written to a per-job `config.json` before launching the trainer subprocess. The `assets/default_config.json` in the trainer package continues to validate key names on startup — harmless.

`start.sh` gains:
```bash
export BEATRICE_TRAINER_ROOT="$PROJECT_ROOT"
```

`download_assets.py` updated: default `--assets-dir` is `assets/beatrice2/` (was `beatrice/trainer/assets/`).

### 1.5 Database migration

```sql
-- profiles table: add pipeline column
ALTER TABLE profiles ADD COLUMN pipeline TEXT NOT NULL DEFAULT 'rvc';

-- new table for beatrice2 step-level loss tracking
CREATE TABLE IF NOT EXISTS beatrice_steps (
    id          TEXT PRIMARY KEY,
    profile_id  TEXT NOT NULL,
    step        INTEGER NOT NULL,
    loss_g      REAL,
    loss_d      REAL,
    loss_mel    REAL,
    loss_ap     REAL,
    loss_loud   REAL,
    loss_adv    REAL,
    loss_fm     REAL,
    trained_at  TEXT NOT NULL
);
```

`pipeline` is **locked at profile creation** — never updated after that. All existing rows default to `'rvc'`.

`ProfileOut` Pydantic model gains `pipeline: str = 'rvc'` field. All three SELECT paths in `routers/profiles.py` include the new column.

---

## Milestone 2 — Refactor `__main__.py` Into Logical Modules

### Rationale

`__main__.py` is a 4,563-line monolith containing:
- Model definitions (6+ distinct neural network classes)
- Training data utilities
- The training loop
- Evaluation / MOS scoring
- Checkpoint save/export logic

This makes the file impossible to navigate, import selectively, or test in isolation. The refactor splits it into focused modules with **zero logic changes** — no behaviour, no variable names, no import structure altered. Only the file that each definition lives in changes.

### Target module structure

```
backend/beatrice2/beatrice_trainer/
  __init__.py               ← re-exports public API (unchanged)
  __main__.py               ← entry point only: arg parsing + `if __name__ == "__main__"` block
  config.py                 ← dict_default_hparams, AttrDict, prepare_training_configs()
  layers/
    __init__.py
    conv.py                 ← CausalConv1d, WSConv1d, WSLinear
    attention.py            ← CrossAttention, ConvNeXtBlock, ConvNeXtStack
    vq.py                   ← VectorQuantizer
  models/
    __init__.py
    phone_extractor.py      ← FeatureExtractor, FeatureProjection, PhoneExtractor
    pitch_estimator.py      ← extract_pitch_features(), PitchEstimator
    vocoder.py              ← D4C synthesis classes, WaveformGenerator
    converter.py            ← ConverterNetwork (generator)
    discriminator.py        ← MultiPeriodDiscriminator and helpers
  data/
    __init__.py
    dataset.py              ← WavDataset, collate_fn, augmentation helpers
    filelist.py             ← get_training_filelist(), get_test_filelist()
  train/
    __init__.py
    loss.py                 ← loss computation functions, GradBalancer
    checkpoint.py           ← save/load checkpoint, export paraphernalia (dump_layer etc.)
    evaluation.py           ← QualityTester, MOS scoring, evaluation loop
    loop.py                 ← prepare_training(), main training loop
  io.py                     ← dump_params(), dump_layer() — shared binary I/O helpers
```

### Rules for the refactor

1. **No logic changes whatsoever.** Variable names, default values, algorithm, class signatures — all preserved verbatim.
2. **No new abstractions.** Do not introduce factory functions, base classes, or registries that do not already exist.
3. **No dependency inversions.** If class A used class B before the refactor, it still imports B directly after.
4. **Circular imports resolved only via local imports** (not by restructuring). Python allows `from .models.phone_extractor import PhoneExtractor` inside a function body if a circular import cannot otherwise be avoided.
5. **`__main__.py` entry point preserved.** Running `python3 -m beatrice_trainer` must still work identically.
6. **`__init__.py` re-exports everything** that was previously importable from `beatrice_trainer`. No external caller is broken.

### Module boundary map

| Lines (approx) | Content | Target file |
|---|---|---|
| 1–56 | Imports, version constant, notebook helpers | `__main__.py` (kept) + `config.py` |
| 57–217 | `repo_root()`, `dict_default_hparams`, `AttrDict`, `prepare_training_configs()` | `config.py` |
| 218–285 | `dump_params()`, `dump_layer()` | `io.py` |
| 286–570 | `CausalConv1d`, `WSConv1d`, `WSLinear` | `layers/conv.py` |
| 571–840 | `CrossAttention`, `ConvNeXtBlock`, `ConvNeXtStack` | `layers/attention.py` |
| 841–990 | `VectorQuantizer` | `layers/vq.py` |
| 991–1399 | `FeatureExtractor`, `FeatureProjection`, `PhoneExtractor` | `models/phone_extractor.py` |
| 1400–1700 | `extract_pitch_features()`, `PitchEstimator` | `models/pitch_estimator.py` |
| 1701–2100 | D4C synthesis, WaveformGenerator | `models/vocoder.py` |
| 2101–2700 | `ConverterNetwork` | `models/converter.py` |
| 2701–3050 | `MultiPeriodDiscriminator` and helpers | `models/discriminator.py` |
| 3051–3200 | `QualityTester`, MOS scoring | `train/evaluation.py` |
| 3201–3420 | `GradBalancer`, loss functions | `train/loss.py` |
| 3421–3600 | `WavDataset`, `collate_fn`, augmentation | `data/dataset.py` |
| 3601–3720 | `get_training_filelist()`, `get_test_filelist()` | `data/filelist.py` |
| 3721–4060 | `prepare_training()` | `train/loop.py` |
| 4061–4563 | Training loop, save/export, scheduler step | `train/loop.py` (continued) + `train/checkpoint.py` |

### Verification

After refactor, all of these must pass:
```bash
# Module imports cleanly
python3 -c "from backend.beatrice2.beatrice_trainer.models.phone_extractor import PhoneExtractor; print('ok')"

# Entry point still works
python3 -m backend.beatrice2.beatrice_trainer --help

# No import errors across the package
python3 -c "import backend.beatrice2.beatrice_trainer"
```

---

## Milestone 3 — Training Wrapper + Subprocess Management

### `backend/beatrice2/training.py`

Mirrors `backend/app/training.py` for the Beatrice 2 pipeline. Key differences:

| | RVC (`app/training.py`) | Beatrice 2 (`beatrice2/training.py`) |
|---|---|---|
| Subprocess command | `python3 train.py -e {exp_dir} ...` | `python3 -m beatrice_trainer -d {data_dir} -o {out_dir} -c {config}` |
| Progress unit | Epoch | Step |
| Checkpoint resume | `G_latest.pth` exists | `checkpoint_latest.pt.gz` exists |
| Stdout progress | `====> Epoch: N` | tqdm step counter |
| DB writer | `TrainingDBWriter` | `BeatriceDBWriter` → `beatrice_steps` table |
| CUDA gate | None | Check `torch.cuda.is_available()` before start |
| Preprocess step | RVC preprocessing scripts | `preprocess.py` splits audio into 4s chunks at 24kHz |

**Preprocess step**: Beatrice 2 requires audio split into ~4s chunks at 24kHz, one subdir per speaker. This is run synchronously (fast) before launching the training subprocess. Input: `profile_dir/audio/*.wav`. Output: `profile_dir/beatrice2_data/{profile_name}/*.wav`.

**Config file written at start** (`{out_dir}/config.json`):
```json
{
  "phone_extractor_file": "assets/beatrice2/pretrained/122_checkpoint_03000000.pt",
  "pitch_estimator_file": "assets/beatrice2/pretrained/104_3_checkpoint_00300000.pt",
  "pretrained_file":      "assets/beatrice2/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz",
  "in_ir_wav_dir":        "assets/beatrice2/ir",
  "in_noise_wav_dir":     "assets/beatrice2/noise",
  "in_test_wav_dir":      "assets/beatrice2/test",
  "n_steps":              10000,
  "batch_size":           8,
  "use_amp":              true,
  "num_workers":          4,
  "save_interval":        2000,
  "evaluation_interval":  2000,
  "record_metrics":       false
}
```

`record_metrics: false` disables UTMOS evaluation (avoids torch.hub download and internet requirement during training). Can be enabled optionally.

**Stdout parsing** — tqdm emits lines like:
```
Training:  20%|██        | 2000/10000 [10:00<40:00,  2.00it/s]
```
Regex: `r'(\d+)/(\d+)\s+\[' ` → extract step / total.

Loss values are written periodically via `dict_scalars` to `{out_dir}/scalars.jsonl` — one JSON object per line. The `BeatriceDBWriter` tails this file and inserts rows into `beatrice_steps`.

**Cancel**: SIGTERM → trainer saves `checkpoint_latest.pt.gz` at current step → exits cleanly.

---

## Milestone 4 — Inference Engine (Offline)

### `backend/beatrice2/inference.py`

```python
class BeatriceInferenceEngine:
    """Loads checkpoint_latest.pt.gz and runs voice conversion."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        # Load gzip checkpoint
        # Reconstruct PhoneExtractor, PitchEstimator, ConverterNetwork from state_dict
        # Set all modules to eval(), move to device

    def convert(
        self,
        audio_16k: np.ndarray,
        target_speaker_id: int = 0,
        pitch_shift_semitones: float = 0.0,
        formant_shift_semitones: float = 0.0,
    ) -> np.ndarray:  # float32 at 24 kHz
        # Use ConverterNetwork.inference() method
        # Block processing handled by caller (offline: whole file; realtime: per block)

    def unload(self):
        del self.phone_extractor, self.pitch_estimator, self.net_g
        torch.cuda.empty_cache()
```

**Python inference path**: Load `checkpoint_latest.pt.gz` → reconstruct full training model → call `net_g.inference()`. This is equivalent to running the trained model in eval mode. The `.bin` binary dump files are for the C++ runtime and are not used here.

**Output SR**: 24 kHz. The offline router writes the output at 24 kHz; the realtime worker resamples 24k → 48k for output.

**Integration in `routers/offline.py`**:
```python
if profile.pipeline == 'beatrice2':
    engine = _get_or_load_beatrice_engine(profile.model_path, device)
    result_audio = _run_beatrice_offline(audio_16k, engine, pitch_shift, formant_shift)
    tgt_sr = 24000
else:
    # existing RVC path
```

Model is cached in a module-level dict keyed by `profile_id` (same as RVC `_rvc_instances`).

---

## Milestone 5 — API Contract Updates

### Profiles

`POST /api/profiles/`:
- New form field: `pipeline: str = Form('rvc')`  — `'rvc'` | `'beatrice2'`
- Validation: if `'beatrice2'`, skip embedder/vocoder validation; check CUDA available
- Stored in `profiles.pipeline` column

`GET /api/profiles/` and `GET /api/profiles/{id}`:
- `ProfileOut` includes `pipeline: str`

### Training

`POST /api/training/start`:
- Read `profile.pipeline` from DB
- Route to RVC or Beatrice 2 training path
- For Beatrice 2: `epochs` param interpreted as `n_steps`

`GET /api/training/status/{profile_id}`:
- Unified response with `pipeline` field
- Beatrice 2: `step`, `total_steps`, `progress_pct`
- RVC: `epoch`, `total_epochs`, `progress_pct` (backward compat)

`GET /api/training/losses/{profile_id}`:
- Beatrice 2: queries `beatrice_steps` table
- Returns `{step, loss_mel, loss_g, loss_d, loss_ap, loss_adv, loss_fm}[]`

### Offline convert

No client-side change — `pipeline` derived from profile on server.
`done` SSE event adds `output_sr: 24000` for Beatrice 2 (vs `32000` for RVC).

### Realtime session

`POST /api/realtime/session/start`:
- New optional fields: `pitch_shift_semitones: float = 0.0`, `formant_shift_semitones: float = 0.0`
- Only used when profile pipeline is `beatrice2`; ignored otherwise

---

## Milestone 6 — Frontend: Profile Creation + Training UI

### Profile creation (`frontend/app/page.tsx`)

Add engine selector as the **first step** in the "New Profile" form, before the audio upload:

```
Engine
  ● RVC (Recommended)          ← indigo tile
    Works on MPS, CPU, CUDA
    Embedder + vocoder choice
    Epoch-based training
  ○ Beatrice 2                 ← amber tile
    CUDA GPU required
    Source-filter vocoder
    Step-based training
```

When Beatrice 2 is selected:
- Hide embedder / vocoder selectors
- Show CUDA warning badge if server reports `has_cuda: false`
- `pipeline` field appended to FormData on submit

Profile card badge:
- RVC: `◈ RVC` in indigo (existing embedder/vocoder badges unchanged)
- Beatrice 2: `◈ B2` in amber; no embedder/vocoder badges

### Training page (`frontend/app/training/page.tsx`)

Beatrice 2 profile selected:
- **Steps slider** replaces epoch slider (range 1000–50000, default 10000, step 500)
- **Progress bar** shows `step / total_steps`
- **Loss chart**: lines for `loss_mel` (primary), `loss_g`, `loss_d`, `loss_adv`, `loss_fm`, `loss_ap`
- **No embedder/vocoder display** (not applicable)
- **CUDA notice** banner: "Beatrice 2 requires a CUDA GPU. Training will fail on CPU/MPS."
- `total_epochs_trained` for Beatrice 2 stored as `total_steps_trained` in a separate display path

---

## Milestone 7 — Realtime Inference Integration

### `backend/app/_realtime_worker.py`

After loading the profile, branch on `pipeline`:

```python
if profile["pipeline"] == "beatrice2":
    engine = _load_beatrice_engine(profile["model_path"], device)
    # Block processing loop uses engine.convert() instead of rvc.infer()
    # Output 24kHz → resample to 48kHz for output
    # pitch_shift and formant_shift from session params
else:
    # existing RVC rtrvc path (unchanged)
```

Block size remains 200ms. Output SR difference (24k vs 32k) handled in the 32k/24k → 48k resample step.

SOLA crossfade, noise reduction, RMS matching, gate — all apply identically regardless of pipeline.

### Realtime page UI

Beatrice 2 profile connected:
- Show `pitch_shift_semitones` slider (−12 to +12 semitones, step 0.5)
- Show `formant_shift_semitones` slider (−3 to +3 semitones, step 0.25)
- Hide `f0_up_key`, `index_rate`, `f0_method` (not applicable)
- `pushParams` sends updated pitch/formant shift to worker via `update_params` command

---

## Milestone 8 — Export / Import Beatrice 2 Format

### Export (`GET /api/profiles/{id}/export`)

For `pipeline == 'beatrice2'`, zip layout:

```
manifest.json              ← format_version=2, pipeline="beatrice2"
audio/*.wav                ← original training audio
checkpoint_latest.pt.gz    ← latest checkpoint (for resume)
paraphernalia_*/           ← most recent paraphernalia dir (inference binaries)
  phone_extractor.bin
  pitch_estimator.bin
  waveform_generator.bin
  speaker_embeddings.bin
  embedding_setter.bin
  *.toml
train_log.txt
```

### Import (`POST /api/profiles/import`)

Read `manifest.json["pipeline"]`:
- `"rvc"`: existing import path unchanged
- `"beatrice2"`: extract to new profile dir, register in DB with `pipeline='beatrice2'`

---

## File Change Map

### New files
```
backend/beatrice2/__init__.py
backend/beatrice2/beatrice_trainer/    ← moved + refactored trainer (M1 + M2)
backend/beatrice2/training.py          ← BeatriceTrainingJob wrapper (M3)
backend/beatrice2/inference.py         ← BeatriceInferenceEngine (M4)
backend/beatrice2/preprocess.py        ← audio preprocessing helper
backend/beatrice2/download_assets.py   ← asset downloader (updated paths)
assets/beatrice2/                      ← pretrained + ir + noise + test (downloaded)
docs/BEATRICE2_INTEGRATION.md          ← this document
```

### Modified files
```
backend/app/db.py                   ← migration: pipeline col, beatrice_steps table
backend/app/routers/profiles.py     ← pipeline field in create/read
backend/app/routers/training.py     ← pipeline-aware start/status/cancel/losses
backend/app/routers/offline.py      ← pipeline-aware convert
backend/app/routers/realtime.py     ← beatrice2 session params
backend/app/_realtime_worker.py     ← beatrice2 inference branch
frontend/app/page.tsx               ← engine selector, profile card badge
frontend/app/training/page.tsx      ← step-based UI branch for beatrice2
frontend/app/realtime/page.tsx      ← pitch/formant sliders for beatrice2
frontend/app/offline/page.tsx       ← pitch/formant params for beatrice2
start.sh                            ← export BEATRICE_TRAINER_ROOT=$PROJECT_ROOT
```

### Deleted files
```
beatrice/    ← entire directory; code migrated to backend/beatrice2/
```

---

## Key Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CUDA-only training | Users on MPS/CPU cannot train Beatrice 2 | Gate in UI + API with clear error; document CUDA requirement |
| UTMOS torch.hub download | Training evaluation requires internet access | `record_metrics: false` in default config; document opt-in |
| Binary inference format | `.bin` files not loadable with `torch.load()` | Use `.pt.gz` checkpoint path for Python inference instead |
| Augmentation data size | ~12 MB IR/noise required before training starts | Setup page download flow; clear error if missing |
| 24 kHz output SR mismatch | Offline output at 24k vs RVC 32k; realtime resample path changes | Clearly label output SR in UI; test 24→48 resample path |
| `n_speakers=1` always | Beatrice 2 is multi-speaker by design | Create `data_dir/{profile_name}/` subdir as the single speaker — works correctly |
| Circular imports in refactor | Splitting `__main__.py` may create import cycles | Resolve with local imports inside function bodies only; never restructure |
| Monolith refactor regression | Silent behaviour change | Strict: copy blocks verbatim, no renames; verify with import smoke tests after each module |

---

## Non-Goals (Out of Scope for This Branch)

- MPS support for Beatrice 2 (CUDA only; documented as architecturally infeasible without deep patching)
- Fine-tuning from a custom pretrained Beatrice 2 base (uses official LibriTTS base only)
- Multi-speaker voice mixing in Beatrice 2 profiles
- Real-time UTMOS quality scoring in the UI
- Beatrice 2 FAISS index (not applicable — no retrieval augmentation)
