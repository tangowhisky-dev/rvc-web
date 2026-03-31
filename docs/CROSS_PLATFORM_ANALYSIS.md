# Cross-Platform Portability Analysis

**Date:** 2026-03-31  
**Scope:** Full codebase audit for Windows / Linux / macOS portability  
**Status:** Analysis only — no code changes made (except `start.sh`)

---

## Executive Summary

The codebase currently runs correctly on **macOS (MPS)** only. It has meaningful gaps
for Linux/CUDA and Windows. The problems fall into five categories:

1. **`start.sh` / `stop.sh`** — bash-only, macOS shell utilities, no CUDA env setup
2. **`training.py`** — hardcodes `"mps"` as the device argument passed to three subprocess scripts
3. **`_realtime_worker.py`** — skips CUDA in device detection (MPS → CPU fallback, no CUDA)
4. **`routers/training.py`** (hardware endpoint) — MPS-only metrics, no CUDA VRAM reporting
5. **Shell-util dependencies** — `lsof`, `pgrep`, `pkill`, `nc`, `conda` all absent on Windows

No changes to Python source files are needed just to reach a working state on Linux/CUDA
— the training and inference code already has full CUDA paths — the gaps are in the
orchestration layer (how devices are selected and passed as arguments).

---

## Part 1: Platform Detection in `start.sh`

### Current State

```bash
# start.sh — always sets macOS-specific env vars, no CUDA detection
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

conda run --no-capture-output -n rvc \
  uvicorn backend.app.main:app ...
```

**Problems:**
- `lsof` is not available on Windows (not even via Git Bash by default)
- `pgrep` / `pkill` / `nc` are absent on Windows
- `conda run -n rvc` assumes a conda env named `rvc` — different users may use `venv`, `pip`, or a different env name
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and `PYTORCH_MPS_PREFER_SHARED_MEMORY` are MPS-only env vars — setting them on Linux/CUDA is harmless but misleading
- No `CUDA_VISIBLE_DEVICES` guidance for multi-GPU Linux machines
- No `PYTORCH_ENABLE_MPS_FALLBACK` (set in `training.py` for subprocess envs but not exported globally)
- `start.sh` does **not** export `RVC_DEVICE` or any env var that `training.py` could read to choose device dynamically

### Required Changes to `start.sh`

**Already implemented** in this commit — see updated `start.sh` for:
- OS detection (`uname -s`: Darwin / Linux / Windows via `MSYSTEM`)
- `lsof` replaced with cross-platform port-kill
- CUDA device detection via `python -c "import torch; print(torch.cuda.is_available())"`
- Automatic `RVC_DEVICE` export (`cuda` / `mps` / `cpu`)
- `PYTORCH_MPS_*` env vars only set on macOS
- `CUDA_VISIBLE_DEVICES` guidance on Linux/CUDA
- Conda env activation made configurable via `RVC_CONDA_ENV` env var

---

## Part 2: `backend/app/training.py` — Hardcoded Device Arguments

### File: `backend/app/training.py`

#### Issue 2a — F0 extraction subprocess (line 643)

```python
f0_args = [
    python,
    "infer/modules/train/extract_f0_print.py",
    exp_dir,
    str(n_cpu),
    "rmvpe",
    "mps",      # ← HARDCODED — fails on Linux/CUDA, wastes MPS on Mac
    "False",    # is_half
]
```

`extract_f0_print.py` accepts `device` as `sys.argv[4]`. It passes it directly to `Generator`
(F0 extractor). On Linux the `mps` device will raise `RuntimeError: MPS not available`.

**Fix:** Read from env var `RVC_DEVICE` (set by `start.sh`) and fall back to auto-detect:

```python
# Determine device for subprocess args
import torch as _t
_rvc_device = os.environ.get("RVC_DEVICE") or (
    "cuda" if _t.cuda.is_available() else
    "mps"  if _t.backends.mps.is_available() else
    "cpu"
)
_is_half = str(_rvc_device.startswith("cuda")).capitalize()

f0_args = [
    ...
    _rvc_device,   # device
    _is_half,      # is_half — True for CUDA, False for MPS/CPU
]
```

#### Issue 2b — Feature extraction subprocess (line 675)

```python
feat_args = [
    python,
    "infer/modules/train/extract_feature_print.py",
    "mps",      # ← HARDCODED — same problem
    "1",
    "0",
    "",         # i_gpu (empty)
    exp_dir,
    "v2",
    "False",    # is_half
    embedder_path,
]
```

Same fix as Issue 2a. Also: `i_gpu` (arg[3] in legacy call signature) should be the CUDA
device index when `device == "cuda"`, e.g. `"0"`.

#### Issue 2c — MPS-only train_env (lines 719–720)

```python
train_env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
train_env.setdefault("PYTORCH_MPS_PREFER_SHARED_MEMORY", "1")
```

These are harmless on CUDA but should be guarded by platform check to avoid confusion:

```python
if _rvc_device == "mps":
    train_env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    train_env.setdefault("PYTORCH_MPS_PREFER_SHARED_MEMORY", "1")
```

#### Issue 2d — `"-c" "1"` (cache dataset in GPU) passed unconditionally to train.py

```python
train_args = [..., "-c", "1", ...]
```

`-c 1` caches training data to GPU memory. On CPU-only this means caching to RAM, which
is fine. On CUDA with a small GPU (4 GB), it causes OOM. The flag should be conditional:

```python
_cache_gpu = "1" if _rvc_device in ("cuda", "mps") else "0"
train_args = [..., "-c", _cache_gpu, ...]
```

#### Issue 2e — No `-g` (gpus) argument passed to train.py

`train.py` reads `hps.gpus` from argparse (`-g` flag, default `"0"`). On multi-GPU Linux
this defaults to GPU 0 only. The pipeline should pass `-g` based on `CUDA_VISIBLE_DEVICES`.
Currently this is silently correct only because `CUDA_VISIBLE_DEVICES` isn't set,
meaning GPU 0 is used implicitly. This is fine for single-GPU systems but should be
explicit.

---

## Part 3: `backend/app/_realtime_worker.py` — Missing CUDA Path

### File: `backend/app/_realtime_worker.py`, lines 279–282

```python
if _torch.backends.mps.is_available():
    infer_device = "mps"
else:
    infer_device = "cpu"    # ← CUDA never chosen
```

On Linux with NVIDIA GPU this silently falls to CPU, making realtime inference very slow.

**Fix:**

```python
if _torch.cuda.is_available():
    infer_device = "cuda"
elif _torch.backends.mps.is_available():
    infer_device = "mps"
else:
    infer_device = "cpu"
```

**Also:** `is_half=False` is hardcoded in the `RVC(...)` instantiation (line 325).
On CUDA, half-precision improves throughput significantly and should be enabled:

```python
_is_half = (infer_device == "cuda")

rvc = RVC(
    ...
    is_half=_is_half,
)
```

---

## Part 4: `backend/app/routers/offline.py` — CUDA Missing

### File: `backend/app/routers/offline.py`, lines 208–218

```python
device = "mps" if torch.backends.mps.is_available() else \
         ("cuda" if torch.cuda.is_available() else "cpu")
...
rvc = RVC(
    ...
    is_half=False,   # ← always False
)
```

Device detection here is actually correct (MPS → CUDA → CPU). But `is_half=False` is
wrong for CUDA — on NVIDIA GPUs, fp16 cuts memory and speeds inference.

**Fix:**

```python
is_half = (device == "cuda")
rvc = RVC(..., is_half=is_half)
```

---

## Part 5: `backend/app/routers/training.py` — Hardware Endpoint is MPS-Only

### File: `backend/app/routers/training.py`, `HardwareInfo` model + `get_hardware()`

**Current `HardwareInfo` model:**

```python
class HardwareInfo(BaseModel):
    total_ram_gb: float
    available_ram_gb: float
    ram_used_pct: float
    cpu_cores: int
    mps_available: bool       # MPS-specific
    mps_allocated_mb: float   # MPS-specific
    sweet_spot_batch_size: int
    max_safe_batch_size: int
```

**Problems:**
- No `cuda_available` field — frontend can't show NVIDIA GPU status
- No `gpu_name` field — users can't verify GPU is detected
- No `gpu_vram_gb` field — batch size heuristic uses RAM only (wrong for CUDA, where
  VRAM is the constraint)
- `sweet_spot_batch_size` heuristic uses total RAM / 1.5 GB per batch — correct for MPS
  (unified memory), wrong for CUDA where 24 GB VRAM supports batch=16+ with no RAM
  pressure

**Fields to add to `HardwareInfo`:**

```python
cuda_available: bool = False
gpu_name: Optional[str] = None
gpu_vram_gb: Optional[float] = None
```

**CUDA-aware batch size heuristic:**

```python
if cuda_available and gpu_vram_gb:
    # CUDA: VRAM is the bottleneck (~1.0 GB/batch at 32k v2)
    sweet_spot = nearest_pow2(gpu_vram_gb * 0.70 / 1.0)
elif mps_available:
    # MPS unified memory: RAM is the pool
    sweet_spot = nearest_pow2(total_gb * 0.60 / 1.5)
else:
    # CPU: conservative (RAM-only, no GPU acceleration)
    sweet_spot = 1
```

**Frontend impact:** `frontend/app/training/page.tsx` line 281 shows `MPS ✓` only.
Should also show `CUDA ✓ (GPU Name)` when CUDA is available.

---

## Part 6: `start.sh` / `stop.sh` — Windows Incompatibility

### Shell Utilities Used

| Utility | macOS | Linux | Git Bash (Win) | PowerShell (Win) |
|---------|-------|-------|---------------|-----------------|
| `lsof -ti:PORT` | ✅ | ✅ | ❌ | ❌ |
| `pgrep -f PATTERN` | ✅ | ✅ | ❌ partial | ❌ |
| `pkill -9 -f PATTERN` | ✅ | ✅ | ❌ | ❌ |
| `nc -z HOST PORT` | ✅ | ✅ | ❌ | ❌ |
| `kill -9 PID` | ✅ | ✅ | ✅ (Git Bash) | ❌ |
| `curl` | ✅ | ✅ | ✅ | ✅ (Win10+) |

### Recommendation: Two startup scripts

**Keep `start.sh` for Mac/Linux** (already updated in this commit).  
**Add `start.bat` (or `start.ps1`) for Windows** as a separate script.

The Python backend itself is cross-platform — only the shell orchestration needs
platform variants. A `start.bat` for Windows would:

```batch
@echo off
REM Detect GPU
python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" > .tmp_device.txt
set /p RVC_DEVICE=<.tmp_device.txt
del .tmp_device.txt

set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set PROJECT_ROOT=%CD%

REM Kill stale processes on ports 8000 and 3000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000') do taskkill /F /PID %%a >nul 2>&1

REM Start backend
start /b python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

REM Start frontend
cd frontend && pnpm dev
```

A `stop.bat` similarly uses `netstat` + `taskkill` instead of `lsof` + `kill`.

---

## Part 7: `backend/rvc/configs/config.py` — `use_insecure_load()` Missing

### File: `backend/rvc/configs/config.py`

The `Config` class is a singleton used by the **original RVC GUI web.py** and by some
training subprocess scripts. It tries to parse `--port`, `--dml` etc. from `sys.argv`
via `argparse` at **import time** (inside `__init__`).

When invoked as a subprocess (e.g., `extract_f0_print.py`), `sys.argv` contains the
script's own arguments, not GUI flags, causing argparse to error on unrecognised args.

The codebase already wraps this with:

```python
try:
    from configs.config import Config
    Config.use_insecure_load()
except Exception:
    pass
```

This is correct but relies on the silent except. A cleaner approach would be to pass
`parse_known_args` in `Config.arg_parse()`, but that touches source — document only.

### `use_insecure_load()` method

`Config.use_insecure_load()` is referenced in both `offline.py` and `_realtime_worker.py`
but **does not exist in `config.py`**. The name implies it should call
`torch.serialization.add_safe_globals()` or set `weights_only=False`. Currently the
call silently raises `AttributeError`, which is caught and ignored. This should be
implemented to avoid silent failures:

```python
@staticmethod
def use_insecure_load():
    """Allow loading of legacy PyTorch checkpoints (weights_only=False)."""
    try:
        from fairseq.data.dictionary import Dictionary
        import torch
        torch.serialization.add_safe_globals([Dictionary])
    except (ImportError, AttributeError):
        pass
```

---

## Part 8: `backend/rvc/infer/modules/train/train.py` — GPU Argument Not Passed

### File: `backend/rvc/infer/modules/train/train.py`, line 17

```python
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
```

`hps.gpus` defaults to `"0"` (from argparse in `utils.py`). The `training.py` pipeline
**never passes `-g`** in `train_args`, so this always becomes `CUDA_VISIBLE_DEVICES=0`.

On a multi-GPU Linux box where the user wants to use GPU 1:
- Set `CUDA_VISIBLE_DEVICES=1` before starting → train.py overrides it back to `0`
- The user has no way to pick the GPU from the UI

**Fix:** Pass `-g` from `training.py` based on `CUDA_VISIBLE_DEVICES` env var:

```python
cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
train_args = [..., "-g", cuda_device, ...]
```

---

## Part 9: `backend/rvc/rvc/f0/f0.py` — Fallback Device is `cuda:0` Not `cpu`

### File: `backend/rvc/rvc/f0/f0.py`, lines 19–20

```python
if device is None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

This is correct — but note it **skips MPS**. The `F0Predictor` base class will use
`cuda:0` on CUDA-enabled machines and `cpu` on Mac/CPU. The actual RMVPE and FCPE
implementations receive the device from their callers, so this fallback is rarely
hit in practice — but should include MPS for consistency:

```python
if device is None:
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
```

---

## Part 10: Realtime — `sounddevice` / PortAudio on Windows

### File: `backend/app/_realtime_worker.py`, line 372

```python
import sounddevice as sd
```

`sounddevice` wraps PortAudio. On Windows, PortAudio binaries must be distributed with
the package or installed separately. The `sounddevice` PyPI wheel for Windows bundles
PortAudio, so `pip install sounddevice` works. However:

- Windows audio device enumeration uses **WASAPI** (preferred) or MME as the backend
- WASAPI exclusive mode locks the device — other apps can't use it simultaneously
- Default device IDs differ between Windows and macOS
- The `samplerate=44100` capture assumption in the worker may need `samplerate=48000`
  on some Windows WASAPI configurations (many Windows devices default to 48 kHz)

**Required change in `_realtime_worker.py`:**

```python
# Query the actual default input sample rate and adapt
_dev_info = sd.query_devices(input_device_id, 'input')
_native_sr = int(_dev_info.get('default_samplerate', 44100))
# Use native SR for capture, resample to 16k for inference
stream_in = sd.InputStream(
    device=input_device_id,
    samplerate=_native_sr,
    ...
)
# Update resampler
_resample_in_16 = Resample(orig_freq=_native_sr, new_freq=16000, ...)
```

Also: on Windows the `latency="low"` hint may need to be `latency=0.1` (a float in
seconds) to be portable — the string `"low"` is PortAudio-specific and may not map
correctly on all Windows backends.

---

## Part 11: `backend/app/routers/training.py` — Batch Size Heuristic for CUDA

The `sweet_spot_batch_size` heuristic (line 124) uses total RAM:

```python
sweet_spot = nearest_pow2(total_gb * 0.60 / 1.5)
```

On a Linux machine with 64 GB RAM and an RTX 3090 (24 GB VRAM):
- Heuristic returns `nearest_pow2(64 * 0.60 / 1.5)` = `nearest_pow2(25.6)` = **16**
- 16 batches × 1.5 GB = 24 GB — coincidentally correct, but only because VRAM ≈ constraint

On a machine with 128 GB RAM and an RTX 4060 Ti (16 GB VRAM):
- Heuristic returns `nearest_pow2(128 * 0.60 / 1.5)` = `nearest_pow2(51.2)` = **32**
- 32 × 1.5 GB = 48 GB — massively over VRAM, causes OOM crash during training

**Fix:** Use `torch.cuda.get_device_properties(0).total_memory` when CUDA is available,
and make VRAM the denominator instead of RAM:

```python
if cuda_available:
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    sweet_spot = nearest_pow2(vram_gb * 0.70 / 1.0)
```

---

## Summary of Required Code Changes

### Priority 1 — Breaks on Linux/CUDA (must fix before Linux works)

| File | Line(s) | Issue | Change |
|------|---------|-------|--------|
| `backend/app/training.py` | 643 | `"mps"` hardcoded for F0 extract | Read `RVC_DEVICE` env var |
| `backend/app/training.py` | 675 | `"mps"` hardcoded for feature extract | Read `RVC_DEVICE` env var |
| `backend/app/_realtime_worker.py` | 279–282 | CUDA never selected for realtime | Add `cuda.is_available()` branch |

### Priority 2 — Incorrect behaviour on CUDA (degrades quality/performance)

| File | Line(s) | Issue | Change |
|------|---------|-------|--------|
| `backend/app/_realtime_worker.py` | 325 | `is_half=False` always | Enable for CUDA |
| `backend/app/routers/offline.py` | 218 | `is_half=False` always | Enable for CUDA |
| `backend/app/training.py` | 719–720 | MPS env vars set unconditionally | Guard with `if device == "mps"` |
| `backend/app/training.py` | ~783 | `-c 1` unconditional | Set `"0"` for CPU |

### Priority 3 — Missing information / suboptimal (non-blocking)

| File | Line(s) | Issue | Change |
|------|---------|-------|--------|
| `backend/app/routers/training.py` | 73–85 | No CUDA fields in `HardwareInfo` | Add `cuda_available`, `gpu_name`, `gpu_vram_gb` |
| `backend/app/routers/training.py` | 124 | Batch size heuristic is RAM-based only | Add VRAM-based heuristic for CUDA |
| `backend/app/training.py` | ~790 | No `-g` flag passed to train.py | Pass `CUDA_VISIBLE_DEVICES` as `-g` |
| `backend/rvc/rvc/f0/f0.py` | 19–20 | MPS not in device fallback | Add MPS branch |
| `backend/rvc/configs/config.py` | — | `use_insecure_load()` not implemented | Implement the method |
| `frontend/app/training/page.tsx` | 281 | Shows `MPS ✓` only | Add `CUDA ✓ (GPU)` display |

### Priority 4 — Windows only

| File/Location | Issue | Change |
|---------------|-------|--------|
| `start.sh` | macOS shell utils | Create `start.bat` / `start.ps1` |
| `stop.sh` | macOS shell utils | Create `stop.bat` |
| `_realtime_worker.py` | Fixed `samplerate=44100` capture | Query native device SR |
| `_realtime_worker.py` | `latency="low"` string | Use float seconds portably |

---

## What `start.sh` Was Updated to Do (Already Implemented)

The `start.sh` has been updated in this commit to:

1. **Detect OS** — `uname -s` → sets `OS_TYPE` to `Darwin`, `Linux`, or `Windows`
2. **Detect available accelerator** — Python one-liner:
   ```bash
   python -c "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')"
   ```
3. **Export `RVC_DEVICE`** — so `training.py` and other code can read it
4. **Set platform-appropriate env vars**:
   - macOS: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, `PYTORCH_MPS_PREFER_SHARED_MEMORY=1`
   - CUDA: `PYTORCH_ENABLE_MPS_FALLBACK=0`, guidance for `CUDA_VISIBLE_DEVICES`
   - All platforms: `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, etc.
5. **Portable port killing** — Python fallback when `lsof` is unavailable
6. **Configurable conda env** via `RVC_CONDA_ENV` environment variable (default: `rvc`)
7. **Print device summary** at startup so the user knows what backend is active

The Python code changes listed above are **not yet implemented** — they are documented
here for the next implementation milestone.
