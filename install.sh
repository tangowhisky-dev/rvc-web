#!/usr/bin/env bash
# =============================================================================
# install.sh — rvc-web environment setup
# Supports: macOS (MPS), Linux + CUDA, Linux CPU-only
# Requires: conda (Miniconda or Anaconda) in PATH
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override with environment variables before running
# ---------------------------------------------------------------------------
ENV_NAME="${RVC_CONDA_ENV:-rvc}"
PYTHON_VERSION="3.11"
NODE_MIN="18"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[install] $*"; }
warn()  { echo "[install] WARNING: $*" >&2; }
die()   { echo "[install] ERROR: $*" >&2; exit 1; }

need_cmd() { command -v "$1" &>/dev/null || die "'$1' not found — install it first"; }

# ---------------------------------------------------------------------------
# 1. Check prerequisites
# ---------------------------------------------------------------------------
info "Checking prerequisites..."
need_cmd conda

# Node.js — needed for frontend
if command -v node &>/dev/null; then
    NODE_VER=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VER" -lt "$NODE_MIN" ]; then
        warn "Node.js $NODE_VER found but ≥$NODE_MIN required. Upgrade before running start.sh."
    else
        info "Node.js $(node --version) ✓"
    fi
else
    warn "Node.js not found — install Node ≥$NODE_MIN before running start.sh."
fi

# pnpm
if ! command -v pnpm &>/dev/null; then
    info "pnpm not found — installing via npm..."
    npm install -g pnpm || warn "pnpm install failed — install manually: https://pnpm.io/installation"
else
    info "pnpm $(pnpm --version) ✓"
fi

# ---------------------------------------------------------------------------
# 2. Detect platform + compute device
# ---------------------------------------------------------------------------
PLATFORM="$(uname -s)"
ARCH="$(uname -m)"
CUDA_DEVICE="cpu"

case "$PLATFORM" in
    Darwin)
        info "Platform: macOS ($ARCH)"
        TORCH_INDEX=""   # PyTorch ships MPS support in the default PyPI index
        FAISS_PKG="faiss-cpu"
        ONNX_PKG="onnxruntime"
        ;;
    Linux)
        info "Platform: Linux ($ARCH)"
        # Probe for CUDA
        if command -v nvidia-smi &>/dev/null; then
            CUDA_VER=$(nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+" || echo "")
            if [ -n "$CUDA_VER" ]; then
                CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
                CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
                info "CUDA $CUDA_VER detected"
                if [ "$CUDA_MAJOR" -ge 13 ]; then
                    TORCH_INDEX=""      # PyPI default wheel ships with CUDA 13 support
                    CUDA_DEVICE="cuda"
                elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then
                    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
                    CUDA_DEVICE="cuda"
                elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then
                    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
                    CUDA_DEVICE="cuda"
                else
                    warn "CUDA $CUDA_VER is not supported (need 12.6+, 12.8, or 13.x) — falling back to CPU build"
                    TORCH_INDEX=""
                fi
            else
                info "nvidia-smi found but no CUDA version detected — using CPU build"
                TORCH_INDEX=""
            fi
        else
            info "No NVIDIA GPU detected — using CPU build"
            TORCH_INDEX=""
        fi
        if [ "$CUDA_DEVICE" = "cuda" ]; then
            FAISS_PKG="faiss-cpu"   # faiss-gpu has no cp311+ wheels on PyPI; faiss-cpu works for index building
            ONNX_PKG="onnxruntime-gpu"
        else
            FAISS_PKG="faiss-cpu"
            ONNX_PKG="onnxruntime"
        fi
        ;;
    *)
        die "Unsupported platform: $PLATFORM. Use install.bat on Windows."
        ;;
esac

# ---------------------------------------------------------------------------
# 2b. Install system audio libraries (PortAudio, libsndfile)
#     sounddevice and soundfile link against these at runtime.
#     Must be installed before pip wheels are built/verified.
# ---------------------------------------------------------------------------
info "Installing system audio libraries..."
case "$PLATFORM" in
    Linux)
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y libportaudio2 libsndfile1 > /dev/null
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y portaudio libsndfile > /dev/null
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm portaudio libsndfile > /dev/null
        else
            warn "Unknown package manager — install libportaudio2 and libsndfile1 manually before running start.sh"
        fi
        ;;
    Darwin)
        if command -v brew &>/dev/null; then
            brew install portaudio libsndfile --quiet
        else
            warn "Homebrew not found — install portaudio and libsndfile manually: brew install portaudio libsndfile"
        fi
        ;;
esac

# ---------------------------------------------------------------------------
# 3. Create / update conda environment
# ---------------------------------------------------------------------------
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    info "Conda env '${ENV_NAME}' already exists — updating..."
    conda install -n "$ENV_NAME" python="$PYTHON_VERSION" -y --quiet
else
    info "Creating conda env '${ENV_NAME}' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y --quiet
fi

# Activate for subsequent steps
# shellcheck disable=SC1091
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

info "Python: $(python --version)"

# ---------------------------------------------------------------------------
# 4. Resolve uv — prefer the standalone binary over python -m uv
#    so the conda env's Python doesn't need uv installed inside it.
# ---------------------------------------------------------------------------
# Search order:
#   1. uv already on PATH (global install, brew, cargo, etc.)
#   2. ~/.cargo/bin/uv   (cargo install uv)
#   3. ~/.local/bin/uv   (pip install --user uv or uv's own installer)
#   4. conda env's pip   (install uv into the env as a last resort)
UV_BIN=""
for candidate in \
    "$(command -v uv 2>/dev/null)" \
    "$HOME/.cargo/bin/uv" \
    "$HOME/.local/bin/uv" \
    "/usr/local/bin/uv"
do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        UV_BIN="$candidate"
        break
    fi
done

if [ -n "$UV_BIN" ]; then
    info "uv found: $UV_BIN"
    UV="$UV_BIN pip"
else
    # Fall back: install uv into the conda env, then use python -m uv
    info "uv not found globally — installing into conda env..."
    pip install uv --quiet
    UV="python -m uv pip"
fi

# ---------------------------------------------------------------------------
# 5. Install PyTorch (must come before other packages that depend on torch)
# ---------------------------------------------------------------------------
info "Installing PyTorch..."
if [ -n "$TORCH_INDEX" ]; then
    info "  Using index: $TORCH_INDEX"
    $UV install torch torchaudio --index-url "$TORCH_INDEX" --quiet
else
    $UV install torch torchaudio --quiet
fi

# ---------------------------------------------------------------------------
# 6. Install faiss (platform-specific)
# ---------------------------------------------------------------------------
info "Installing $FAISS_PKG..."
$UV install "$FAISS_PKG" --quiet

# ---------------------------------------------------------------------------
# 7. Install fairseq from One-sixth fork
# (must be after torch; build needs Cython + numpy already present)
# ---------------------------------------------------------------------------
info "Installing build prerequisites for fairseq..."
$UV install Cython numpy --quiet

info "Installing fairseq (One-sixth fork — Python 3.10+ compatible)..."
$UV install "fairseq @ git+https://github.com/One-sixth/fairseq.git" --quiet

# ---------------------------------------------------------------------------
# 8. Install remaining requirements (excluding torch, faiss, fairseq, onnx
#    — already handled above or platform-specific)
# ---------------------------------------------------------------------------
info "Installing remaining Python dependencies..."

# Build a filtered requirements list on the fly
TMPFILE="$(mktemp /tmp/rvc_req_XXXXXX.txt)"
grep -vE "^\s*#|^$|^torch$|^torchaudio|^faiss|^fairseq|^onnxruntime|^torch " requirements.txt \
    | grep -v "sys_platform\|platform_machine" \
    > "$TMPFILE"

$UV install -r "$TMPFILE" --quiet
rm -f "$TMPFILE"

# onnxruntime — correct variant for this platform
info "Installing $ONNX_PKG..."
$UV install "$ONNX_PKG" --quiet

# ---------------------------------------------------------------------------
# 9. Install frontend dependencies
# ---------------------------------------------------------------------------
info "Installing frontend dependencies (pnpm)..."
cd frontend
pnpm install --frozen-lockfile 2>/dev/null || pnpm install
cd ..

# ---------------------------------------------------------------------------
# 10. Verify key imports
# ---------------------------------------------------------------------------
info "Verifying key imports..."
python - <<'PYCHECK'
import sys
failures = []
checks = [
    "torch", "torchaudio", "fairseq", "faiss",
    "fastapi", "uvicorn", "librosa", "sounddevice",
    "soundfile", "numpy", "scipy", "sklearn",
    "torchcrepe", "torchfcpe",
]
for mod in checks:
    try:
        __import__(mod)
        print(f"  ✓ {mod}")
    except ImportError as e:
        print(f"  ✗ {mod}: {e}")
        failures.append(mod)

import torch
print(f"\n  torch={torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  MPS available:  {torch.backends.mps.is_available()}")

if failures:
    print(f"\n  FAILED imports: {failures}", file=sys.stderr)
    sys.exit(1)
PYCHECK

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
info ""
info "═══════════════════════════════════════════════════════"
info "  Installation complete"
info "  Conda env : ${ENV_NAME}"
info "  Device    : ${CUDA_DEVICE:-mps/cpu}"
info "  Run:  ./start.sh"
info "═══════════════════════════════════════════════════════"
