#!/usr/bin/env bash
# =============================================================================
# install.sh — rvc-web environment setup
# Supports: macOS (MPS), Linux + CUDA, Linux CPU-only
# Requires: uv (https://github.com/astral-sh/uv) in PATH
#           Install uv:  curl -LsSf https://astral.sh/uv/install.sh | sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override with environment variables before running
# ---------------------------------------------------------------------------
ENV_NAME="${RVC_ENV_NAME:-rvc}"
ENV_DIR="$HOME/.${ENV_NAME}"
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

# uv
if ! command -v uv &>/dev/null; then
    die "uv not found. Install it with:
    curl -LsSf https://astral.sh/uv/install.sh | sh
  Then restart your shell and re-run install.sh."
fi
info "uv $(uv --version) ✓"

# Node.js — needed for frontend
if command -v node &>/dev/null; then
    NODE_VER=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VER" -lt "$NODE_MIN" ]; then
        warn "Node.js $NODE_VER found but >=$NODE_MIN required. Upgrade before running start.sh."
    else
        info "Node.js $(node --version) ✓"
    fi
else
    warn "Node.js not found — install Node >=$NODE_MIN before running start.sh."
fi

# pnpm
if ! command -v pnpm &>/dev/null; then
    info "pnpm not found — installing via npm..."
    npm install -g pnpm || warn "pnpm install failed — install manually: https://pnpm.io/installation"
else
    info "pnpm $(pnpm --version) ✓"
fi

# git (needed for fairseq install from GitHub)
need_cmd git

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
                    TORCH_INDEX=""   # PyPI now ships native CUDA 13 builds
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
#     sounddevice links against PortAudio at runtime.
#     soundfile links against libsndfile at runtime.
#
#     NOTE: ffmpeg is NOT required as a system package.
#       - The `av` Python package bundles its own FFmpeg shared libs.
#       - pydub has been removed (offline output is always WAV now).
# ---------------------------------------------------------------------------
info "Installing system audio libraries (PortAudio, libsndfile)..."
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
# 3. Create / update uv virtual environment at ~/.rvc
# ---------------------------------------------------------------------------
info "Setting up uv environment '${ENV_NAME}' at ${ENV_DIR}..."

if [ -d "$ENV_DIR" ]; then
    info "Environment already exists — reusing (run 'rm -rf ${ENV_DIR}' to start fresh)"
else
    uv venv "$ENV_DIR" --python "$PYTHON_VERSION"
    info "Created uv venv at ${ENV_DIR}"
fi

# Resolve the Python interpreter inside the venv
RVC_PYTHON="${ENV_DIR}/bin/python"
[ -x "$RVC_PYTHON" ] || die "Python interpreter not found at ${RVC_PYTHON}"
info "Python: $($RVC_PYTHON --version)"

# Activate the venv for uv — setting VIRTUAL_ENV is the correct way to
# point `uv pip install` at a specific venv without the --python flag.
export VIRTUAL_ENV="$ENV_DIR"
export PATH="$ENV_DIR/bin:$PATH"
UV="uv pip install"

# ---------------------------------------------------------------------------
# 4. Install PyTorch (must come before other packages that depend on torch)
# ---------------------------------------------------------------------------
info "Installing PyTorch..."
if [ -n "${TORCH_INDEX:-}" ]; then
    info "  Using index: $TORCH_INDEX"
    $UV torch torchaudio --index-url "$TORCH_INDEX"
else
    $UV torch torchaudio
fi

# ---------------------------------------------------------------------------
# 5. Install faiss (platform-specific)
# ---------------------------------------------------------------------------
info "Installing $FAISS_PKG..."
$UV "$FAISS_PKG"

# ---------------------------------------------------------------------------
# 6. Install fairseq from One-sixth fork
#    Must be after torch; build needs Cython + numpy already present.
# ---------------------------------------------------------------------------
info "Installing build prerequisites for fairseq..."
$UV Cython numpy

info "Installing fairseq (One-sixth fork — Python 3.10+ compatible)..."
$UV "fairseq @ git+https://github.com/One-sixth/fairseq.git"

# ---------------------------------------------------------------------------
# 7. Install remaining requirements
#    Excludes: torch/torchaudio, faiss, fairseq, onnxruntime (platform-specific)
# ---------------------------------------------------------------------------
info "Installing remaining Python dependencies..."

TMPFILE="$(mktemp /tmp/rvc_req_XXXXXX.txt)"
grep -vE "^\s*#|^$" requirements.txt \
    | grep -vE "^torch$|^torchaudio|^faiss|^fairseq|^onnxruntime" \
    | grep -v "sys_platform\|platform_machine" \
    > "$TMPFILE"

$UV -r "$TMPFILE"
rm -f "$TMPFILE"

# onnxruntime — correct variant for this platform
info "Installing $ONNX_PKG..."
$UV "$ONNX_PKG"

# ---------------------------------------------------------------------------
# 8. Install frontend dependencies
# ---------------------------------------------------------------------------
info "Installing frontend dependencies (pnpm)..."
cd frontend
pnpm install --frozen-lockfile 2>/dev/null || pnpm install
cd ..

# ---------------------------------------------------------------------------
# 9. Verify key imports
# ---------------------------------------------------------------------------
info "Verifying key imports..."
"$RVC_PYTHON" - <<'PYCHECK'
import sys
failures = []
checks = [
    "torch", "torchaudio", "fairseq", "faiss",
    "fastapi", "uvicorn", "librosa", "sounddevice",
    "soundfile", "numpy", "scipy", "sklearn",
    "torchcrepe", "torchfcpe", "av",
    "noisereduce", "parselmouth", "pyworld",
    "tensorboard", "psutil",
]
for mod in checks:
    try:
        __import__(mod)
        print(f"  ok  {mod}")
    except ImportError as e:
        print(f"  ERR {mod}: {e}")
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
info "======================================================="
info "  Installation complete"
info "  venv      : ${ENV_DIR}"
info "  Python    : ${RVC_PYTHON}"
info "  Device    : ${CUDA_DEVICE}"
info "  Run:  ./start.sh"
info "======================================================="
