#!/usr/bin/env bash
# start.sh — cross-platform startup for rvc-web
# Supports: macOS (MPS), Linux (CUDA or CPU), Windows (Git Bash / MSYS2)
set -euo pipefail

# Anchor to rvc-web/ regardless of where the script is called from
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# 1. Detect OS
# ---------------------------------------------------------------------------
OS_TYPE="$(uname -s 2>/dev/null || echo "Windows")"
case "$OS_TYPE" in
  Darwin)  PLATFORM="mac"     ;;
  Linux)   PLATFORM="linux"   ;;
  MINGW*|MSYS*|CYGWIN*) PLATFORM="windows" ;;
  *)       PLATFORM="linux"   ;;  # WSL, BSD — treat as linux
esac
echo "[start] Platform: $PLATFORM ($OS_TYPE)"

# ---------------------------------------------------------------------------
# 2. Asset pre-check
#    Verify all required model weights are present before launching anything.
#    Missing assets cause cryptic errors deep in Python — catch them here.
# ---------------------------------------------------------------------------
echo "[start] Checking assets..."

ASSET_ERRORS=0

check_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "$path" ]; then
    echo "  [MISSING] $label"
    echo "            -> $path"
    ASSET_ERRORS=$((ASSET_ERRORS + 1))
  else
    echo "  [OK]      $label"
  fi
}

check_dir_nonempty() {
  local dir="$1"
  local pattern="$2"   # glob pattern to count, e.g. "*.pth"
  local label="$3"
  local min="${4:-1}"
  if [ ! -d "$dir" ]; then
    echo "  [MISSING] $label (directory not found)"
    echo "            -> $dir"
    ASSET_ERRORS=$((ASSET_ERRORS + 1))
    return
  fi
  local count
  count=$(find "$dir" -maxdepth 1 -name "$pattern" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$count" -lt "$min" ]; then
    echo "  [MISSING] $label (found $count/$min files matching $pattern)"
    echo "            -> $dir"
    ASSET_ERRORS=$((ASSET_ERRORS + 1))
  else
    echo "  [OK]      $label ($count files)"
  fi
}

echo ""
echo "  -- RVC core (required for all inference + training) --"
check_file "assets/hubert/hubert_base.pt"      "HuBERT base model"
check_file "assets/rmvpe/rmvpe.pt"             "RMVPE F0 estimator (PyTorch)"
check_file "assets/rmvpe/rmvpe.onnx"           "RMVPE F0 estimator (ONNX)"
check_dir_nonempty "assets/pretrained_v2" "*.pth" "RVC v2 pretrained weights" 12
check_dir_nonempty "assets/pretrained"   "*.pth" "RVC v1 pretrained weights" 12

echo ""
echo "  -- Embedder (default: spin-v2) --"
check_file "assets/spin-v2/config.json"        "SPIN-v2 embedder config"
check_file "assets/spin-v2/pytorch_model.bin"  "SPIN-v2 embedder weights"

echo ""
echo "  -- Speaker analysis (ECAPA-TDNN) --"
check_file "assets/ecapa/ecapa_tdnn.pt"        "ECAPA-TDNN speaker encoder"

echo ""
echo "  -- Beatrice 2 (required for B2 pipeline) --"
check_file "assets/beatrice2/pretrained/122_checkpoint_03000000.pt"                    "B2 phone extractor checkpoint"
check_file "assets/beatrice2/pretrained/104_3_checkpoint_00300000.pt"                  "B2 pitch estimator checkpoint"
check_file "assets/beatrice2/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz"  "B2 base model (LibriTTS-R 200)"
check_file "assets/beatrice2/utmos/utmos22_strong_step7459_v1.pt"                      "UTMOS MOS scorer"
check_dir_nonempty "assets/beatrice2/ir"    "*.flac" "Beatrice 2 impulse responses (IR)"  20
check_dir_nonempty "assets/beatrice2/noise" "*.flac" "Beatrice 2 background noise files"  20

echo ""
if [ "$ASSET_ERRORS" -gt 0 ]; then
  echo "  +-----------------------------------------------------------------+"
  echo "  |  $ASSET_ERRORS required asset(s) missing -- cannot start          |"
  echo "  +-----------------------------------------------------------------+"
  echo ""
  echo "  To fix:"
  echo ""
  echo "    RVC weights (manual download):"
  echo "      https://huggingface.co/lj1995/VoiceConversionWebUI"
  echo "      hubert_base.pt          -> assets/hubert/"
  echo "      rmvpe.pt + rmvpe.onnx   -> assets/rmvpe/"
  echo "      pretrained/*.pth        -> assets/pretrained/"
  echo "      pretrained_v2/*.pth     -> assets/pretrained_v2/"
  echo "      spin-v2/ directory      -> assets/spin-v2/"
  echo "      ecapa_tdnn.pt           -> assets/ecapa/"
  echo ""
  echo "    Beatrice 2 weights (automated, downloads 20 IR + 20 noise files):"
  echo "      python -m backend.beatrice2.download_assets"
  echo ""
  exit 1
fi

echo "  All assets present -- proceeding."
echo ""

# ---------------------------------------------------------------------------
# 3. Resolve Python interpreter
#    Priority: conda env 'rvc' → uv venv ~/.rvc
# ---------------------------------------------------------------------------
ENV_NAME="${RVC_ENV_NAME:-rvc}"
RVC_PYTHON=""
RVC_ENV_SOURCE=""

# 1) Try conda
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null)" || CONDA_BASE=""
  if [ -n "$CONDA_BASE" ]; then
    CONDA_PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
    if [ -x "$CONDA_PY" ]; then
      RVC_PYTHON="$CONDA_PY"
      RVC_ENV_SOURCE="conda env '${ENV_NAME}'"
    fi
  fi
fi

# 2) Fall back to uv venv at ~/.rvc
if [ -z "$RVC_PYTHON" ]; then
  UV_PY="$HOME/.${ENV_NAME}/bin/python"
  if [ -x "$UV_PY" ]; then
    RVC_PYTHON="$UV_PY"
    RVC_ENV_SOURCE="uv venv '$HOME/.${ENV_NAME}'"
  fi
fi

if [ -z "$RVC_PYTHON" ]; then
  echo "[start] ERROR: No Python environment named '${ENV_NAME}' found."
  echo "[start]        Tried:"
  echo "[start]          conda: \$(conda info --base)/envs/${ENV_NAME}/bin/python"
  echo "[start]          uv:    $HOME/.${ENV_NAME}/bin/python"
  echo "[start]        Run ./install.sh to create the environment."
  exit 1
fi

echo "[start] Python: $RVC_PYTHON  (via $RVC_ENV_SOURCE)"

# Detect available compute device
if "$RVC_PYTHON" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  export RVC_DEVICE="cuda"
elif "$RVC_PYTHON" -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
  export RVC_DEVICE="mps"
else
  export RVC_DEVICE="cpu"
fi
echo "[start] Compute device: $RVC_DEVICE"

# ---------------------------------------------------------------------------
# 4. Universal threading env vars
# ---------------------------------------------------------------------------
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---------------------------------------------------------------------------
# 5. Platform-specific PyTorch tuning
# ---------------------------------------------------------------------------
if [ "$RVC_DEVICE" = "mps" ]; then
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  export PYTORCH_MPS_PREFER_SHARED_MEMORY=1
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  echo "[start] MPS tuning applied"
elif [ "$RVC_DEVICE" = "cuda" ]; then
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "[start] CUDA_VISIBLE_DEVICES defaulted to 0 (set externally to override)"
  else
    echo "[start] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from environment)"
  fi
  export PYTORCH_ENABLE_MPS_FALLBACK=0
fi

# ---------------------------------------------------------------------------
# 6. Kill stale processes
# ---------------------------------------------------------------------------
kill_port() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null) || true
    if [ -n "$pids" ]; then
      echo "[start] Killing stale process(es) on port $port..."
      echo "$pids" | xargs kill -9 2>/dev/null || true
      sleep 1
      return
    fi
  fi
  if command -v fuser >/dev/null 2>&1; then
    local pids
    pids=$(fuser "$port/tcp" 2>/dev/null) || true
    if [ -n "$pids" ]; then
      echo "[start] Killing stale process(es) on port $port..."
      echo "$pids" | xargs kill -9 2>/dev/null || true
      sleep 1
      return
    fi
  fi
  python - "$port" <<'PYEOF' 2>/dev/null || true
import sys, subprocess
port = sys.argv[1]
try:
    out = subprocess.check_output(
        ["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL
    )
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            parts = line.split()
            pid = parts[-1]
            if pid.isdigit():
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
except Exception:
    pass
PYEOF
}

kill_next_dev() {
  if command -v pgrep >/dev/null 2>&1; then
    local pids
    pids=$(pgrep -f "next dev" 2>/dev/null) || true
    if [ -n "$pids" ]; then
      echo "[start] Killing stale next dev process(es)..."
      echo "$pids" | xargs kill -9 2>/dev/null || true
      sleep 1
    fi
  fi
}

kill_dangling() {
  if command -v pgrep >/dev/null 2>&1; then
    local pids
    pids=$(pgrep -f "train\.py\|spawn_main\|_realtime_worker\|extract_f0_print\|extract_feature_print\|preprocess\.py" 2>/dev/null) || true
    if [ -n "$pids" ]; then
      echo "[start] Killing dangling training/inference processes..."
      echo "$pids" | xargs kill -9 2>/dev/null || true
      sleep 1
    fi
  fi
}

kill_port 8000
kill_port 3000
kill_port 3001
kill_next_dev
kill_dangling

mkdir -p .pids logs data/profiles

# ---------------------------------------------------------------------------
# 6b. Sweep temp/ — delete files older than 1 hour
# ---------------------------------------------------------------------------
mkdir -p temp
echo "[start] Sweeping temp/ (files older than 1h)..."
find temp/ -maxdepth 1 -type f -mmin +60 -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# 7. PROJECT_ROOT
# ---------------------------------------------------------------------------
export PROJECT_ROOT="$(pwd)"
echo "[start] PROJECT_ROOT=$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 8. Detect all IP addresses and let user choose
# ---------------------------------------------------------------------------
echo "[start] Detecting network interfaces..."

IP_LIST="localhost"

if [ "$PLATFORM" = "mac" ]; then
  ADDRS=$(ifconfig 2>/dev/null | grep -oE 'inet [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | awk '{print $2}' | grep -v '^127\.0\.0\.1$' | sort -u)
  if [ -n "$ADDRS" ]; then
    IP_LIST="$IP_LIST"$'\n'"$ADDRS"
  fi
elif [ "$PLATFORM" = "linux" ]; then
  if command -v ip >/dev/null 2>&1; then
    ADDRS=$(ip -4 addr show 2>/dev/null | grep -oP 'inet \K[\d.]+' | grep -v '^127\.0\.0\.1$' | sort -u)
  else
    ADDRS=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^127\.0\.0\.1$' | sort -u)
  fi
  if [ -n "$ADDRS" ]; then
    IP_LIST="$IP_LIST"$'\n'"$ADDRS"
  fi
elif [ "$PLATFORM" = "windows" ]; then
  ADDRS=$(ipconfig 2>/dev/null | grep -oE 'IPv4 Address[^:]*: [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | grep -v '^127\.0\.0\.1$' | sort -u)
  if [ -n "$ADDRS" ]; then
    IP_LIST="$IP_LIST"$'\n'"$ADDRS"
  fi
fi

IP_COUNT=$(printf '%s\n' "$IP_LIST" | wc -l | tr -d ' ')

if [ "$IP_COUNT" -le 1 ]; then
  export NEXT_PUBLIC_API_URL="http://localhost:8000"
  echo "[start] No network interfaces found — using localhost only"
else
  echo ""
  echo "  Available network interfaces for frontend access:"
  echo ""

  IDX=1
  while IFS= read -r ip; do
    if [ "$ip" = "localhost" ]; then
      echo "    $IDX) $ip (local only)"
    else
      echo "    $IDX) $ip"
    fi
    IDX=$((IDX + 1))
  done <<< "$IP_LIST"
  echo ""

  DEFAULT_CHOICE=1
  IDX=1
  while IFS= read -r ip; do
    if [ "$ip" != "localhost" ]; then
      DEFAULT_CHOICE=$IDX
      break
    fi
    IDX=$((IDX + 1))
  done <<< "$IP_LIST"

  read -rp "  Select interface [default: $DEFAULT_CHOICE]: " CHOICE
  CHOICE=${CHOICE:-$DEFAULT_CHOICE}

  if echo "$CHOICE" | grep -qE '^[0-9]+$' && [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le "$IP_COUNT" ]; then
    SELECTED_IP=$(echo "$IP_LIST" | sed -n "${CHOICE}p")
    export NEXT_PUBLIC_API_URL="http://${SELECTED_IP}:8000"
    echo "[start] Selected: $SELECTED_IP -> NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL"
  else
    echo "[start] Invalid selection — falling back to localhost"
    export NEXT_PUBLIC_API_URL="http://localhost:8000"
  fi
  echo ""
fi

# ---------------------------------------------------------------------------
# 9. Print device summary banner
# ---------------------------------------------------------------------------
echo "[start] -----------------------------------------------"
echo "[start] Device summary"
echo "[start]   OS:      $PLATFORM"
echo "[start]   Device:  $RVC_DEVICE"
"$RVC_PYTHON" - <<'PYEOF' 2>/dev/null || true
import torch, sys
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024**3
    print(f"[start]   GPU:     {props.name}  ({vram:.1f} GB VRAM)")
elif torch.backends.mps.is_available():
    print("[start]   MPS:     Apple Silicon  (unified memory)")
else:
    print(f"[start]   CPU:     {torch.get_num_threads()} threads")
PYEOF
echo "[start] -----------------------------------------------"

# ---------------------------------------------------------------------------
# 10. Start backend
# ---------------------------------------------------------------------------
echo "[start] Starting backend..."
"$RVC_PYTHON" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "$BACKEND_PID" > .pids/backend.pid
echo "[start] Backend PID $BACKEND_PID"

# ---------------------------------------------------------------------------
# 11. Wait for backend to be ready (up to 30s)
# ---------------------------------------------------------------------------
echo "[start] Waiting for backend to be ready..."
for i in $(seq 1 30); do
  if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "[start] Backend ready after ${i}s"
    break
  fi
  sleep 1
done

# ---------------------------------------------------------------------------
# 12. Start frontend
# ---------------------------------------------------------------------------
echo "[start] Starting frontend..."
if [ ! -d "frontend/node_modules" ]; then
  echo "[start] Installing frontend dependencies..."
  (cd frontend && pnpm install)
fi

_NEXT_HOSTNAME=$(echo "$NEXT_PUBLIC_API_URL" | sed 's|http://||' | sed 's|:8000||')
if [ "$_NEXT_HOSTNAME" = "localhost" ]; then
  (cd frontend && pnpm dev) &
else
  (cd frontend && pnpm dev --hostname "$_NEXT_HOSTNAME") &
fi
FRONTEND_PID=$!
echo "$FRONTEND_PID" > .pids/frontend.pid
echo "[start] Frontend PID $FRONTEND_PID"

# ---------------------------------------------------------------------------
# 13. Wait for frontend (up to 60s)
# ---------------------------------------------------------------------------
echo "[start] Waiting for frontend to compile..."
FRONTEND_URL=""
for i in $(seq 1 60); do
  if curl -s http://localhost:3000 >/dev/null 2>&1; then
    FRONTEND_URL="http://localhost:3000"
    break
  elif curl -s http://localhost:3001 >/dev/null 2>&1; then
    FRONTEND_URL="http://localhost:3001"
    break
  fi
  sleep 1
done

if [ -z "$FRONTEND_URL" ]; then
  echo "[start] Warning: frontend did not respond within 60s — opening http://localhost:3000 anyway"
  FRONTEND_URL="http://localhost:3000"
fi

echo "[start] Done — $FRONTEND_URL"
