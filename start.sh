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
# 2. Detect available compute device
#    Priority: CUDA > MPS > CPU
#    Exports RVC_DEVICE so Python subprocesses can read it.
#    Use conda env's Python — system Python may not have torch or may be
#    CPU-only.
# ---------------------------------------------------------------------------
CONDA_ENV="${RVC_CONDA_ENV:-rvc}"
if conda run -n "$CONDA_ENV" python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  export RVC_DEVICE="cuda"
elif conda run -n "$CONDA_ENV" python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
  export RVC_DEVICE="mps"
else
  export RVC_DEVICE="cpu"
fi
echo "[start] Compute device: $RVC_DEVICE"

# ---------------------------------------------------------------------------
# 3. Universal threading env vars (all platforms)
#    Prevents faiss + fairseq OpenMP collision.
# ---------------------------------------------------------------------------
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---------------------------------------------------------------------------
# 4. Platform-specific PyTorch tuning
# ---------------------------------------------------------------------------
if [ "$RVC_DEVICE" = "mps" ]; then
  # MPS: disable memory cap so unified memory is fully usable;
  # prefer shared memory for cheaper host-device transfers
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  export PYTORCH_MPS_PREFER_SHARED_MEMORY=1
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  echo "[start] MPS tuning applied"
elif [ "$RVC_DEVICE" = "cuda" ]; then
  # CUDA: set visible devices if not already set by caller
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "[start] CUDA_VISIBLE_DEVICES defaulted to 0 (set externally to override)"
  else
    echo "[start] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from environment)"
  fi
  export PYTORCH_ENABLE_MPS_FALLBACK=0
fi

# ---------------------------------------------------------------------------
# 5. Kill stale processes — cross-platform
# ---------------------------------------------------------------------------
kill_port() {
  local port=$1
  # Try lsof first (macOS, most Linux)
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
  # Try fuser (Linux)
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
  # Fallback: Python-based port kill (works on Windows Git Bash)
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
                subprocess.run(["taskkill", "/F", "/PID", pid],
                               capture_output=True)
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
# 6. PROJECT_ROOT — single source of truth for all path resolution
# ---------------------------------------------------------------------------
export PROJECT_ROOT="$(pwd)"
echo "[start] PROJECT_ROOT=$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 7. Detect all IP addresses and let user choose
# ---------------------------------------------------------------------------
echo "[start] Detecting network interfaces..."

# Collect IPs into a newline-separated string (avoids bash array issues with set -u)
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

# Count entries (trim whitespace from wc output)
IP_COUNT=$(printf '%s\n' "$IP_LIST" | wc -l | tr -d ' ')

if [ "$IP_COUNT" -le 1 ]; then
  export NEXT_PUBLIC_API_URL="http://localhost:8000"
  echo "[start] No network interfaces found — using localhost only"
else
  echo ""
  echo "  Available network interfaces for frontend access:"
  echo ""

  # Display options
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

  # Default to first non-localhost IP
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
    echo "[start] Selected: $SELECTED_IP → NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL"
  else
    echo "[start] Invalid selection — falling back to localhost"
    export NEXT_PUBLIC_API_URL="http://localhost:8000"
  fi
  echo ""
fi

# ---------------------------------------------------------------------------
# 8. Print device summary banner
# ---------------------------------------------------------------------------
echo "[start] -----------------------------------------------"
echo "[start] Device summary"
echo "[start]   OS:      $PLATFORM"
echo "[start]   Device:  $RVC_DEVICE"
python - <<'PYEOF' 2>/dev/null || true
import torch, sys
d = ""
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1024**3
    d = f"  GPU:     {props.name}  ({vram:.1f} GB VRAM)"
elif torch.backends.mps.is_available():
    d = "  MPS:     Apple Silicon  (unified memory)"
else:
    d = f"  CPU:     {torch.get_num_threads()} threads"
print(f"[start] {d}")
PYEOF
echo "[start] -----------------------------------------------"

# ---------------------------------------------------------------------------
# 9. Start backend
#    Respects RVC_CONDA_ENV (default: rvc) or falls back to plain python
#    if conda is not in use.
# ---------------------------------------------------------------------------
RVC_CONDA_ENV="${RVC_CONDA_ENV:-rvc}"

echo "[start] Starting backend..."
if command -v conda >/dev/null 2>&1; then
  KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n "$RVC_CONDA_ENV" \
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
else
  # Plain venv / system python — assumes dependencies are already installed
  python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
fi
BACKEND_PID=$!
echo "$BACKEND_PID" > .pids/backend.pid
echo "[start] Backend PID $BACKEND_PID"

# ---------------------------------------------------------------------------
# 10. Wait for backend to be ready (up to 30s)
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
# 11. Start frontend
# ---------------------------------------------------------------------------
echo "[start] Starting frontend..."
if [ ! -d "frontend/node_modules" ]; then
  echo "[start] Installing frontend dependencies..."
  (cd frontend && pnpm install)
fi
(cd frontend && pnpm dev) &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > .pids/frontend.pid
echo "[start] Frontend PID $FRONTEND_PID"

# ---------------------------------------------------------------------------
# 12. Wait for frontend (up to 60s)
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
