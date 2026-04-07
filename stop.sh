#!/usr/bin/env bash
# stop.sh — cross-platform shutdown for rvc-web

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# 1. Kill worker subprocesses FIRST (before killing backend)
# ---------------------------------------------------------------------------
echo "[stop] Killing worker subprocesses (spawn_main, realtime worker)..."
if command -v pkill >/dev/null 2>&1; then
  pkill -9 -f "spawn_main" 2>/dev/null || true
  pkill -9 -f "_realtime_worker" 2>/dev/null || true
fi
sleep 1

# ---------------------------------------------------------------------------
# 2. If backend is running, try clean shutdown via API
#    Use curl with Python fallback if curl is not available
# ---------------------------------------------------------------------------
echo "[stop] Attempting clean shutdown via API..."

check_port() {
  local port=$1
  if command -v nc >/dev/null 2>&1; then
    nc -z localhost "$port" 2>/dev/null
  elif command -v bash >/dev/null 2>&1; then
    # Bash built-in /dev/tcp
    (echo >/dev/tcp/localhost/"$port") 2>/dev/null
  else
    # Python fallback
    python -c "import socket; s=socket.socket(); s.settimeout(1); result=s.connect_ex(('localhost', $port)); s.close(); exit(0 if result==0 else 1)" 2>/dev/null
  fi
}

http_post() {
  local url=$1
  local data=$2
  if command -v curl >/dev/null 2>&1; then
    curl -s "$url" -X POST -H 'Content-Type: application/json' -d "$data" 2>/dev/null || true
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c "
import urllib.request
import sys
try:
    req = urllib.request.Request('$url', data=b'$data', headers={'Content-Type': 'application/json'}, method='POST')
    urllib.request.urlopen(req, timeout=3)
except:
    pass
" 2>/dev/null || true
  elif command -v python >/dev/null 2>&1; then
    python -c "
import urllib.request
import sys
try:
    req = urllib.request.Request('$url', data=b'$data', headers={'Content-Type': 'application/json'}, method='POST')
    urllib.request.urlopen(req, timeout=3)
except:
    pass
" 2>/dev/null || true
  fi
}

if check_port 8000; then
  http_post "http://localhost:8000/api/realtime/stop" '{"session_id":"all"}'
  sleep 1
fi

# ---------------------------------------------------------------------------
# 3. Kill processes by PID file
# ---------------------------------------------------------------------------
stop_pid() {
  local name=$1
  local pidfile=".pids/${name}.pid"
  if [ -f "$pidfile" ]; then
    local pid
    pid=$(cat "$pidfile")
    if [ -n "$pid" ]; then
      echo "[stop] Stopping $name (PID $pid)..."
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  else
    echo "[stop] $name PID file not found — skipping"
  fi
}

# ---------------------------------------------------------------------------
# 4. Kill by port (cross-platform)
# ---------------------------------------------------------------------------
stop_port() {
  local port=$1
  local killed=false
  
  # Try lsof first (macOS, most Linux)
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null) || true
    if [ -n "$pids" ]; then
      echo "[stop] Killing process(es) on port $port (via lsof)..."
      echo "$pids" | xargs kill -9 2>/dev/null || true
      killed=true
    fi
  fi
  
  # Try fuser (Linux)
  if [ "$killed" = false ] && command -v fuser >/dev/null 2>&1; then
    local pids
    pids=$(fuser "$port/tcp" 2>/dev/null) || true
    if [ -n "$pids" ]; then
      echo "[stop] Killing process(es) on port $port (via fuser)..."
      echo "$pids" | xargs kill -9 2>/dev/null || true
      killed=true
    fi
  fi
  
  # Python fallback for Windows Git Bash
  if [ "$killed" = false ]; then
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
  fi
}

stop_pid backend
stop_pid frontend

# Belt-and-suspenders: kill by port in case PID files are stale
stop_port 8000
stop_port 3000
stop_port 3001

# ---------------------------------------------------------------------------
# 5. Kill any lingering dev/build/training processes
# ---------------------------------------------------------------------------
echo "[stop] Cleaning up dev processes..."
if command -v pgrep >/dev/null 2>&1; then
  NEXT_PIDS=$(pgrep -f "next dev" 2>/dev/null) || true
  if [ -n "$NEXT_PIDS" ]; then
    echo "$NEXT_PIDS" | xargs kill -9 2>/dev/null || true
  fi
  
  TRAIN_PIDS=$(pgrep -f "train\.py\|extract_f0_print\|extract_feature_print\|preprocess\.py" 2>/dev/null) || true
  if [ -n "$TRAIN_PIDS" ]; then
    echo "$TRAIN_PIDS" | xargs kill -9 2>/dev/null || true
  fi
fi

echo "[stop] Done."

# ---------------------------------------------------------------------------
# 6. Sweep temp/ on shutdown — clear all temp audio files
# ---------------------------------------------------------------------------
if [ -d "temp" ]; then
  echo "[stop] Sweeping temp/..."
  find temp/ -maxdepth 1 -type f -delete 2>/dev/null || true
fi
