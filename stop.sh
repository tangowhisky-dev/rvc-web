#!/usr/bin/env bash

cd "$(dirname "$0")"

# Step 1: Kill worker subprocesses FIRST (before killing backend)
echo "[stop] Killing worker subprocesses (spawn_main, realtime worker)..."
pkill -9 -f "spawn_main" 2>/dev/null || true
pkill -9 -f "_realtime_worker" 2>/dev/null || true
sleep 1

# Step 2: If backend is running, try clean shutdown via API
echo "[stop] Attempting clean shutdown via API..."
if nc -z localhost 8000 2>/dev/null; then
  timeout 3 curl -s http://localhost:8000/api/realtime/stop \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"session_id":"all"}' 2>/dev/null || true
  sleep 1
fi

# Step 3: Kill processes by PID file
stop_pid() {
  local name=$1
  local pidfile=".pids/${name}.pid"
  if [ -f "$pidfile" ]; then
    local pid
    pid=$(cat "$pidfile")
    echo "[stop] Stopping $name (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    rm -f "$pidfile"
  else
    echo "[stop] $name PID file not found — skipping"
  fi
}

# Step 4: Kill by port (forceful)
stop_port() {
  local port=$1
  local pids
  pids=$(lsof -ti:"$port" 2>/dev/null) || true
  if [ -n "$pids" ]; then
    echo "[stop] Killing process(es) on port $port..."
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
}

stop_pid backend
stop_pid frontend

# Belt-and-suspenders: kill by port in case PID files are stale
stop_port 8000
stop_port 3000
stop_port 3001

# Step 5: Kill any lingering dev/build/training processes
echo "[stop] Cleaning up dev processes..."
NEXT_PIDS=$(pgrep -f "next dev" 2>/dev/null) || true
if [ -n "$NEXT_PIDS" ]; then
  echo "$NEXT_PIDS" | xargs kill -9 2>/dev/null || true
fi

TRAIN_PIDS=$(pgrep -f "train\.py\|extract_f0_print\|extract_feature_print\|preprocess\.py" 2>/dev/null) || true
if [ -n "$TRAIN_PIDS" ]; then
  echo "$TRAIN_PIDS" | xargs kill -9 2>/dev/null || true
fi

echo "[stop] Done."
