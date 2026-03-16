#!/usr/bin/env bash

cd "$(dirname "$0")/.."

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

# Belt-and-suspenders: kill by port in case PID files are stale or missing
stop_port 8000
stop_port 3000
stop_port 3001

# Kill any lingering next dev processes
NEXT_PIDS=$(pgrep -f "next dev" 2>/dev/null) || true
if [ -n "$NEXT_PIDS" ]; then
  echo "[stop] Killing lingering next dev process(es)..."
  echo "$NEXT_PIDS" | xargs kill -9 2>/dev/null || true
fi

echo "[stop] Done."
