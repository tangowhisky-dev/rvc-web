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

stop_pid backend
stop_pid frontend

echo "[stop] Done."
