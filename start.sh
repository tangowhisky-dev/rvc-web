#!/usr/bin/env bash
set -euo pipefail

# Anchor to rvc-web/ regardless of where the script is called from
cd "$(dirname "$0")"

# Kill any process on the given port
kill_port() {
  local port=$1
  local pids
  pids=$(lsof -ti:"$port" 2>/dev/null) || true
  if [ -n "$pids" ]; then
    echo "[start] Killing stale process(es) on port $port..."
    echo "$pids" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
}

# Kill any running next dev process (handles .next/dev/lock conflicts)
kill_next_dev() {
  local pids
  pids=$(pgrep -f "next dev" 2>/dev/null) || true
  if [ -n "$pids" ]; then
    echo "[start] Killing stale next dev process(es)..."
    echo "$pids" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
}

kill_port 8000
kill_port 3000
kill_next_dev

# Kill any dangling training or inference processes
kill_dangling() {
  local pids
  pids=$(pgrep -f "train\.py\|spawn_main\|_realtime_worker\|extract_f0_print\|extract_feature_print\|preprocess\.py" 2>/dev/null) || true
  if [ -n "$pids" ]; then
    echo "[start] Killing dangling training/inference processes..."
    echo "$pids" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
}
kill_dangling

mkdir -p .pids
mkdir -p logs
mkdir -p data/profiles

# PROJECT_ROOT is rvc-web/ — the single source of truth for all path resolution.
# Replaces the old RVC_ROOT which pointed at the Retrieval-based-Voice-Conversion-WebUI submodule.
export PROJECT_ROOT="$(pwd)"

# Fix libomp duplicate-init crash when PyTorch and system OpenMP both load
export KMP_DUPLICATE_LIB_OK=TRUE
# Restrict OMP/MKL/BLAS to single thread — prevents faiss+fairseq OpenMP collision
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[start] PROJECT_ROOT=$PROJECT_ROOT"
echo "[start] Starting backend (conda rvc env)..."
KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n rvc \
  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "$BACKEND_PID" > .pids/backend.pid
echo "[start] Backend PID $BACKEND_PID"

# Wait for backend to be ready (up to 30s)
echo "[start] Waiting for backend to be ready..."
for i in $(seq 1 30); do
  if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "[start] Backend ready after ${i}s"
    break
  fi
  sleep 1
done

echo "[start] Starting frontend..."
(cd frontend && pnpm dev) &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > .pids/frontend.pid
echo "[start] Frontend PID $FRONTEND_PID"

# Wait for frontend to be ready (up to 60s)
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
