#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[start] Killing any stale process on port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

mkdir -p .pids

export RVC_ROOT=Retrieval-based-Voice-Conversion-WebUI

echo "[start] Starting backend (conda rvc env)..."
conda run --no-capture-output -n rvc \
  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "$BACKEND_PID" > .pids/backend.pid
echo "[start] Backend PID $BACKEND_PID"

sleep 2

echo "[start] Starting frontend..."
(cd frontend && pnpm dev) &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > .pids/frontend.pid
echo "[start] Frontend PID $FRONTEND_PID"

echo "[start] Waiting for frontend to compile (8s)..."
sleep 8

echo "[start] Opening browser..."
open http://localhost:3000
echo "[start] Done — http://localhost:3000"
