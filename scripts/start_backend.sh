#!/usr/bin/env bash
# scripts/start_backend.sh — start the FastAPI server with Uvicorn
set -euo pipefail

: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${WORKERS:=1}"
: "${LOG_LEVEL:=info}"

exec uvicorn backend.api.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --log-level "$LOG_LEVEL" \
  --reload
