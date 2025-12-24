#!/usr/bin/env bash
# scripts/start_frontend.sh — start the Streamlit UI
set -euo pipefail

: "${API_BASE_URL:=http://localhost:8000}"
: "${PORT:=8501}"

export API_BASE_URL

exec streamlit run frontend/streamlit_app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0
