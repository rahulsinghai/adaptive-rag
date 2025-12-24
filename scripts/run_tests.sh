#!/usr/bin/env bash
# scripts/run_tests.sh — run the full test suite
set -euo pipefail

exec uv run pytest "$@"
