#!/usr/bin/env bash
# Stop MovieMind API (FastAPI/uvicorn) and UI (Streamlit) on configured ports.
#
# Usage (from repo root or anywhere):
#   bash scripts/stop_moviemind.sh
#
# Env overrides (same as restart_moviemind.sh):
#   MOVIEMIND_API_PORT=8000
#   MOVIEMIND_UI_PORT=8502
#
# Equivalent to: bash scripts/restart_moviemind.sh --stop-only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/scripts/restart_moviemind.sh" --stop-only "$@"
