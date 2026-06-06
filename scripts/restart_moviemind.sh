#!/usr/bin/env bash
# MovieMind: stop API + Streamlit on known ports, then start fresh (background).
# Local demo only: binds 127.0.0.1 by default — no TLS, API auth, or rate limits.
#
# Usage (from anywhere):
#   bash scripts/restart_moviemind.sh
#   bash scripts/restart_moviemind.sh --stop-only
#   bash scripts/stop_moviemind.sh              # same as --stop-only
#   bash scripts/restart_moviemind.sh --start-only
#   bash scripts/restart_moviemind.sh --with-ollama    # also ensure Ollama is up
#   bash scripts/restart_moviemind.sh --foreground   # API+UI in foreground (Ctrl+C stops both)
#
# Env overrides:
#   MOVIEMIND_API_PORT=8000
#   MOVIEMIND_UI_PORT=8502
#   MOVIEMIND_API_HOST=127.0.0.1
#   MOVIEMIND_UVICORN_RELOAD=1   # default on; set 0 to disable --reload

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

API_PORT="${MOVIEMIND_API_PORT:-8000}"
UI_PORT="${MOVIEMIND_UI_PORT:-8502}"
API_HOST="${MOVIEMIND_API_HOST:-127.0.0.1}"
API_URL="http://${API_HOST}:${API_PORT}"
UI_URL="http://127.0.0.1:${UI_PORT}"
LOG_DIR="${ROOT}/evidence/runtime"
API_PID_FILE="${LOG_DIR}/moviemind_api.pid"
UI_PID_FILE="${LOG_DIR}/moviemind_ui.pid"
RELOAD="${MOVIEMIND_UVICORN_RELOAD:-0}"

DO_STOP=1
DO_START=1
DO_OLLAMA=0
FOREGROUND=0

for arg in "$@"; do
  case "${arg}" in
    --stop-only) DO_START=0 ;;
    --start-only) DO_STOP=0 ;;
    --with-ollama) DO_OLLAMA=1 ;;
    --foreground) FOREGROUND=1 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "[MovieMind] Unknown arg: ${arg} (try --help)"
      exit 1
      ;;
  esac
done

log() { echo "[MovieMind] $*"; }

kill_port() {
  local port="$1"
  local label="$2"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  fi
  if [[ -z "${pids}" ]]; then
    log "No listener on port ${port} (${label})"
    return 0
  fi
  log "Stopping ${label} on port ${port} (pid(s): ${pids//$'\n'/ })"
  # shellcheck disable=SC2086
  kill ${pids} 2>/dev/null || true
  sleep 1
  pids="$(lsof -ti tcp:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    log "Force kill still listening on ${port}"
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
  fi
}

kill_pid_file() {
  local file="$1"
  local label="$2"
  if [[ -f "${file}" ]]; then
    local pid
    pid="$(cat "${file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      log "Stopping ${label} from pid file (${pid})"
      kill "${pid}" 2>/dev/null || true
      sleep 1
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${file}"
  fi
}

stop_services() {
  log "Stopping MovieMind API + UI..."
  kill_pid_file "${API_PID_FILE}" "API"
  kill_pid_file "${UI_PID_FILE}" "UI"
  kill_port "${API_PORT}" "API (uvicorn)"
  kill_port "${UI_PORT}" "UI (streamlit)"
}

activate_venv() {
  if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/.venv/bin/activate"
    log "Using venv: ${ROOT}/.venv"
  else
    log "No .venv found — using current python on PATH"
  fi
}

wait_for_url() {
  local url="$1"
  local label="$2"
  local tries="${3:-30}"
  local i
  for ((i = 1; i <= tries; i++)); do
    if curl -sf "${url}" >/dev/null 2>&1; then
      log "${label} ready: ${url}"
      return 0
    fi
    sleep 1
  done
  log "WARNING: ${label} not ready after ${tries}s: ${url}"
  return 1
}

start_api() {
  log "Starting API on ${API_URL} ..."
  if [[ "${RELOAD}" == "1" ]]; then
    nohup uvicorn src.api.app:app --host "${API_HOST}" --port "${API_PORT}" --reload \
      >"${LOG_DIR}/api.log" 2>&1 &
  else
    nohup uvicorn src.api.app:app --host "${API_HOST}" --port "${API_PORT}" \
      >"${LOG_DIR}/api.log" 2>&1 &
  fi
  disown 2>/dev/null || true
  echo $! >"${API_PID_FILE}"
  log "API pid $(cat "${API_PID_FILE}") — log: ${LOG_DIR}/api.log"
}

start_ui() {
  log "Starting Streamlit on ${UI_URL} ..."
  nohup streamlit run app/streamlit_app.py \
    --server.port "${UI_PORT}" \
    --server.address 127.0.0.1 \
    --server.headless true \
    >"${LOG_DIR}/ui.log" 2>&1 &
  disown 2>/dev/null || true
  echo $! >"${UI_PID_FILE}"
  log "UI pid $(cat "${UI_PID_FILE}") — log: ${LOG_DIR}/ui.log"
}

start_foreground() {
  log "Foreground mode: API in background, Streamlit in foreground (Ctrl+C stops UI only)"
  start_api
  wait_for_url "${API_URL}/health" "API" 40 || true
  export MOVIEMIND_API_URL="${API_URL}"
  exec streamlit run app/streamlit_app.py --server.port "${UI_PORT}" --server.address 127.0.0.1
}

ensure_ollama() {
  log "Checking Ollama (optional --with-ollama)..."
  bash "${ROOT}/scripts/setup_local_ollama.sh"
}

start_services() {
  mkdir -p "${LOG_DIR}"
  activate_venv
  if [[ "${DO_OLLAMA}" -eq 1 ]]; then
    ensure_ollama
  fi
  if [[ "${FOREGROUND}" -eq 1 ]]; then
    start_foreground
    return
  fi
  start_api
  start_ui
  # First API boot loads large processed CSVs into memory; allow extra time on cold start.
  wait_for_url "${API_URL}/health" "API" 90 || true
  wait_for_url "${UI_URL}" "UI" 60 || true
  log "Done."
  log "  API: ${API_URL}/docs"
  log "  UI:  ${UI_URL}"
  log "  Logs: ${LOG_DIR}/api.log  ${LOG_DIR}/ui.log"
  log "  Stop: bash scripts/restart_moviemind.sh --stop-only"
}

[[ "${DO_STOP}" -eq 1 ]] && stop_services
[[ "${DO_START}" -eq 1 ]] && start_services

if [[ "${DO_STOP}" -eq 1 && "${DO_START}" -eq 0 ]]; then
  log "Stop-only complete."
fi
