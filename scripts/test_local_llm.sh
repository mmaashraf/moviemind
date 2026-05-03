#!/usr/bin/env bash
set -euo pipefail

# MovieMind local LLM smoke test.
# - Assumes API is already running on MOVIEMIND_API_URL (default: http://127.0.0.1:8000)
# - Saves output to evidence/phase8
#
# Usage:
#   bash scripts/test_local_llm.sh
#   MOVIEMIND_API_URL=http://127.0.0.1:8000 bash scripts/test_local_llm.sh

API_URL="${MOVIEMIND_API_URL:-http://127.0.0.1:8000}"
OUT_DIR="evidence/phase8"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_FILE="${OUT_DIR}/local_llm_smoke_${STAMP}.txt"

mkdir -p "${OUT_DIR}"

run_check() {
  local label="$1"
  local method="$2"
  local url="$3"
  local payload="${4:-}"
  local headers_file body_file curl_time api_latency
  headers_file="$(mktemp)"
  body_file="$(mktemp)"

  if [[ "${method}" == "GET" ]]; then
    curl_time="$(curl -sS -D "${headers_file}" -o "${body_file}" -w "%{time_total}" "${url}")"
  else
    curl_time="$(curl -sS -D "${headers_file}" -o "${body_file}" -w "%{time_total}" -X "${method}" "${url}" \
      -H "Content-Type: application/json" -d "${payload}")"
  fi

  api_latency="$(awk -F': ' 'tolower($1)=="x-latency-ms"{gsub("\r","",$2); print $2}' "${headers_file}")"
  echo "${label}"
  cat "${body_file}"
  echo
  printf "timing: curl_total_sec=%s, api_x_latency_ms=%s\n\n" "${curl_time}" "${api_latency:-n/a}"
  rm -f "${headers_file}" "${body_file}"
}

{
  echo "=== MovieMind Local LLM Smoke Test ==="
  echo "timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "api_url: ${API_URL}"
  echo

  run_check "1) Health check" "GET" "${API_URL}/health"
  run_check "2) Local LLM parse (should be local-llm-ollama if Ollama is up, else fallback)" "POST" "${API_URL}/nlp/query" \
    '{"query":"top 5 action movies for user 10 with tuned model","runtime_mode":"local-llm"}'
  run_check "3) Rule-only parse (control check)" "POST" "${API_URL}/nlp/query" \
    '{"query":"top 5 action movies for user 10 with tuned model","runtime_mode":"rule-only"}'
} | tee "${OUT_FILE}"

echo "[MovieMind] Smoke test output saved to ${OUT_FILE}"
