#!/usr/bin/env bash
# Quick check: are MovieMind API + UI reachable on localhost?
# Run from moviemind/ after scripts/restart_moviemind.sh

set -euo pipefail

API_URL="${MOVIEMIND_API_URL:-http://127.0.0.1:8000}"
UI_URL="${MOVIEMIND_UI_URL:-http://127.0.0.1:8502}"

fail=0

check() {
  local url="$1"
  local label="$2"
  if curl -sf -o /dev/null --max-time 5 "$url"; then
    echo "OK  $label — $url"
  else
    echo "FAIL $label — $url (connection refused or timeout)"
    fail=1
  fi
}

echo "MovieMind local connectivity check"
echo "Listeners on 8000/8502:"
lsof -nP -iTCP:8000,8502 -sTCP:LISTEN 2>/dev/null || echo "  (none — run: bash scripts/restart_moviemind.sh)"
echo ""
check "${API_URL}/health" "API health"
check "${UI_URL}/" "Streamlit UI"
check "${UI_URL}/_stcore/health" "Streamlit core"

if [[ "$fail" -eq 0 ]]; then
  echo ""
  echo "Open in your browser: ${UI_URL}"
  exit 0
fi

echo ""
echo "Fix: cd moviemind && bash scripts/restart_moviemind.sh"
echo "If ports are busy: bash scripts/restart_moviemind.sh --stop-only && retry"
exit 1
