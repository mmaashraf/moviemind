#!/usr/bin/env bash
# Record a brief MovieMind UI walkthrough (Playwright → WebM).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -d .venv ]]; then
  echo "Create venv first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import playwright" 2>/dev/null; then
  pip install playwright
  playwright install chromium
fi

python scripts/record_demo_video.py --manage-services --with-ollama "$@"
