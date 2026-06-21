#!/usr/bin/env bash
# Record capstone notebook cell execution in JupyterLab (Playwright → WebM).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -d data/ml-1m ]]; then
  echo "Missing data/ml-1m/. Run: bash scripts/download_review_artifacts.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import playwright" 2>/dev/null; then
  pip install -r requirements-demo.txt
  playwright install chromium
fi

python scripts/record_notebook_video.py --manage-services "$@"
