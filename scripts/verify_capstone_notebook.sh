#!/usr/bin/env bash
# Execute MovieMind_capstone.ipynb headlessly (fast path: skip re-train if artifacts exist).
# Run from moviemind/ with venv active after download_review_artifacts.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NB="notebooks/MovieMind_capstone.ipynb"
OUT="${MOVIEMIND_CAPSTONE_EXEC_OUT:-notebooks/_capstone_executed.ipynb}"

if [[ ! -f "$NB" ]]; then
  echo "Missing $NB — are you on the capstone branch?" >&2
  exit 1
fi

if [[ ! -d "data/ml-1m" ]]; then
  echo "Missing data/ml-1m/. Run: bash scripts/download_review_artifacts.sh" >&2
  echo "  or: python3 src/data_loader.py" >&2
  exit 1
fi

export MOVIEMIND_SKIP_TUNE_DL="${MOVIEMIND_SKIP_TUNE_DL:-1}"
export MOVIEMIND_SKIP_TUNE_ML="${MOVIEMIND_SKIP_TUNE_ML:-1}"
export MOVIEMIND_SKIP_POST="${MOVIEMIND_SKIP_POST:-1}"

echo "[MovieMind] Executing $NB (fast path, skip flags above)"
echo "  Full re-train: MOVIEMIND_RUN_FULL=1 MOVIEMIND_SKIP_TUNE_DL=0 $0"

python -m jupyter nbconvert \
  --to notebook \
  --execute "$NB" \
  --output "$(basename "$OUT")" \
  --output-dir "$(dirname "$OUT")" \
  --ExecutePreprocessor.timeout="${MOVIEMIND_NB_TIMEOUT:-600}"

echo "[MovieMind] OK — wrote $OUT"
