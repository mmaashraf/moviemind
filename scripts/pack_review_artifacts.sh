#!/usr/bin/env bash
# Pack gitignored data/ + models/ for a GitHub Release (reviewer fast path).
# Run from moviemind/ after a successful Path A or B build.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

for f in data/processed/train_features.csv models/gradient_boosting.pkl; do
  if [[ ! -f "$f" ]]; then
    echo "Missing $f — run build_model_artifacts.py features + ml first."
    exit 1
  fi
done

OUT="${1:-moviemind-artifacts.tar.gz}"
tar -czf "$OUT" data models
echo "Created $OUT ($(du -h "$OUT" | cut -f1)). Upload to a GitHub Release and set MOVIEMIND_ARTIFACTS_URL."
