#!/usr/bin/env bash
# Download pre-built data/ + models/ for a fast "clone and run" reviewer path.
#
# Usage (from moviemind/):
#   export MOVIEMIND_ARTIFACTS_URL="https://github.com/.../releases/download/.../moviemind-artifacts.tar.gz"
#   bash scripts/download_review_artifacts.sh
#
# Or pass the URL as the first argument.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

URL="${1:-${MOVIEMIND_ARTIFACTS_URL:-}}"

need() {
  local path="$1"
  [[ -e "$path" ]] || return 0
  return 1
}

if [[ -f "data/processed/train_features.csv" && -f "models/gradient_boosting.pkl" ]]; then
  echo "Artifacts already present (data/processed + models/gradient_boosting.pkl). Skipping download."
  exit 0
fi

if [[ -z "$URL" ]]; then
  cat <<'EOF'
No artifact URL set.

For ~2 minute clone-and-run, publish a tarball of data/ and models/ (see scripts/pack_review_artifacts.sh),
upload it to a GitHub Release (or similar), then:

  export MOVIEMIND_ARTIFACTS_URL="https://.../moviemind-artifacts.tar.gz"
  bash scripts/download_review_artifacts.sh

Otherwise build locally (15–45 min): REVIEWER_SETUP.md §4.2–4.3 Path A.
EOF
  exit 1
fi

TMP="$(mktemp -t moviemind-artifacts.XXXXXX.tar.gz)"
trap 'rm -f "$TMP"' EXIT

echo "Downloading artifacts from: $URL"
curl -fsSL "$URL" -o "$TMP"
tar -xzf "$TMP" -C "$ROOT"

if [[ ! -f "data/processed/train_features.csv" || ! -f "models/gradient_boosting.pkl" ]]; then
  echo "Download finished but expected files are missing."
  echo "Archive should extract with top-level paths: data/ and models/ (relative to moviemind/)."
  exit 1
fi

echo "Artifacts ready under data/ and models/."
