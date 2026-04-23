#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
#   bash evidence/pr/create_prs.sh                # all PRs
#   bash evidence/pr/create_prs.sh --phase 6      # only Phase 6 PR
#   bash evidence/pr/create_prs.sh --phase all    # same as default

REPO_SLUG="mmaashraf/moviemind"

has_gh() {
  command -v gh >/dev/null 2>&1
}

has_open() {
  command -v open >/dev/null 2>&1
}

usage() {
  echo "Usage: bash evidence/pr/create_prs.sh [--phase <1|2|3|4|6|all>]"
}

create_or_print_pr() {
  local base="$1"
  local head="$2"
  local title="$3"
  local body="$4"
  local compare_url="https://github.com/${REPO_SLUG}/compare/${base}...${head}?expand=1"

  echo "------------------------------------------------------------"
  echo "Base:  ${base}"
  echo "Head:  ${head}"
  echo "Title: ${title}"

  if has_gh; then
    # Check if PR already exists for this head/base pair
    if gh pr list --base "${base}" --head "${head}" --json number --jq 'length' | grep -q '^1$'; then
      echo "PR already exists for ${head} -> ${base}. Skipping."
      return
    fi

    gh pr create \
      --base "${base}" \
      --head "${head}" \
      --title "${title}" \
      --body "${body}" || {
        echo "gh PR create failed for ${head} -> ${base}. Open manually:"
        echo "${compare_url}"
      }
  else
    echo "gh CLI not found. Open/create PR manually:"
    echo "${compare_url}"
    if has_open; then
      open "${compare_url}" || true
    fi
  fi
}

BODY_1=$'## Summary\n- set up project scaffold and dependencies\n- add MovieLens data loading pipeline\n- add initial repo hygiene for clean development\n\n## Test plan\n- run data loader and verify dataset files are created\n- verify repository structure matches setup plan'
BODY_2=$'## Summary\n- add EDA notebooks for rating/user/movie distributions\n- analyze sparsity and long-tail effects\n- document cold-start insights for design decisions\n\n## Test plan\n- execute notebooks end-to-end\n- verify key charts and findings are reproducible'
BODY_3=$'## Summary\n- implement feature engineering pipeline\n- add strict time-based split to avoid leakage\n- generate processed train/test artifacts\n\n## Test plan\n- run feature pipeline script\n- verify processed CSV outputs and schema'
BODY_4=$'## Summary\n- add ML model training/evaluation pipeline\n- add DL NCF training workflow\n- add tuning foundation and concept documentation\n\n## Test plan\n- run ML training script and verify logged metrics\n- run DL training script and verify saved artifacts'
BODY_5=$'## Summary\n- add Phase 6 post-analysis with crash-safe t-SNE behavior\n- generate embeddings/PCA/t-SNE/XAI evidence artifacts\n- sync core docs and include UI mock assets\n\n## Test plan\n- run `python src/post_analysis.py` and verify phase6 artifacts\n- run `MOVIEMIND_ENABLE_TSNE=1 python src/post_analysis.py` and verify t-SNE outputs\n- review docs/evidence consistency'

run_phase() {
  local phase="$1"
  case "${phase}" in
    1)
      create_or_print_pr "main" "phase-1-setup" \
        "feat(phase-1): initialize project structure and dataset loader" \
        "${BODY_1}"
      ;;
    2)
      create_or_print_pr "phase-1-setup" "phase-2-eda" \
        "feat(phase-2): add exploratory data analysis notebooks and findings" \
        "${BODY_2}"
      ;;
    3)
      create_or_print_pr "phase-2-eda" "phase-3-features" \
        "feat(phase-3): implement feature engineering with time-based split" \
        "${BODY_3}"
      ;;
    4)
      create_or_print_pr "phase-3-features" "phase-4-ml-modeling" \
        "feat(phase-4-5): add ML baselines, DL model, and tuning foundation" \
        "${BODY_4}"
      ;;
    6)
      create_or_print_pr "phase-4-ml-modeling" "phase-6-analysis" \
        "feat(phase-6): post-analysis pipeline, XAI artifacts, and UI mock kit" \
        "${BODY_5}"
      ;;
    all)
      run_phase 1
      run_phase 2
      run_phase 3
      run_phase 4
      run_phase 6
      ;;
    *)
      echo "Invalid phase: ${phase}"
      usage
      exit 1
      ;;
  esac
}

PHASE="all"
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--phase" && $# -eq 2 ]]; then
    PHASE="$2"
  else
    usage
    exit 1
  fi
fi

run_phase "${PHASE}"

echo "------------------------------------------------------------"
echo "Done."
