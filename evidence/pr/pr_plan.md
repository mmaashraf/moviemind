# Pull Request Plan

Repository: `mmaashraf/moviemind`

## PR 1
- **Base:** `main`
- **Head:** `phase-1-setup`
- **Title:** `feat(phase-1): initialize project structure and dataset loader`
- **Body:**
  - set up project scaffold and baseline dependencies
  - add data loader script for MovieLens dataset
  - add basic hygiene files for clean repo setup

## PR 2
- **Base:** `phase-1-setup`
- **Head:** `phase-2-eda`
- **Title:** `feat(phase-2): add exploratory data analysis notebooks and findings`
- **Body:**
  - add EDA notebooks for distributions, sparsity, and long-tail analysis
  - document dataset behavior and cold-start implications
  - establish analysis baseline for feature engineering

## PR 3
- **Base:** `phase-2-eda`
- **Head:** `phase-3-features`
- **Title:** `feat(phase-3): implement feature engineering with time-based split`
- **Body:**
  - implement feature generation pipeline for user/movie signals
  - enforce chronological train/test split to prevent leakage
  - generate processed datasets for downstream model training

## PR 4
- **Base:** `phase-3-features`
- **Head:** `phase-4-ml-modeling`
- **Title:** `feat(phase-4-5): add ML baselines, DL model, and tuning foundation`
- **Body:**
  - add ML training pipeline and model evaluation logging
  - add DL NCF training pipeline
  - add hyperparameter tuning foundation and concept wiki updates

## PR 5
- **Base:** `phase-4-ml-modeling`
- **Head:** `phase-6-analysis`
- **Title:** `feat(phase-6): post-analysis pipeline, XAI artifacts, and UI mock kit`
- **Body:**
  - add crash-safe phase-6 post-analysis pipeline (embeddings, PCA/t-SNE, feature importance)
  - add structured evidence artifacts and periodic code-review notes
  - sync README/tracker/handover/wiki and include UI mock assets
