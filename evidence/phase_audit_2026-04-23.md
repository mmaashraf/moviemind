# Evidence Audit - 2026-04-23

This audit checks which outputs already exist from previous phases.
It uses file presence and available logs as evidence.

## Phase 1 (Setup)
- Evidence found:
  - `requirements.txt`
  - `src/data_loader.py`
  - project folder structure present (`data/`, `src/`, `models/`, `notebooks/`)
- Status from evidence: likely completed

## Phase 2 (EDA)
- Evidence found:
  - `notebooks/01_eda.ipynb`
  - `notebooks/02_long_tail_and_cold_start.ipynb`
- Status from evidence: notebooks present; execution details not logged in evidence folder yet

## Phase 3 (Feature Engineering)
- Evidence found:
  - `data/processed/train_features.csv`
  - `data/processed/test_features.csv`
- Status from evidence: completed

## Phase 4 (ML Modeling)
- Evidence found:
  - `models/linear_regression.pkl`
  - `models/random_forest.pkl`
  - `models/gradient_boosting.pkl`
  - `models/ml_training_log.txt`
  - `evidence/phase4/ml_backfill_run_2026-04-23.md`
- Status from evidence: completed with artifacts and training log

## Phase 5 (DL Baseline)
- Evidence found:
  - `models/dl_training_log.txt` (contains epoch-level loss and RMSE outputs)
  - `models/ncf_model.pt`
  - `models/dl_loss_curve.png`
- Missing in current snapshot:
  - none observed
- Status from evidence: completed with key artifacts present

## Phase 5b (DL Hyperparameter Tuning)
- Evidence found:
  - `models/best_dl_params.txt` (best RMSE recorded: 1.0061)
  - `models/tune_dl_log.txt`
  - `evidence/phase5b/final_tuning_summary_2026-04-23.md`
- Status from evidence: completed (50 trials)

## Phase 6 (Post-Modeling Analysis / XAI)
- Evidence found:
  - `models/ncf_tuned_best.pt`
  - `evidence/phase6/post_analysis_log.txt`
  - `evidence/phase6/post_analysis_summary.txt`
  - `evidence/phase6/user_embeddings_raw.csv`
  - `evidence/phase6/user_embeddings_pca_2d.csv`
  - `evidence/phase6/user_embeddings_pca_2d.png`
  - `evidence/phase6/gradient_boosting_feature_importance.csv`
  - `evidence/phase6/tsne_status.txt`
- Status from evidence: completed in safe mode (t-SNE skipped by default to avoid local crash)

## Action Needed
1. Begin Phase 7 backend API implementation with model-selection support.
2. Keep this audit updated after each new phase milestone.
