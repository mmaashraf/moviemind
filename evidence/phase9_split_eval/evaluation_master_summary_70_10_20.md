# Evaluation Master Summary (70/10/20 Protocol)

This file consolidates evaluation outcomes across all implemented model families after moving to chronological `train/val/test`.

## Protocol snapshot

- Split: **Train 70% / Validation 10% / Test 20%** (chronological by timestamp)
- Train rows: `700,146`
- Val rows: `100,020`
- Test rows: `200,043`

## Table A — Raw model metrics (baseline + ML) on TEST

Source: `evidence/phase9_split_eval/ml_default_val_test_metrics_70_10_20.csv`

| Model | Family | RMSE (test) | MAE (test) | Fit time (sec) | Key hyperparameters |
|---|---|---:|---:|---:|---|
| Baseline (Global Mean) | baseline | 1.1048 | 0.9181 | 0.00 | global mean only |
| Linear Regression | ml | 0.9005 | 0.7088 | 0.06 | sklearn defaults |
| Random Forest (raw) | ml | 0.8998 | 0.7084 | 20.72 | `n_estimators=100`, `max_depth=10`, `random_state=42`, `n_jobs=-1` |
| Gradient Boosting (raw) | ml | **0.8981** | **0.7062** | 170.06 | `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`, `random_state=42` |

## Table B — Tuned ML metrics (validation-selected, tested on TEST)

Sources:
- `evidence/phase9_split_eval/ml_tuned_val_test_metrics_70_10_20.csv`
- `models/best_ml_params_70_10_20.json`
- `evidence/phase9_split_eval/tune_ml_run_70_10_20.txt`

| Model | Family | RMSE (val) | RMSE (test) | MAE (test) | Best hyperparameters | Tuning mode |
|---|---|---:|---:|---:|---|---|
| Random Forest (tuned) | ml_tuned | 0.9165 | 0.8998 | 0.7084 | `n_estimators=120`, `max_depth=10`, `min_samples_leaf=1`, `n_jobs=4`, `random_state=42` | fast-pass sampled train/val |
| Gradient Boosting (tuned) | ml_tuned | 0.9156 | 0.8993 | 0.7082 | `n_estimators=120`, `learning_rate=0.05`, `max_depth=4`, `random_state=42` | fast-pass sampled train/val |

## Table C — Raw DL and tuned DL metrics

Sources:
- Raw DL run: `evidence/phase9_split_eval/dl_model_run_70_10_20.txt`
- Tuned DL run: `evidence/phase9_split_eval/tune_dl_run_70_10_20.txt`
- Tuned params: `models/best_dl_params.txt`

| Model | Family | Optimization target | Best/Final metric | Hyperparameters |
|---|---|---|---|---|
| NCF (raw) | dl | val RMSE at training epochs | Epoch 10 val RMSE: `1.1997` | `embedding_dim=32`, hidden `128->64->32`, `dropout=0.2`, `lr=0.001` |
| NCF (tuned Optuna) | dl_tuned | val RMSE (50 trials) | Best trial val RMSE: `1.0887` | `embedding_dim=32`, hidden `256->192`, `dropout=0.1005`, `lr=0.000685`, `n_layers=2` |

## Table D — Summary by champion status (current run)

| Category | Winner | Evidence |
|---|---|---|
| Raw models (test RMSE) | Gradient Boosting (raw) | Table A |
| Tuned ML (test RMSE) | Random Forest tuned ~ Gradient Boosting tuned (both slightly behind raw GB) | Table B |
| DL path (validation objective) | Tuned NCF outperforms raw NCF on val RMSE | Table C |
| Overall recommendation for current reporting | Keep **raw Gradient Boosting** as primary champion | Tables A+B |

## Downstream effects — what should be updated next

### 1) Core docs (high priority)
- `README.md`
  - Update split protocol to 70/10/20.
  - Update latest metric table and champion note.
- `CONTEXT_HANDOVER.md`
  - Add this run block and note val is now used for tuning.
- `PROGRESS_TRACKER.md`
  - Add phase9 entry for split/eval/tuning.
- `WEBAPP_AGENT_WIKI.md`
  - Mention tuned-checkpoint loader fallback and train/val/test semantics.

### 2) Presentation artifacts (high priority if presenting this run)
- `evidence/presentation/results_slides_content.md`
  - Replace old metrics with this run’s table.
- `evidence/presentation/presentation_delivery_detailed_notes.md`
  - Update evaluation protocol section to 70/10/20 + tuned ML outcomes.
- PPT file(s)
  - Update Results 1/2 tables and one note: “ML fast-pass tuning did not beat raw GB”.

### 3) App / API surfaces (optional but recommended)
- `app/streamlit_app.py`
  - If showing model metrics in UI, refresh to new summary source.
- `src/api/model_registry.py`
  - If exposing tuned ML as selectable artifacts in API, add model defs and loading logic.

### 4) Evidence indexing
- Keep this file as the canonical phase9 summary.
- Link it from `evidence/README.md` and (optionally) from project root `README.md`.

## Notes / caveats

- `src/ml_models.py` standard run ended with a pickle compatibility save error in one run path; metrics were still captured via direct recompute and ML tuning outputs.
- ML tuning in this pass used sampled train/val for faster completion; test metrics are still evaluated on full test split.
