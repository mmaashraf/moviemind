# MovieMind: Movie Recommendation System & AI Agent

MovieMind is an end-to-end capstone recommender system built on MovieLens 1M.
The project follows a strict AI Development Life Cycle (AIDLC) with reproducible
phase outputs, evidence capture, and milestone-based documentation updates.

## Current Project Status

- Phase 1 (Setup): completed
- Phase 2 (EDA): completed
- Phase 3 (Feature Engineering): completed
- Phase 4 (ML Modeling): completed (best model: Gradient Boosting, RMSE ~0.897)
- Phase 5 (DL Baseline): completed (NCF baseline underperforms GB)
- Phase 5b (Optuna tuning): completed (50 trials, best RMSE 1.0061)
- Phase 6 (Post-analysis): completed (safe-mode run; t-SNE skipped by default)
- Phase 7 (API): completed
- Phase 8 (UI + NLP baseline): completed

## Architecture and Key Features

- Data: MovieLens 1M with demographics (`age`, `gender`, `occupation`)
- ML models: Baseline, Linear Regression, Random Forest, Gradient Boosting
- DL model: Neural Collaborative Filtering (PyTorch embeddings + dense features)
- XAI direction: feature importance + embedding visualizations
- UI/Agent direction: Streamlit dashboard + natural language query layer

## Core Reproducibility Rules

1. Keep code modular (`src/`, `data/`, `models/`, `notebooks/`).
2. Log model runs to both console and file logs.
3. Record run durations for model comparisons.
4. Use time-based splits (no random leakage in recommenders).
5. Keep Train / Validation / Test logic explicit.
6. Capture evidence in `evidence/<phase>/` after each milestone.
7. Update all three docs every milestone:
   - `PROGRESS_TRACKER.md`
   - `CONTEXT_HANDOVER.md`
   - `AI_CONCEPTS_WIKI.md`

## Setup

### 0) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 1) Data Acquisition

```bash
python3 src/data_loader.py
```

Expected:
- `data/ml-1m/ratings.dat`
- `data/ml-1m/movies.dat`
- `data/ml-1m/users.dat`

### 2) EDA

Run notebooks:
- `notebooks/01_eda.ipynb`
- `notebooks/02_long_tail_and_cold_start.ipynb`

Expected:
- Insights on sparsity, long-tail, cold-start patterns

### 3) Feature Engineering

```bash
python3 src/features.py
```

Expected:
- `data/processed/train_features.csv`
- `data/processed/test_features.csv`

### 4) ML Modeling

```bash
python3 src/ml_models.py
```

Expected:
- `models/linear_regression.pkl`
- `models/random_forest.pkl`
- `models/gradient_boosting.pkl`
- `models/ml_training_log.txt`

### 5) DL Baseline

```bash
python3 src/dl_model.py
```

Expected:
- `models/ncf_model.pt`
- `models/dl_training_log.txt`
- `models/dl_loss_curve.png`

### 5b) DL Hyperparameter Tuning (Optuna)

```bash
python3 src/tune_dl.py
```

Expected:
- `models/best_dl_params.txt`
- `models/tune_dl_log.txt`
- `evidence/phase5b/tune_dl_observations.txt`

### 6) Post-Modeling Analysis and XAI

```bash
python3 src/post_analysis.py
```

Optional (enable t-SNE explicitly):
```bash
MOVIEMIND_ENABLE_TSNE=1 python3 src/post_analysis.py
```

Expected:
- `models/ncf_tuned_best.pt`
- `evidence/phase6/post_analysis_log.txt`
- `evidence/phase6/post_analysis_summary.txt`
- `evidence/phase6/user_embeddings_raw.csv`
- `evidence/phase6/user_embeddings_pca_2d.csv`
- `evidence/phase6/user_embeddings_pca_2d.png`
- `evidence/phase6/gradient_boosting_feature_importance.csv`
- `evidence/phase6/gradient_boosting_feature_importance.png`
- `evidence/phase6/tsne_status.txt` (present when t-SNE is skipped in safe mode)

Optional outputs (only when `MOVIEMIND_ENABLE_TSNE=1`):
- `evidence/phase6/user_embeddings_tsne_2d_sample.csv`
- `evidence/phase6/user_embeddings_tsne_2d_sample.png`

### 7) Backend API (FastAPI)

Start API:
```bash
source .venv/bin/activate
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Core endpoints:
- `GET /health`
- `GET /models`
- `GET /models/{model_id}/info`
- `POST /predict`
- `POST /recommend`
- `POST /nlp/query`

Expected phase artifacts:
- `src/api/app.py`
- `src/api/model_registry.py`
- `src/api/schemas.py`
- `src/api/nlp.py`
- `evidence/phase7/api_smoke_test_2026-04-24.txt`
- `evidence/phase7/phase7_milestone_log_2026-04-24.md`

### 8) Web UI + NLP Runtime Toggle (Streamlit)

Start UI:
```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py --server.port 8502
```

UI pages:
- Recommend
- Explain
- Model Inspector
- System

NLP runtime modes:
- `Rule-only`
- `Local LLM`
- `API LLM` (guarded fallback unless explicitly configured)

Expected phase artifacts:
- `app/streamlit_app.py`
- `WEBAPP_AGENT_WIKI.md`
- `evidence/phase8/streamlit_root_response_2026-04-24.html`
- `evidence/phase8/nlp_live_2026-04-24.json`
- `evidence/phase8/phase8_milestone_log_2026-04-24.md`

## Evidence Map

- `evidence/phase5b/`: tuning observations and result snapshots
- `evidence/phase6/`: post-analysis outputs and final summary
- `evidence/phase7/`: API smoke/live checks and milestone log
- `evidence/phase8/`: UI/NLP smoke checks and milestone log
- `evidence/code_reviews/`: periodic review notes and findings

## Milestone Update Protocol (Mandatory)

After every major run/phase completion:

1. Save outputs in `evidence/<phase>/`.
2. Update `PROGRESS_TRACKER.md` status and checkboxes.
3. Update `CONTEXT_HANDOVER.md` with latest state and next step.
4. Update `AI_CONCEPTS_WIKI.md` for new theory decisions.
5. Update this `README.md` with:
   - new runnable commands,
   - expected artifacts,
   - known caveats and best model snapshot.

This README is the primary replication guide and must stay in sync with each milestone.