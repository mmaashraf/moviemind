# Project Progress Tracker

## Phase 1: Project Initialization & Data Acquisition
- [x] Set up the repository structure (`data/`, `notebooks/`, `src/`, `models/`, `app/`)
- [x] Create `.gitignore` and `requirements.txt`
- [x] Initialize git branch `phase-1-setup`
- [x] Write `src/data_loader.py` to download and extract MovieLens dataset
- **Status:** **COMPLETED**

## Phase 2: Exploratory Data Analysis (EDA)
- [x] Initialize git branch `phase-2-eda`
- [x] Analyze rating distributions, users, movies, and genres
- [x] Document sparsity and data quality insights
- **Status:** **COMPLETED**

## Phase 3: Data Preparation & Feature Engineering
- [x] Implement user/movie feature extraction
- [x] Implement time-based train/test split
- [x] Run feature pipeline to generate processed datasets
- **Status:** **COMPLETED** (`data/processed/train_features.csv` and `data/processed/test_features.csv` generated)

## Phase 4: Machine Learning (ML) Modeling
- [x] Implement baseline and ML models (`src/ml_models.py`)
- [x] Add evaluation logic (`src/evaluation.py`)
- **Status:** **COMPLETED** (Gradient Boosting achieved lowest RMSE: 0.897)

## Phase 5: Deep Learning (DL) Modeling
- [x] Build and train PyTorch model with embeddings (`src/dl_model.py`)
- [x] Compare against ML baselines
- **Status:** **COMPLETED** (Gradient Boosting ML model still holds the lead, PyTorch NCF overfit without heavy tuning)

## Phase 5b: DL Hyperparameter Tuning (Optuna)
- [x] Create `src/tune_dl.py` with Optuna search space
- [x] Increase trials from pilot run to `N_TRIALS = 50`
- [x] Save best trial params to `models/best_dl_params.txt` (final run best RMSE: 1.0061)
- [x] Run and complete the full 50-trial tuning job
- [x] Record final 5b result and compare against Gradient Boosting (RMSE 0.897)
- **Status:** **COMPLETED** (50 trials done; tuned DL did not beat Gradient Boosting)

## Phase 6: Post-Modeling Analysis
- [x] Implement `src/post_analysis.py` for embedding extraction + PCA/t-SNE + feature importance
- [x] Run post-analysis script and generate evidence artifacts
- [x] Review outputs and write final Phase 6 observations
- **Status:** **COMPLETED** (safe-mode run complete; t-SNE skipped by default to avoid local crash)

## Phase 7: Backend API Development
- [ ] Build FastAPI endpoints `/recommend` and `/predict`
- **Status:** *Pending*

## Phase 8: Web UI & NLP Agent Deployment
- [ ] Build Streamlit "Iron Man" dashboard
- [ ] Integrate NLP Agent for natural language queries
- **Status:** *Pending*
