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
- [x] Build FastAPI endpoints `/health`, `/models`, `/models/{model_id}/info`, `/predict`, `/recommend`
- [x] Implement model registry adapters for Baseline, ML, DL, Tuned DL
- [x] Add NLP parser endpoint `/nlp/query` with runtime-mode contract and guardrails
- [x] Run API smoke tests and live endpoint checks; save evidence under `evidence/phase7/`
- **Status:** **COMPLETED**

## Phase 8: Web UI & NLP Agent Deployment
- [x] Build Streamlit web app with pages: Recommend, Model Inspector, Embedding Space, Model Visualizers, Lifecycle Evidence, AI Concepts, System
- [x] Add runtime toggle in UI (`Rule-only`, `Local LLM`, `API LLM`)
- [x] Wire UI flows to FastAPI endpoints and validate startup behavior
- [x] Capture UI/NLP evidence under `evidence/phase8/`
- [x] Create dedicated wiki `WEBAPP_AGENT_WIKI.md`
- [x] Validate local Ollama path end-to-end (`parsed_by=local-llm-ollama`) and capture latency evidence
- **Status:** **COMPLETED**

## Phase 8x: Tool Agent Hardening, Streaming, and UX Reliability
- [x] Add multi-step tool agent endpoint `POST /agent/query` with Ollama `/api/chat` tool calls
- [x] Add streaming endpoint `POST /agent/query/stream` (SSE) for incremental assistant/tool updates
- [x] Add `TOOL_AGENT_WIKI.md` with architecture, contracts, troubleshooting, and multi-turn prompt recipes
- [x] Improve Streamlit Agent UX with collapsible trace panel, explicit run status (`running/done/failed`), and SSE fallback behavior
- [x] Add genre alias helpers (`genre_any`/`genre_filter`) including `fiction` expansion and OR-filter over-fetch logic
- [x] Add resilience guardrails: read-timeout retries (`MOVIEMIND_OLLAMA_READ_RETRIES`) and pseudo-tool JSON nudge retries (`MOVIEMIND_AGENT_PSEUDO_TOOL_RETRIES`)
- [x] Add agent-side invalid user-id handling and API-level `404` for missing user resource (`GET /users/{user_id}/summary`)
- [x] Expose diversity guidance in agent prompt and tool schema (`diversity_alpha` alignment with manual flow)
- **Status:** **COMPLETED**
