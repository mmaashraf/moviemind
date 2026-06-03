# Context Handover: Movie Recommendation System Capstone

**Last Updated:** Phase 8 Finalized (API + UI + Local LLM validated)

## Project Overview
Capstone Project for an AI/ML/DL course. Building an end-to-end Movie Recommendation System following the AI Development Life Cycle (AIDLC). Full plan is in `learning/AI_LIFECYCLE_PLAN.md`. Progress is tracked in `PROGRESS_TRACKER.md` (this folder).

## Dataset
*   **Name:** MovieLens 1M (released 2003, movies up to year 2000)
*   **Path:** `data/ml-1m/`
*   **Why 1M?** It is the only MovieLens dataset that includes user demographics (`users.dat` with age, gender, occupation). Newer datasets removed demographics for privacy. This enables richer feature engineering and complex Agent queries like "What should a 25yr old programmer watch?"
*   **Files:** `ratings.dat` (1M rows), `movies.dat` (3,883 rows), `users.dat` (6,040 rows)

## Tech Stack
*   **Environment:** Python 3 (Anaconda), macOS M1 16GB RAM. PyTorch `mps` backend available.
*   **ML:** Scikit-Learn (baselines, ensembles)
*   **DL:** PyTorch (embeddings, Neural Collaborative Filtering)
*   **Backend:** FastAPI
*   **Frontend:** Streamlit (customized with "Iron Man" aesthetic: dark mode, neon accents, sliders, toggles)
*   **Agent:** LLM-powered natural language query layer on top of FastAPI

## Advanced Capstone Features
1.  **Explainable AI (XAI):** API returns reasoning for each recommendation (feature importance or nearest neighbor).
2.  **Vector Space Visualization:** PCA/t-SNE plots of User and Movie embeddings.
3.  **Serendipity / Diversity Engine:** Toggle in UI to recommend movies *outside* the user's usual genre bubble.

## Current Status
*   **Phase 1 (Setup):** COMPLETED. Repo structure, `.gitignore`, `requirements.txt`, `src/data_loader.py`.
*   **Phase 2 (EDA):** COMPLETED. `01_eda.ipynb` and `02_long_tail_and_cold_start.ipynb`. Key findings: 95% matrix sparsity, long tail proves cold start problem, ratings skewed positive (mean ~3.58), user base skewed male 20-30s.
*   **Phase 3 (Features):** COMPLETED. `src/features.py` executed. Output files exist in `data/processed/` (`train_features.csv`, `test_features.csv`). Time-based split used to prevent leakage.
*   **Phase 4 (ML Models):** COMPLETED. `src/ml_models.py` and `src/evaluation.py` implemented and run.
    *   **Results:** Baseline RMSE 1.104; best model remains Gradient Boosting with RMSE 0.897 and MAE 0.705.
*   **Phase 5 (Deep Learning):** COMPLETED. `src/dl_model.py` executed on M1 `mps`.
    *   **Results:** Base NCF run reached RMSE ~1.097 (below Gradient Boosting performance).
*   **Phase 5b (DL Hyperparameter Tuning):** COMPLETED. 50-trial Optuna run finished.
*   **Final 50-trial result:** `models/best_dl_params.txt` now shows RMSE 1.0061 (embedding dim 32).
*   **Comparison:** Tuned DL still does not beat Gradient Boosting (RMSE 0.897), so GB remains best model.
*   **Phase 6 (Post-Modeling Analysis & XAI):** COMPLETED with safe-mode execution.
    *   Retrained best tuned DL config: validation RMSE 1.0302 (5-epoch retrain run).
    *   Saved embeddings + PCA artifacts and Gradient Boosting feature-importance outputs in `evidence/phase6/`.
    *   t-SNE is now disabled by default due local segmentation fault risk; enable manually with `MOVIEMIND_ENABLE_TSNE=1`.
*   **Phase 7 (Backend API):** COMPLETED.
    *   Added FastAPI service in `src/api/` with `/health`, `/models`, `/models/{model_id}/info`, `/predict`, `/recommend`, `/nlp/query`.
    *   Added unified model registry to serve Baseline, ML, DL, Tuned DL from one interface.
    *   Saved smoke and live API evidence under `evidence/phase7/`.
*   **Phase 8 (Web UI + NLP Layer):** COMPLETED and validated with local model.
    *   Streamlit app in `app/streamlit_app.py` includes: Recommend, Model Inspector, Embedding Space, Model Visualizers, Lifecycle Evidence, AI Concepts, System.
    *   Runtime modes: `local-llm`, `api-llm` with schema guardrails; multi-step tool agent on `main` (Phase 8x).
    *   Local LLM setup and smoke scripts added:
        * `scripts/setup_local_ollama.sh`
        * `scripts/test_local_llm.sh`
    *   Local LLM evidence captured with latency and parser identity:
        * `evidence/phase8/local_llm_smoke_2026-04-30_23-31-48.txt`
        * confirms `parsed_by=local-llm-ollama`.
    *   Documentation: `docs/APP_AND_API.md`, `docs/OLLAMA.md`, `docs/AGENT.md`, `REVIEWER_SETUP.md`, `REPLICATION.md`
*   **Phase 8x:** Tool agent + SSE merged to `main` (PR #11).

## Key EDA Insights That Drive Design
*   95% sparsity means simple KNN fails. Must use embeddings/matrix factorization.
*   30%+ movies have fewer than 20 ratings (cold start). Justifies hybrid content+collaborative approach.
*   Ratings skew positive. Predicting low scores will be harder for models.
*   Demographics (age, gender, occupation) available for all 6,040 users. Strong feature for personalization.

## Rules & Constraints (Set by User)
1.  **Modular Architecture:** Keep strict separation across `src/`, `data/`, `models/`, and `notebooks/`.
2.  **Dual-Stream Logging:** Training scripts must log to both console and a dedicated `.txt` file in `models/`.
3.  **Execution Timers:** Record training duration for ML and DL runs.
4.  **No Placeholders:** No dummy MVP shortcuts; capstone-grade implementations only.
5.  **Git Hygiene:** Use feature branches and descriptive commits; keep tree clean between phases.
6.  **Big Three Documentation:** Keep `docs/internal/PROGRESS_TRACKER.md`, `docs/internal/CONTEXT_HANDOVER.md`, and `docs/internal/learning/AI_CONCEPTS_WIKI.md` synchronized.
7.  **Time-Based Splitting:** Use chronological splits to avoid leakage.
8.  **Rigorous Validation:** Maintain Train/Validation/Test discipline, especially during tuning.
9.  **Hardware Acceleration:** Use M1 `mps` (or best available accelerator) for DL.
10. **XAI First:** Accuracy is not enough; include explainability and embedding visualizations.
11. **Evidence Capture:** Every phase must save observations, outputs, and proof artifacts under `evidence/<phase>/`.

## Practical Workflow Constraints
*   **Communication:** Ask before git commits; stop at phase boundaries for recap/confirmation.
*   **References:** Credit all external code sources.
*   **Comments:** Moderate, simple-language comments.
*   **File Operations:** Restricted to `/Users/ashraf/iitd-aiml/final_project/`.
*   **Git/Python Execution:** User terminal remains source of truth for final run confirmation; agent maintains reproducible scripts and evidence logs.
