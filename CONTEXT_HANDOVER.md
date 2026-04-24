# Context Handover: Movie Recommendation System Capstone

**Last Updated:** Phase 4 (ML Models Code Written)

## Project Overview
Capstone Project for an AI/ML/DL course. Building an end-to-end Movie Recommendation System following the AI Development Life Cycle (AIDLC). Full plan is in `AI_LIFECYCLE_PLAN.md`. Progress is tracked in `PROGRESS_TRACKER.md`.

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
*   **Phase 3 (Features):** COMPLETED. `src/features.py` executed. Output: `data/processed/train_features.csv` (800K rows, 15 cols) and `test_features.csv`. Time-based split to prevent data leakage.
*   **Phase 4 (ML Models):** CODE WRITTEN. `src/ml_models.py` and `src/evaluation.py`. Awaiting execution: `python3 src/ml_models.py`.
*   **Next Step:** Phase 5 (Deep Learning with PyTorch).

## Key EDA Insights That Drive Design
*   95% sparsity means simple KNN fails. Must use embeddings/matrix factorization.
*   30%+ movies have fewer than 20 ratings (cold start). Justifies hybrid content+collaborative approach.
*   Ratings skew positive. Predicting low scores will be harder for models.
*   Demographics (age, gender, occupation) available for all 6,040 users. Strong feature for personalization.

## Rules & Constraints (Set by User)
*   **Code Quality:** No messy scratchpad code. Clean, professional one-liners. Do code review after each phase.
*   **Communication:** Stop and ask before Git commits. Stop at phase boundaries for recap/confirmation.
*   **References:** Credit all external code sources.
*   **Comments:** Moderate, simple language.
*   **File Operations:** Restricted to `/Users/ashraf/iitd-aiml/final_project/` directory only.
*   **Git/Python Execution:** Must be done by the user from their terminal (macOS sandbox restriction).
