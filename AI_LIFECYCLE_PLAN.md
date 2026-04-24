# End-to-End AI Lifecycle Plan: Movie Recommendation System

This plan outlines the complete AI Development Life Cycle (AIDLC) for building the Movie Recommendation System. It mirrors the structure of a Software Development Life Cycle (SDLC) adapted for machine learning. We will use Git branches for each phase to maintain version control best practices. 

Before each phase, a brief overview of the concepts (ML/DL theory) will be provided. Code will have moderate comments using simple language to explain the core logic. All external code snippets will include source references.

## Architecture & Setup Decisions

*   **Environment:** Standard Python 3 virtual environment (`venv`) with `requirements.txt`.
*   **Deployment:** We will implement everything locally first. This allows for rapid iteration and debugging. It is very easy to containerize (Docker) the application later as a final step.
*   **Data Acquisition:** A dedicated Python script will download the MovieLens 1M dataset automatically from grouplens.org.
*   **UI Design:** We will start with Streamlit for rapid development, but we will inject custom CSS and interactive elements (sliders, toggles, metric bars) to create a futuristic, "Iron Man" style dashboard with explainability features.

## Proposed Phases

### Phase 1: Project Initialization & Data Acquisition
**Branch:** `phase-1-setup`

*   **Objective:** Set up the repository structure, define the environment, and acquire the raw data.
*   **Tasks:**
    *   Create the standardized directory structure (`data/`, `notebooks/`, `src/`, `models/`, `app/`).
    *   Create `.gitignore` to prevent committing raw data and secrets.
    *   Initialize `requirements.txt`.
    *   Write a script (`src/data_loader.py`) to download and load the MovieLens dataset.

### Phase 2: Exploratory Data Analysis (EDA)
**Branch:** `phase-2-eda`

*   **Objective:** Understand the data distribution, identify anomalies, and uncover patterns that will inform feature engineering.
*   **Tasks:**
    *   Create Jupyter notebooks for interactive analysis.
    *   Analyze user rating distributions, movie popularity, and genre frequencies.
    *   Document findings regarding sparsity and data quality.
    *   Establish baseline understanding of user behavior.

### Phase 3: Data Preparation & Feature Engineering
**Branch:** `phase-3-features`

*   **Objective:** Transform raw data into structured features suitable for modeling.
*   **Tasks:**
    *   Implement `src/features.py`.
    *   Create user-item matrices.
    *   Extract user features (average rating, count) and movie features (genres, year).
    *   Implement a strict time-based Train/Test split to avoid data leakage.

### Phase 4: Machine Learning (ML) Modeling
**Branch:** `phase-4-ml-modeling`

*   **Objective:** Build traditional machine learning models to establish strong baselines.
*   **Tasks:**
    *   Implement `src/ml_models.py`.
    *   Train Baseline (mean prediction), Linear Regression, and a Tree-based model (Random Forest / LightGBM).
    *   Implement evaluation metrics in `src/evaluation.py` (RMSE, MAE, Precision@K).
    *   Log performance and save model artifacts.

### Phase 5: Deep Learning (DL) Modeling
**Branch:** `phase-5-dl-modeling`

*   **Objective:** Capture complex non-linear interactions using neural networks.
*   **Tasks:**
    *   Implement `src/dl_model.py` using PyTorch.
    *   Build an architecture utilizing User and Movie embeddings concatenated with dense features.
    *   Train the model using batching and monitor loss.
    *   Compare DL performance against ML baselines.

### Phase 6: Post-Modeling Analysis
**Branch:** `phase-6-analysis`

*   **Objective:** Interpret the models and extract actionable insights.
*   **Tasks:**
    *   Implement `src/analysis.py`.
    *   Extract embeddings from the trained PyTorch model.
    *   Apply K-Means clustering on embeddings to identify user segments and movie groups.
    *   Visualize clusters using PCA or t-SNE.

### Phase 7: Backend API Development
**Branch:** `phase-7-api`

*   **Objective:** Serve the trained models via a robust API.
*   **Tasks:**
    *   Implement `app/api.py` using FastAPI.
    *   Create read-only endpoints: `/recommend` (returns top-K movies) and `/predict` (returns specific rating prediction).
    *   Implement input validation and error handling.

### Phase 8: Web UI Deployment
**Branch:** `phase-8-ui`

*   **Objective:** Provide a futuristic, interactive interface to interact with the recommendation system.
*   **Tasks:**
    *   Implement `app/ui.py` using Streamlit.
    *   Apply custom CSS to give it a dark, "Iron Man" aesthetic (neon accents, glassmorphism).
    *   Add interactive toggles, metric bars, and sliders to explore the recommendations.
    *   Include explainability elements (e.g., "Why was this recommended?").

## Verification Plan

### Automated Tests
*   Write basic unit tests for data loading, feature generation, and model prediction logic using `pytest`.

### Manual Verification
*   Execute the entire pipeline from end to end.
*   Test the FastAPI endpoints using Swagger UI (built-in).
*   Interact with the Streamlit UI to ensure data flows correctly from the frontend to the model and back.
