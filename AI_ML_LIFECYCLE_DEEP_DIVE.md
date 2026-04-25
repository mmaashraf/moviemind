# MovieMind: AI/ML Lifecycle Deep Dive

This document is a detailed reflection of the AI/ML development lifecycle followed in MovieMind, including why each phase was done, what decisions were made, what outputs were produced, and what was learned.

---

## 0) Problem Framing and Success Criteria

### Problem
Build a movie recommendation system that is not only accurate, but also explainable and production-ready for API/UI integration.

### Core Success Criteria
1. End-to-end pipeline from raw data to model artifacts.
2. Strong baseline and comparative modeling (ML vs DL).
3. Leakage-safe evaluation design.
4. Explainability outputs (feature importance and embedding analysis).
5. Reproducible phase-wise documentation and evidence.

### Why this framing matters
Many projects stop at "a model runs." This lifecycle emphasizes reliability, explainability, and handover quality, which are critical in real ML engineering and capstone evaluation.

---

## 1) Phase 1 - Project Initialization and Data Acquisition

### Objective
Create a reproducible foundation: repo structure, dependencies, and deterministic data ingestion.

### What was done
- Standardized folders (`src/`, `data/`, `models/`, `notebooks/`, etc.).
- Dependency setup (`requirements.txt`).
- Data loading/downloading pipeline (`src/data_loader.py`).

### Why this was necessary
- Without disciplined structure, later phases become hard to debug.
- Reproducibility starts with deterministic data access and environment setup.

### Output value
- A reliable project skeleton that supports iterative experiments without chaos.

---

## 2) Phase 2 - Exploratory Data Analysis (EDA)

### Objective
Understand the statistical reality of the recommendation problem before modeling.

### What was discovered
- Very high user-item sparsity.
- Long-tail movie popularity.
- Cold-start risk for new/rare users/items.
- Rating distribution skew.

### Why this mattered
EDA directly informed architecture choices:
- simple memory-based methods would struggle under sparsity;
- need robust feature engineering and model families that handle sparse interactions;
- need explainability and careful split strategy.

### Output value
- Data-driven design decisions instead of blind model selection.

---

## 3) Phase 3 - Feature Engineering and Data Splitting

### Objective
Transform raw interactions into model-ready features while preserving temporal realism.

### Key engineering decisions
- User features (activity/mean behavior signals).
- Movie features (popularity/average signals).
- Metadata features (e.g., demographics and release year).
- Chronological (time-based) train/test split.

### Why time-based split was critical
Recommendation systems are time-sensitive. Random splits can leak future behavior into training and inflate metrics.

### Output value
- Processed train/test artifacts suitable for fair evaluation and downstream modeling.

---

## 4) Phase 4 - Traditional ML Modeling

### Objective
Establish strong tabular baselines and compare model families under the same feature space.

### Models trained
- Baseline global mean.
- Linear Regression.
- Random Forest.
- Gradient Boosting.

### Result
- Gradient Boosting emerged as best performer (champion model in this lifecycle stage).

### Why this phase mattered
- Baselines quantify whether complex modeling is justified.
- A strong ML champion is a high bar that DL must beat to be practically valuable.

### Output value
- Reproducible benchmark suite and model artifacts for serving/comparison.

---

## 5) Phase 5 - Deep Learning Baseline (NCF)

### Objective
Learn richer latent user-item representations with neural collaborative filtering.

### What was built
- Embedding-based DL recommender with dense layers.
- Training on MPS-capable hardware where available.

### Result
- Useful representation learning, but primary metric did not surpass the ML champion in current runs.

### Why this was still valuable
- DL introduces latent-space understanding and enables embedding-based analysis.
- Even if not champion on RMSE, DL artifacts power explainability and future personalization extensions.

---

## 5b) Hyperparameter Tuning (Optuna)

### Objective
Systematically improve DL model performance via automated search.

### What was done
- Optuna-based search over learning rate, embedding size, network depth/width, dropout, etc.
- Multi-trial optimization and best-parameter tracking.

### Result
- Tuned DL improved vs untuned runs but still did not displace Gradient Boosting as overall champion.

### Why this phase mattered
- Confirms conclusions are robust: DL was given a fair optimization attempt.
- Prevents biased comparisons where one family is heavily tuned and the other is not.

### Output value
- Best-parameter artifacts and evidence logs supporting model selection decisions.

---

## 6) Phase 6 - Post-Modeling Analysis and Explainability (XAI)

### Objective
Convert model internals into interpretable outputs for trust, debugging, and presentation.

### What was produced
- Retrained tuned DL checkpoint for analysis.
- Raw user embedding exports.
- PCA embedding projection and visualization.
- Optional t-SNE projection path.
- Gradient Boosting feature importance table/plot.
- Phase analysis logs and summary files.

### Important engineering refinement
- t-SNE stability issue was mitigated with safer execution flow so pipeline completion is not blocked by environment-specific failures.

### Why this phase mattered
- Moves project from "black-box predictor" to "explainable system."
- Supports API/UI features like "why recommended?" and visual behavior inspection.

### Output value
- Evidence-grade artifacts for reporting, debugging, and stakeholder communication.

---

## 7) Documentation and Evidence Discipline (Cross-Phase)

A major strength of this lifecycle is that technical work was paired with explicit operational discipline:

- `PROGRESS_TRACKER.md`: phase status truth board.
- `CONTEXT_HANDOVER.md`: continuity brain for next sessions.
- `README.md`: reproducible execution guide.
- `AI_CONCEPTS_WIKI.md`: active learning/theory record.
- `evidence/`: run outputs, audits, reviews, and artifacts.

### Why this matters in ML lifecycle terms
ML systems are iterative and stateful. Without disciplined records, results become non-reproducible and decision rationale is lost.

---

## Model Governance Outcome So Far

### Champion model (current)
- Gradient Boosting (best observed error profile in this lifecycle).

### DL status
- Valuable for representation learning and explainability assets.
- Not currently superior on primary metric in the completed tuning cycle.

### Decision quality
- Evidence-backed, not assumption-backed.
- Comparative runs, logs, and artifacts support the conclusion.

---

## What Comes Next (Lifecycle Continuation)

### Phase 7 - API Layer
- Serve predictions/recommendations through FastAPI.
- Support runtime model selection across all available models.
- Add input validation and robust error handling.

### Phase 8 - Web App and Agent Experience
- Streamlit (or equivalent) interactive app.
- Model selector in UI.
- Explainability panel using Phase 6 artifacts.
- NLP/Agent query layer for natural-language recommendations.

---

## Meta-Learning From This Lifecycle

1. Data and split strategy often matter more than model novelty.
2. A tuned "fancy model" does not automatically beat strong classical baselines.
3. Explainability is not an afterthought; it is a lifecycle phase.
4. Reproducibility/documentation is part of ML engineering, not admin overhead.
5. Phase-gated workflow with evidence collection greatly improves project quality and confidence.

