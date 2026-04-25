# PPT Content - Results 1 and Results 2

Use this directly for the two result slides.
All numbers below are grounded in current project evidence files.

---

## Results 1 - Model Evaluation Comparison

### Slide title suggestion
**Results 1: Offline Evaluation Across Model Families**

### Main table (paste into PPT)

| Model | Family | RMSE | MAE | Train Time (sec) | Notes |
|---|---|---:|---:|---:|---|
| Baseline (Global Mean) | Baseline | 1.1043 | 0.9195 | 0.00 | Reference floor |
| Linear Regression | ML | 0.9002 | 0.7085 | 0.07 | Strong fast baseline |
| Random Forest | ML | 0.8988 | 0.7076 | 20.23 | Better than linear |
| Gradient Boosting | ML | **0.8971** | **0.7054** | 193.17 | **Champion model** |
| NCF Baseline | DL | 1.1146 | 0.9307 | - | Underperformed ML champion |
| NCF Tuned (Optuna + retrained weights) | DL Tuned | 0.9754 | 0.7936 | Trial-based | Improved vs untuned DL, still behind GB |

### Key takeaway bullets
- Best overall offline performer is **Gradient Boosting** (lowest RMSE/MAE).
- DL tuning improved NCF, but did not beat the ML champion on final comparison.
- Time/accuracy trade-off visible: GB is best accuracy, but slower than linear/RF.

### Evidence sources
- `models/ml_training_log.txt`
- `evidence/phase4/ml_backfill_run_2026-04-23.md`
- `models/best_dl_params.txt`
- `evidence/presentation/dl_metrics_recomputed_2026-04-25.txt`
- `CONTEXT_HANDOVER.md`

---

## Results 2 - Tuned DL, XAI, and Deployment Readiness

### Slide title suggestion
**Results 2: Tuned DL Details + Explainability + Serving Readiness**

### Table A - Tuned DL hyperparameters (best trial)

| Hyperparameter | Value |
|---|---:|
| learning_rate | 0.0054401120 |
| embedding_dim | 32 |
| dropout_rate | 0.4431288799 |
| n_layers | 3 |
| n_units_l0 | 256 |
| n_units_l1 | 64 |
| n_units_l2 | 32 |
| best_rmse (trial objective) | 1.0061 |

### Table B - Post-model analysis and production signals

| Item | Value | Why it matters |
|---|---|---|
| Retrain RMSE (best tuned config, 5 epochs) | 0.9754 | Confirms tuned config behavior in analysis rerun |
| t-SNE status | completed | Embedding space visual artifacts available |
| Top GB features | movie_avg_rating, user_avg_rating, user_rating_count | Clear interpretability signal |
| Served models via API | baseline, linear, RF, GB, NCF, tuned NCF | Production-style multi-model registry |

### Table C - Diversity Slider (UI + Recommendation Logic)

| Control / Output | Definition | Impact |
|---|---|---|
| `diversity_alpha` slider | User control for novelty-vs-relevance tradeoff | Higher alpha increases diversity pressure |
| Reranking formula | `adjusted_score = predicted_rating_raw - alpha * overlap_penalty` | Penalizes repetitive genre overlap |
| Debug outputs per item | `predicted_rating_raw`, `overlap_penalty`, `adjusted_score`, `overlap_genres` | Makes ranking transparent and explainable |
| Taste Map linkage | Compares user historical genre mix vs recommendation mix | Helps user see exploration effect visually |

### Key takeaway bullets
- Tuning produced a stronger DL configuration (vs untuned NCF), but not enough to replace GB as champion.
- Explainability is available via feature-importance and embedding visual analysis.
- Deployment layer is complete: API + UI can switch across model families.
- Diversity slider adds a controllable exploration mechanism instead of fixed relevance-only ranking.

### Evidence sources
- `models/best_dl_params.txt`
- `evidence/phase6/post_analysis_summary.txt`
- `evidence/phase6/gradient_boosting_feature_importance.csv`
- `evidence/phase7/models_live_2026-04-24.json`

---

## Presenter one-liners for defense

- "Model selection was based on leakage-safe offline metrics, not just complexity."
- "Gradient Boosting wins on error; tuned DL adds representation and future extensibility."
- "We intentionally deployed all families behind one registry for robustness and demos."
