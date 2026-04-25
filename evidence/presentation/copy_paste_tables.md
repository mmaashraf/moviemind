# Copy-Paste Tables for PPT (Grounded from Evidence)

These tables are generated from current repository evidence files and are safe to paste into PowerPoint tables.

---

## Table A - Phase 6 Key Outcomes

Source: `evidence/phase6/post_analysis_summary.txt`

| Metric | Value | Note |
|---|---:|---|
| Best tuned RMSE (from tuning file) | 1.0061 | Read from `best_dl_params.txt` summary |
| RMSE from retraining best tuned config (5 epochs) | 0.9754 | Validation rerun value |
| t-SNE status | completed | Phase 6 run status |
| Top 3 GB features | movie_avg_rating, user_avg_rating, user_rating_count | From feature importance output |

---

## Table B - Model Availability Snapshot

Source: `evidence/phase7/models_live_2026-04-24.json`

| Model ID | Display Name | Family | Available |
|---|---|---|---|
| baseline_global_mean | Baseline (Global Mean) | baseline | Yes |
| linear_regression | Linear Regression | ml | Yes |
| random_forest | Random Forest | ml | Yes |
| gradient_boosting | Gradient Boosting | ml | Yes |
| ncf_baseline | NCF Baseline | dl | Yes |
| ncf_tuned | NCF Tuned Best | dl_tuned | Yes |

---

## Table C - Gradient Boosting Feature Importance

Source: `evidence/phase6/gradient_boosting_feature_importance.csv`

| Feature | Importance |
|---|---:|
| movie_avg_rating | 0.6406477435 |
| user_avg_rating | 0.3184542817 |
| user_rating_count | 0.0203292313 |
| age | 0.0066379608 |
| release_year | 0.0058062992 |
| occupation | 0.0047557152 |
| movie_rating_count | 0.0033687684 |

---

## Slide Mapping (where to paste)

- **Slide 5 (Results 1):** Table A + Table C
- **Slide 6 (Results 2):** Table B (optional small inset) or API response screenshot

