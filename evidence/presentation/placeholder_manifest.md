# MovieMind PPT Placeholder Manifest (Grounded)

Use this with `evidence/presentation/MovieMind_Final_Presentation_manual_placeholders.pptx`.

## Where placeholders are in the deck

- **Slide 4 - Methodology, Datasets, and Tools used**
  - Placeholder text in slide body:
    - `[PLACEHOLDER] Insert pipeline diagram image here (data -> train -> serve).`
  - Recommended source:
    - Create or export a flow diagram from `AI_LIFECYCLE_PLAN.md`.

- **Slide 5 - Results 1**
  - Placeholder texts in slide body:
    - `[PLACEHOLDER TABLE] Paste final model comparison table here (RMSE/MAE/etc).`
    - `[PLACEHOLDER IMAGE] Insert feature-importance plot here.`
  - Manual insert zone textbox at bottom:
    - `MANUAL INSERT ZONE: You can replace this box with your final screenshot/table.`
  - Grounded sources available now:
    - Table content: `evidence/presentation/copy_paste_tables.md`
    - Image: `evidence/presentation/assets/feature_importance_top7.png`

- **Slide 6 - Results 2**
  - Placeholder texts in slide body:
    - `[PLACEHOLDER IMAGE] Insert Streamlit Recommend page screenshot here.`
    - `[PLACEHOLDER IMAGE] Insert Model Inspector screenshot here.`
    - `[PLACEHOLDER IMAGE] Insert PCA/t-SNE embedding visualization here.`
    - `[PLACEHOLDER IMAGE] Insert API response (recommend endpoint) screenshot/table here.`
    - `[PLACEHOLDER TABLE] Insert Diversity Slider logic table (alpha, formula, debug outputs) here.`
  - Manual insert zone textbox at bottom:
    - `MANUAL INSERT ZONE: You can replace this box with your final screenshot/table.`
  - Grounded sources available now:
    - PCA image: `evidence/presentation/assets/pca_scatter.png`
    - t-SNE image: `evidence/presentation/assets/tsne_scatter.png`
    - API response data: `evidence/phase7/recommend_live_2026-04-24.json`
    - Diversity table content: `evidence/presentation/results_slides_content.md` (Table C)

## Quick insertion plan (recommended)

1. Keep `MovieMind_Final_Presentation_manual_placeholders.pptx` as your editing deck.
2. On slide 5, paste the table from `copy_paste_tables.md`, then insert `feature_importance_top7.png`.
3. On slide 6, insert Streamlit screenshots (manual), then insert `pca_scatter.png` and `tsne_scatter.png`.
4. Remove placeholder lines after each insert.

## Evidence grounding used

- `evidence/phase6/post_analysis_summary.txt`
- `evidence/phase6/gradient_boosting_feature_importance.csv`
- `evidence/phase7/models_live_2026-04-24.json`
