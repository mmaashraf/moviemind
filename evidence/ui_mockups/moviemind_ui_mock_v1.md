# MovieMind UI Mockup v1 (Wireframe Spec)

This is a high-detail wireframe/spec for the final web app vision.
Goal: premium "Iron Man" style interface with all models selectable at inference time.

## 1) Page Layout (Desktop)

```
+--------------------------------------------------------------------------------------------------+
| MovieMind  | AI Movie Recommendation Studio                                [Data: ML-1M] [MPS] |
+------------------------------+---------------------------------------------+---------------------+
| Recommendation Controls      | Top Recommendations                         | Why This Rec?       |
|-----------------------------|---------------------------------------------|---------------------|
| User ID: [ 1234         ]    | #1 The Godfather      ★ 4.73  [GB]         | Key Drivers         |
| Top-K:  [==== 10 ======]     | #2 Shawshank Redemption ★ 4.68 [GB]         | - movie_avg_rating  |
| Diversity: [ON]              | #3 Pulp Fiction       ★ 4.61  [GB]         | - user_avg_rating   |
| Explainability: [ON]         | #4 Dark Knight        ★ 4.57  [GB]         | - movie_popularity  |
|                              | #5 Matrix             ★ 4.54  [GB]         |                     |
| Inference Model              |                                             | Explanation         |
| [Baseline              v]    | Each card includes: genre chips,            | "Recommended because|
| [Linear Regression     v]    | confidence bar, and model tag               | you highly rate     |
| [Random Forest         v]    |                                             | crime dramas..."    |
| [Gradient Boosting     v]    |                                             |                     |
| [NCF (DL)              v]    |                                             |                     |
| [Tuned NCF (Optuna)    v]    |                                             |                     |
| [Generate Recommendations]   |                                             |                     |
+------------------------------+---------------------------------------------+---------------------+
| User Embedding Map (PCA/t-SNE)                    | Model Comparison (RMSE/MAE + time)              |
| [interactive scatter with highlighted user point] | [horizontal bars; GB currently best]             |
+--------------------------------------------------------------------------------------------------+
| Ask MovieMind Agent: "Suggest underrated sci-fi under 2 hours for a 25-year-old engineer"      |
+--------------------------------------------------------------------------------------------------+
```

## 2) Core User Flows

1. User chooses `userId` and `top_k`.
2. User selects one model from the inference model selector.
3. User clicks `Generate Recommendations`.
4. App shows ranked list + explanation panel + optional embedding highlight.
5. User compares model behavior via Model Comparison panel.

## 3) Model Selector Requirement (Critical)

The UI must expose all available trained models:
- `Baseline`
- `Linear Regression`
- `Random Forest`
- `Gradient Boosting`
- `NCF (DL)`
- `Tuned NCF (Optuna)`

Backend should route inference based on selected model key.

## 4) Visual Design System (Iron Man Theme)

- Background: near-black / deep navy gradient.
- Accent A: neon cyan (data/active states).
- Accent B: warm orange (alerts/highlights).
- Cards: glassmorphism (blur + subtle border glow).
- Typography: clean sans-serif, high contrast, readable at distance.
- Motion: gentle glow pulse on active cards, no distracting heavy animation.

## 5) Data Panels

### Top Recommendations
- Movie title
- Predicted rating
- Genre tags
- "Chosen by model" badge
- Optional confidence ribbon

### Why This Recommendation?
- Top feature importance bars (for tree models)
- Model-specific explanation notes
- For embedding models, nearest-neighbor style explanation

### Embedding Map
- 2D projection (PCA or t-SNE)
- Highlight current user
- Hover shows nearby user cluster ids

### Model Comparison
- RMSE and MAE bars
- Optional runtime bar (training/inference)
- Visual marker on current selected model

## 6) Responsiveness

- Desktop: 3-column control/content/explanation layout.
- Tablet: controls collapse to top drawer; two-column content.
- Mobile: card-stack layout; charts under accordion panels.

## 7) Build Notes for Implementation

- Recommended stack: Streamlit with custom CSS first, then React/FastAPI if needed.
- Keep model selection state as a single source of truth.
- Cache loaded model artifacts for fast switching.
- Always display active model name next to every prediction output.

## 8) Milestone Output Checklist

- [ ] App shell with themed layout
- [ ] Working model selector for all models
- [ ] Recommendation generation for selected model
- [ ] Explanation side panel
- [ ] Embedding map panel
- [ ] Model comparison panel
- [ ] Evidence screenshots saved under `evidence/ui_mockups/`
