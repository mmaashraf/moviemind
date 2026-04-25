# Code Review - UI/API Visualizer and User Summary Fixes (2026-04-25)

## Reviewed Scope
- `app/streamlit_app.py`
- `src/api/app.py`
- `src/api/model_registry.py`
- `src/api/schemas.py`
- `requirements.txt`

## What Was Fixed
1. **User summary visibility**
   - Added backend endpoint `GET /users/{user_id}/summary`.
   - Added Streamlit fallback summary computed from local training data when API returns unavailable/404.
   - UI now shows selected-user context immediately in Recommend tab.

2. **Model visualizer completeness**
   - Raw NCF now explicitly shows hidden layers `[128, 64, 32]`.
   - Tuned NCF now shows hidden-layer units from tuned params.
   - Added neural scoring equations for clarity.
   - Added Linear Regression equation and coefficient visualization.
   - Added Random Forest tree snapshot (single tree, shallow depth for readability).

3. **Dependency reliability**
   - Added `torchinfo` to `requirements.txt`.
   - Installed in local venv for active testing.
   - Removed optional torchviz dependency path to avoid graphviz/system binary blockers in demo.

## Validation Performed
- Lint checks for touched files: **pass**
- Compile check: `python -m compileall app/streamlit_app.py` **pass**
- Dependency install check: `pip install torchinfo` **pass**

## Known Notes / Residual Risk
- If user summary endpoint still shows 404, API process likely needs restart to load latest routes.
- Random Forest tree visual is intentionally clipped to depth 2 for UX readability (full tree is too dense).
- sklearn version mismatch warnings may still appear for persisted artifacts; behavior remains functional under current guardrails.

