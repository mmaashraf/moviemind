# Phase 7 Milestone Log (2026-04-24)

## Scope Completed
- Implemented FastAPI app with endpoints:
  - `GET /health`
  - `GET /models`
  - `GET /models/{model_id}/info`
  - `POST /predict`
  - `POST /recommend`
  - `POST /nlp/query`
- Implemented model registry for Baseline, ML, DL, Tuned DL.

## Executed Commands
- `source .venv/bin/activate && python - <<PY ... TestClient smoke ... PY`
- `source .venv/bin/activate && uvicorn src.api.app:app --host 127.0.0.1 --port 8000`
- `curl http://127.0.0.1:8000/health`
- `curl http://127.0.0.1:8000/models`
- `curl -X POST http://127.0.0.1:8000/recommend ...`

## Pass/Fail Outcomes
- API smoke test: **PASS**
  - Evidence: `evidence/phase7/api_smoke_test_2026-04-24.txt`
- Live endpoint checks: **PASS**
  - Evidence:
    - `evidence/phase7/health_live_2026-04-24.json`
    - `evidence/phase7/models_live_2026-04-24.json`
    - `evidence/phase7/recommend_live_2026-04-24.json`

## Errors and Fixes
1. `ModuleNotFoundError: No module named 'fastapi'`
   - Fix: created local venv and installed requirements.

2. `AttributeError` when calling `get_params()` on unpickled sklearn model
   - Cause: sklearn version mismatch with persisted artifacts.
   - Fix: made registry robust by avoiding strict `get_params()` dependency and marking models unavailable only on actual load failure.

## Decision Notes
- Kept inference robust under artifact/version mismatch by handling model-loading exceptions gracefully.
- Added clipped rating output to keep predictions inside MovieLens rating bounds (1-5).

