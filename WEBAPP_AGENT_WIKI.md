# MovieMind WebApp-Agent Wiki

## 1) System Architecture

MovieMind Phase 7-8 uses a backend-plus-frontend split:

- FastAPI backend in `src/api/` for model listing, prediction, recommendation, and NLP query parsing.
- Streamlit UI in `app/streamlit_app.py` for human interaction and demo flow.
- Existing model artifacts from previous phases in `models/` as inference backends.

High-level runtime flow:

1. User interacts in Streamlit.
2. Streamlit calls FastAPI endpoints (`/models`, `/predict`, `/recommend`, `/nlp/query`).
3. FastAPI model registry loads/uses Baseline, ML, DL, and Tuned DL models.
4. Responses return to Streamlit for rendering and inspector/explain views.

## 2) Endpoint Catalog and Contracts

### `GET /health`
- Purpose: service status check.
- Response: `{ "status": "ok", "service": "moviemind-api" }`.

### `GET /models`
- Purpose: list model options and availability.
- Response: list of model summaries:
  - `model_id`
  - `display_name`
  - `family`
  - `artifact_path`
  - `available`

### `GET /models/{model_id}/info`
- Purpose: model inspector data for UI.
- Response includes:
  - identity fields (`model_id`, `display_name`, `family`)
  - `params`
  - `metrics`
  - `inspector` (family-specific fields)

### `POST /predict`
- Purpose: predict one user-movie rating.
- Request:
  - `model_id`
  - `user_id`
  - `movie_id`
- Response:
  - predicted rating (clipped to 1-5 range)
  - clipping indicator

### `POST /recommend`
- Purpose: top-N movie recommendations for one user.
- Request:
  - `model_id`
  - `user_id`
  - `top_n`
  - `diversity_alpha`
- Response:
  - ordered list of recommendation items with title, genres, and score.

### `POST /nlp/query`
- Purpose: parse natural-language intent into structured recommendation controls.
- Request:
  - `query`
  - `runtime_mode` in `rule-only | local-llm | api-llm`
- Response:
  - `intent`
  - `filters` (schema-bounded)
  - `model_hint`
  - `confidence`
  - parser metadata and explanation

## 3) Runtime Modes and Usage Guidance

### `Rule-only`
- Deterministic regex/keyword parser.
- Best for reproducibility and strict evaluations.

### `Local LLM`
- Ollama-backed local parser with strict JSON-schema guardrails.
- Runtime calls local endpoint and validates intent/filters/model hint.
- If local model is unavailable, automatically falls back to deterministic parser.
- Env controls:
  - `MOVIEMIND_OLLAMA_URL` (default `http://127.0.0.1:11434`)
  - `MOVIEMIND_OLLAMA_MODEL` (default `llama3.1:8b`)
  - `MOVIEMIND_OLLAMA_TIMEOUT_SEC` (default `12`)

### `API LLM`
- Optional cloud-LLM path.
- Disabled unless configured.
- Current implementation keeps fallback parser behavior when API is not configured.

## 4) Model Inspector Semantics

Inspector view is moderate scope (Phase 8 lock):

- **General fields**
  - model family
  - availability
  - artifact path
  - parameters
  - known metrics (when available)

- **Gradient Boosting**
  - top feature importances from fitted estimator

- **DL / Tuned DL**
  - embedding dimensions
  - parameter count
  - inference device

Advanced compare/download diagnostics are deferred to Phase 8.x.

## 5) Known Failure Modes and Troubleshooting

1. **Missing dependencies**
   - Symptom: `ModuleNotFoundError` for FastAPI/Streamlit/etc.
   - Fix: activate project venv and install requirements.

2. **Model artifact compatibility warnings**
   - Symptom: sklearn pickle version warnings.
   - Fix: inference may still work; if it breaks, retrain and re-save artifacts in current environment.

3. **API unavailable from UI**
   - Symptom: Streamlit shows model load/health errors.
   - Fix: start API server and verify `MOVIEMIND_API_URL`.

4. **NLP API mode not configured**
   - Symptom: API LLM mode falls back to guarded parser.
   - Fix: configure API credentials/flag when API LLM path is implemented.

5. **Local LLM unavailable**
   - Symptom: Local LLM mode returns `local-llm-fallback`.
   - Fix: start Ollama and pull selected model, then re-test `/nlp/query`.

## 6) Reproducible Run/Test Steps

From project root:

1. Create and activate environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `python -m pip install -r requirements.txt`

2. Start API:
   - `uvicorn src.api.app:app --host 127.0.0.1 --port 8000`

3. Start UI:
   - `streamlit run app/streamlit_app.py --server.port 8502`

4. Smoke checks:
   - `curl http://127.0.0.1:8000/health`
   - `curl http://127.0.0.1:8000/models`
   - `curl -X POST http://127.0.0.1:8000/nlp/query -H "Content-Type: application/json" -d '{"query":"top 5 action movies for user 10 tuned model","runtime_mode":"local-llm"}'`
   - `bash scripts/test_local_llm.sh`
     - Saves smoke + latency evidence to `evidence/phase8/local_llm_smoke_<timestamp>.txt`

5. Evidence capture:
   - Save command outputs/logs under `evidence/phase7/` and `evidence/phase8/`.

