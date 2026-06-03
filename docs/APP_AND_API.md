# MovieMind WebApp-Agent Wiki

**Reviewers:** use [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) for clone → build → run → checklist.

**Local only:** FastAPI and Streamlit are meant for **localhost** (`127.0.0.1`). This stack has **no TLS, API auth, or rate limits** — see [`SECURITY.md`](SECURITY.md).

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

### `POST /agent/query` (multi-step tool agent)
- Purpose: Ollama **`/api/chat`** loop with **tools** (`list_available_models`, `get_user_summary`, `get_recommendations` + optional genre filter). Separate from quick `/nlp/query`.
- Full technical detail: **`docs/AGENT.md`**.

### `POST /agent/query/stream` (tool agent, SSE)
- Purpose: Same body as **`/agent/query`**; response **`text/event-stream`** with JSON events (`assistant`, `tool`, `done`, optional `error`) so UIs can update **between** Ollama round-trips.
- Ollama itself is still called with **`stream: false`** per round for reliable **`tool_calls`**; SSE is FastAPI **`StreamingResponse`** around **`iter_tool_agent_events`** — see **`docs/AGENT.md`** § streaming.

## 2b) Ollama monitor tab (Streamlit)

The **Ollama** tab calls the local daemon’s HTTP API from the **Streamlit server** using `MOVIEMIND_OLLAMA_URL` (defaults to `http://127.0.0.1:11434`). It shows `/api/version`, `/api/tags`, and `/api/ps`, plus relevant env vars. Click **Refresh** to update the snapshot without hammering Ollama on every Streamlit rerun.

## 3) Runtime Modes and Usage Guidance

API `POST /nlp/query` accepts **`local-llm`** or **`api-llm`** only (see `src/api/schemas.py`).

### `local-llm`
- Ollama `/api/generate` with strict JSON guardrails.
- Returns **HTTP 503** if Ollama is down or returns invalid JSON (no silent rule fallback).
- Env: `MOVIEMIND_OLLAMA_URL`, `MOVIEMIND_OLLAMA_MODEL`, `MOVIEMIND_OLLAMA_TIMEOUT_SEC`.

### `api-llm`
- Guarded deterministic fallback unless `MOVIEMIND_API_LLM_ENABLED` is set.

### Multi-step tool agent
- Separate from NLP modes: `POST /agent/query` and `/agent/query/stream` — always Ollama `/api/chat` with tools. See **`docs/AGENT.md`**.

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

