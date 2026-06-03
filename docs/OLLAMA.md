# Local LLM Wiki (Ollama)

## Purpose

This document explains how MovieMind runs the local LLM path for NLP query parsing.

The local LLM mode converts natural-language user requests (for example, "top 5 action movies for user 10") into a strict structured payload used by the API/UI.

## Where It Lives

- Parser entrypoint: `src/api/nlp.py`
- API endpoint: `POST /nlp/query` in `src/api/app.py`
- UI trigger: Recommend page in `app/streamlit_app.py`

## Runtime Modes

`runtime_mode` supports:

- `rule-only`: deterministic regex parser
- `local-llm`: Ollama-backed parser with strict guardrails
- `api-llm`: optional external mode (guarded fallback when not configured)

## Local LLM Flow

1. UI sends `POST /nlp/query` with `runtime_mode="local-llm"` and `query`.
2. API calls `parse_query()` in `src/api/nlp.py`.
3. `parse_query()` routes to `_local_llm_parse()`.
4. `_local_llm_parse()` calls Ollama at `/api/generate` with a schema-constrained prompt.
5. Response is parsed and validated into:
   - `intent`
   - `filters` (`user_id`, `top_n`, `genre`)
   - `model_hint`
6. If local model output is invalid or Ollama is down, parser safely falls back to deterministic rules.

## Guardrails

- Intent is forced to one of: `recommend | predict | explain`.
- `top_n` is bounded to `1..100`.
- `user_id` is coerced to positive integer.
- `genre` is accepted only from supported genre list.
- `model_hint` is accepted only from known model IDs.
- Fallback parser always returns schema-safe output.

## Environment Variables

- `MOVIEMIND_OLLAMA_URL`  
  Default: `http://127.0.0.1:11434`
- `MOVIEMIND_OLLAMA_MODEL`  
  Default: `llama3.1:8b`
- `MOVIEMIND_OLLAMA_TIMEOUT_SEC`  
  Default: `12`

## Setup

1. Run helper setup script:
   - `bash scripts/setup_local_ollama.sh`
2. Optional custom model:
   - `OLLAMA_MODEL=llama3.1:8b bash scripts/setup_local_ollama.sh`
3. Start MovieMind API:
   - `uvicorn src.api.app:app --host 127.0.0.1 --port 8000`
4. Start Streamlit:
   - `streamlit run app/streamlit_app.py --server.port 8502`

## API Example

Request:

```json
{
  "query": "top 5 action movies for user 10 with tuned model",
  "runtime_mode": "local-llm"
}
```

Response shape:

```json
{
  "runtime_mode": "local-llm",
  "parsed_by": "local-llm-ollama",
  "confidence": 0.82,
  "intent": "recommend",
  "filters": {
    "user_id": 10,
    "top_n": 5,
    "genre": "action"
  },
  "model_hint": "ncf_tuned",
  "explanation": "..."
}
```

If Ollama is unavailable, `parsed_by` becomes `local-llm-fallback`.

## Validation Checklist

- `GET /health` returns `ok`.
- `POST /nlp/query` works with `rule-only`.
- `POST /nlp/query` works with `local-llm`.
- Local mode returns `local-llm-ollama` when Ollama is running.
- Local mode returns safe fallback when Ollama is stopped.
- Run automated smoke test and capture evidence:
  - `bash scripts/test_local_llm.sh`
  - output file path: `evidence/phase8/local_llm_smoke_<timestamp>.txt`

## Troubleshooting

- **Connection refused to Ollama**: verify Ollama is running and URL is correct.
- **Slow response**: reduce model size or increase timeout.
- **Unexpected parse fields**: guardrails drop unsupported fields by design.
- **No model hint selected**: parser may return `null` if not confident/valid.
