# Phase 8 Milestone Log (2026-04-30)

## Scope Completed
- Finalized Phase 8 web app + NLP runtime integration.
- Added local system setup helper for Ollama:
  - `scripts/setup_local_ollama.sh`
- Added repeatable local LLM smoke test:
  - `scripts/test_local_llm.sh`
- Added dedicated local LLM documentation:
  - `LOCAL_LLM_WIKI.md`

## Executed Commands
- `bash scripts/setup_local_ollama.sh`
- `uvicorn src.api.app:app --host 127.0.0.1 --port 8000`
- `bash scripts/test_local_llm.sh`

## Pass/Fail Outcomes
- API liveness check: **PASS**
  - Evidence: `evidence/phase8/local_llm_smoke_2026-04-30_23-31-48.txt`
- Local LLM runtime check (`runtime_mode=local-llm`): **PASS**
  - `parsed_by=local-llm-ollama`
  - model used: `llama3.1:8b`
  - latency: `curl_total_sec=3.198751`, `api_x_latency_ms=3195.36`
  - Evidence: `evidence/phase8/local_llm_smoke_2026-04-30_23-31-48.txt`
- Rule parser control check (`runtime_mode=rule-only`): **PASS**
  - `parsed_by=rule-parser`
  - latency: `curl_total_sec=0.003033`, `api_x_latency_ms=1.56`
  - Evidence: `evidence/phase8/local_llm_smoke_2026-04-30_23-31-48.txt`

## Errors and Fixes
- Initial local LLM attempts fell back due read timeout to Ollama (`timeout=12s`).
- Resolved by increasing timeout and rerunning smoke gate until local parser path returned `local-llm-ollama`.

## Decision Notes
- Keep deterministic fallback path active to avoid user-facing failure when local model is slow/unavailable.
- Keep local LLM orchestration schema-bounded:
  - model interprets language only;
  - backend controls routing and final API calls.

