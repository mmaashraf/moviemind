# MovieMind: Movie Recommendation System & AI Agent

End-to-end capstone on **MovieLens 1M**: feature engineering, ML/DL models, FastAPI backend, Streamlit UI, and optional **Ollama** NLP + multi-step **tool agent**.

**Best model in this repo’s evals:** Gradient Boosting (RMSE ~0.897). See `evidence/phase9_split_eval/` for split summaries.

**Deployment scope:** **Local run only** (`127.0.0.1`) — no TLS, API auth, or rate limits. Not intended as a public internet service. See [`docs/SECURITY.md`](docs/SECURITY.md).

---

## Start here

| I want to… | Open |
|------------|------|
| **Run the app** (install, build, verify) | [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) |
| **Artifacts + scripts** (pack / download / restart) | [`docs/ARTIFACTS_AND_RUNTIME.md`](docs/ARTIFACTS_AND_RUNTIME.md) |
| **~2 min run** (release tarball) | [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) §2.4 |
| **Reproduce training** (notebooks + script order) | [`REPLICATION.md`](REPLICATION.md) |
| **Tool agent / SSE** | [`docs/AGENT.md`](docs/AGENT.md) |
| **UI ↔ API** | [`docs/APP_AND_API.md`](docs/APP_AND_API.md) |
| **Ollama / local LLM** | [`docs/OLLAMA.md`](docs/OLLAMA.md) |
| **All docs index** | [`docs/README.md`](docs/README.md) |
| **Security (before public)** | [`docs/SECURITY.md`](docs/SECURITY.md) |
| **Metrics & plots already run** | `evidence/phase9_split_eval/`, `evidence/phase6/` |
| **Phase checklist (author)** | [`docs/internal/PROGRESS_TRACKER.md`](docs/internal/PROGRESS_TRACKER.md) |

```bash
git clone https://github.com/mmaashraf/moviemind.git
cd moviemind
```

SSH: `git clone git@github.com:mmaashraf/moviemind.git`

---

## What runs when you start the app

| Process | Technology | Default URL |
|---------|------------|-------------|
| **Backend (API)** | `uvicorn src.api.app:app` | http://127.0.0.1:8000 |
| **Frontend (UI)** | `streamlit run app/streamlit_app.py` | http://127.0.0.1:8502 |

The UI calls the API at `MOVIEMIND_API_URL` (default `http://127.0.0.1:8000`).  
**Ollama** (`http://127.0.0.1:11434`) is only for **Local LLM** parse and the **tool agent**, not for Manual recommend.

Quick start: `bash scripts/restart_moviemind.sh` after artifacts exist — see [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) §2.4 and §7.

---

## Project status (summary)

| Phase | Status | Notes |
|-------|--------|--------|
| 1–6 | Done | Data → features → ML/DL → post-analysis |
| 7 API | Done | FastAPI + model registry |
| 8 UI + NLP | Done | `local-llm`, `api-llm` |
| 8x Tool agent | Done | `/agent/query`, `/agent/query/stream` (SSE) |

Detail: [`docs/internal/PROGRESS_TRACKER.md`](docs/internal/PROGRESS_TRACKER.md)

---

## Architecture (short)

- **Data:** MovieLens 1M + user demographics.
- **Models:** `baseline_global_mean`, sklearn regressors, NCF baseline/tuned — see `/models` when artifacts exist.
- **API:** predict, recommend, user summary, NLP parse, multi-step tool agent.
- **UI:** Recommend, inspectors, embeddings, evidence browser, Ollama monitor, system status.
- **Best eval:** Gradient Boosting; diversity via `diversity_alpha` on recommend + agent tools.

---

## API endpoints

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | Liveness |
| GET | `/models` | `available` per artifact |
| GET | `/models/{model_id}/info` | Inspector |
| GET | `/users/{user_id}/summary` | **404** if user out of range |
| POST | `/predict` | Single rating |
| POST | `/recommend` | Top-N |
| POST | `/nlp/query` | `local-llm` or `api-llm` |
| POST | `/agent/query` | Tool agent (JSON) |
| POST | `/agent/query/stream` | Tool agent (SSE) |

Interactive docs: http://127.0.0.1:8000/docs

---

## Helper scripts

| Script | Purpose |
|--------|---------|
| `scripts/restart_moviemind.sh` | Start API + UI |
| `scripts/stop_moviemind.sh` | Stop API + UI — see [`docs/ARTIFACTS_AND_RUNTIME.md`](docs/ARTIFACTS_AND_RUNTIME.md) |
| `scripts/verify_local_app.sh` | Health check for :8000 / :8502 |
| `scripts/build_model_artifacts.py` | Chain training phases — see [`REPLICATION.md`](REPLICATION.md) |
| `scripts/download_review_artifacts.sh` | Pull pre-built `data/` + `models/` — see [`docs/ARTIFACTS_AND_RUNTIME.md`](docs/ARTIFACTS_AND_RUNTIME.md) |
| `scripts/pack_review_artifacts.sh` | Create tarball for GitHub Release |
| `scripts/setup_local_ollama.sh` | Ollama install + default model |
| `scripts/test_local_llm.sh` | API + NLP smoke |

---

## Evidence (committed outputs)

- `evidence/phase5b/` — DL tuning  
- `evidence/phase6/` — post-analysis  
- `evidence/phase7/` — API smoke  
- `evidence/phase8/` — UI/NLP + screenshots  
- `evidence/phase9_split_eval/` — split eval summaries  

Index: [`evidence/README.md`](evidence/README.md)

---

## Replication and maintenance

- **Commands and notebook order:** [`REPLICATION.md`](REPLICATION.md) only (do not duplicate in this file).
- **Setup/run changes:** update [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md).
- **Phase milestones:** update [`docs/internal/PROGRESS_TRACKER.md`](docs/internal/PROGRESS_TRACKER.md).
