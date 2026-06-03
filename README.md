# MovieMind: Movie Recommendation System & AI Agent

End-to-end capstone on **MovieLens 1M**: feature engineering, ML/DL models, FastAPI backend, Streamlit UI, and optional **Ollama** NLP + multi-step **tool agent**.

**Best model in this repo’s evals:** Gradient Boosting (RMSE ~0.897). See `evidence/phase9_split_eval/` for split summaries.

---

## Setup and run (read this first)

**Dedicated guide:** [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md)

That file is the single place for:

- what to **clone** and which branch to use  
- **install** (venv, `requirements.txt`, Ollama when needed)  
- **what is already in git** (`notebooks/`, `evidence/`) vs **gitignored** `data/` and `models/`  
- **three paths:** inspect notebooks/evidence only · run the app · optional full reproduction  
- **~2 min run:** GitHub Release tarball + `scripts/download_review_artifacts.sh` (see §2.4); re-running notebooks does **not** break the app (§2.5)  
- **build** artifacts when you need the live API/UI and have no release zip  
- **run** the app and **what is launched** (API on port **8000**, UI on **8502**)  
- **`bash scripts/restart_moviemind.sh`** (stop old processes, start both)  
- **verify** pass/fail checklist and troubleshooting  

This README keeps **project context** and **full phase-by-phase replication** for developers and reviewers who need implementation depth.

---

## Repository

```bash
git clone git@github.com:mmaashraf/moviemind.git
cd moviemind
```

HTTPS: `git clone https://github.com/mmaashraf/moviemind.git`

---

## What runs when you “start the app”

| Process | Technology | Default URL |
|---------|------------|-------------|
| **Backend (API)** | `uvicorn src.api.app:app` | http://127.0.0.1:8000 |
| **Frontend (UI)** | `streamlit run app/streamlit_app.py` | http://127.0.0.1:8502 |

The UI calls the API at `MOVIEMIND_API_URL` (default `http://127.0.0.1:8000`).  
**Ollama** (`http://127.0.0.1:11434`) is separate — only for **Local LLM** parse and the **tool agent**, not for Manual recommend.

Quick start: [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) §4–6.

---

## Current project status

| Phase | Status | Notes |
|-------|--------|--------|
| 1 Setup | Done | Repo layout, data loader |
| 2 EDA | Done | Notebooks under `notebooks/` |
| 3 Features | Done | `data/processed/*.csv` |
| 4 ML | Done | Best: Gradient Boosting RMSE ~0.897 |
| 5 DL | Done | NCF baseline; GB still better |
| 5b Optuna | Done | 50 trials; tuned DL RMSE ~1.006 |
| 6 Post-analysis | Done | PCA, feature importance; t-SNE optional |
| 7 API | Done | FastAPI + model registry |
| 8 UI + NLP | Done | Streamlit; `/nlp/query` (`local-llm`, `api-llm`) |
| 8x Tool agent | Done on feature branch | `/agent/query`, SSE; see `PROGRESS_TRACKER.md` |

Detail: [`PROGRESS_TRACKER.md`](PROGRESS_TRACKER.md)

---

## Architecture and features

- **Data:** MovieLens 1M with user demographics (age, gender, occupation).
- **Models (registry):** `baseline_global_mean`, `linear_regression`, `random_forest`, `gradient_boosting`, `ncf_baseline`, `ncf_tuned` — availability depends on files under `models/`.
- **API:** Unified `ModelRegistry` for predict, recommend, user summary, NLP parse, tool agent.
- **UI:** Streamlit tabs — Recommend, Model Inspector, Embedding Space, Model Visualizers, Lifecycle Evidence, AI Concepts, Ollama monitor, System.
- **XAI / analysis:** GB feature importance, user embedding PCA (`evidence/phase6/`).
- **Diversity:** `diversity_alpha` on `/recommend` and agent tool `get_recommendations`.

---

## API endpoints (accurate as of Phase 8x branch)

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | Liveness |
| GET | `/models` | Includes `available` per artifact |
| GET | `/models/{model_id}/info` | Inspector payload |
| GET | `/users/{user_id}/summary` | **404** if user id out of range |
| POST | `/predict` | Single rating |
| POST | `/recommend` | Top-N list |
| POST | `/nlp/query` | `runtime_mode`: **`local-llm`** or **`api-llm`** only |
| POST | `/agent/query` | Multi-step tool agent (JSON) — **Phase 8x branch** |
| POST | `/agent/query/stream` | Same agent, SSE — **Phase 8x branch** |

Docs: http://127.0.0.1:8000/docs

**NLP runtime modes (API):**

- **`local-llm`** — Ollama `/api/generate`; returns **503** if Ollama is down (no silent rule fallback).
- **`api-llm`** — guarded fallback unless `MOVIEMIND_API_LLM_ENABLED` is set.

**UI Recommend modes:**

- **Manual** — no Ollama; uses `/recommend` directly.
- **Agent (NLP)** + **Parse Query** — `/nlp/query` with Local LLM or API LLM.
- **Multi-step tool agent** — `/agent/query` or `/agent/query/stream`; always Ollama; independent of NLP runtime dropdown.

---

## Full replication by phase (implementation)

Run from **`moviemind/`** with venv active.  
`data/` and `models/` are **gitignored** — generate locally.

### 0) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 1) Data

```bash
python3 src/data_loader.py
```

Expected: `data/ml-1m/ratings.dat`, `movies.dat`, `users.dat`

### 2) EDA (optional for run-only reviewers)

- `notebooks/01_eda.ipynb`
- `notebooks/02_long_tail_and_cold_start.ipynb`

### 3) Features

```bash
python3 src/features.py
```

Expected: `data/processed/train_features.csv`, `test_features.csv`, `val_features.csv`

### 4) ML models

```bash
python3 src/ml_models.py
```

Expected: `models/*.pkl`, `models/ml_training_log.txt`  
Sklearn saves use **pickle protocol 4** (`save_sklearn_estimator` in `src/ml_models.py`).

### 5) DL baseline

```bash
python3 src/dl_model.py
```

Expected: `models/ncf_model.pt`, `models/dl_training_log.txt`

### 5b) DL tuning (Optuna)

```bash
python3 src/tune_dl.py
```

Expected: `models/best_dl_params.txt`, `models/tune_dl_log.txt`

### 6) Post-analysis

```bash
python3 src/post_analysis.py
```

Optional t-SNE: `MOVIEMIND_ENABLE_TSNE=1 python3 src/post_analysis.py`  
Expected: `models/ncf_tuned_best.pt`, artifacts under `evidence/phase6/`

### 6.1) One-shot build script

```bash
python3 scripts/build_model_artifacts.py all --skip-tune-dl
```

Commands: `features`, `ml`, `dl`, `tune-dl`, `post`, `all`. See `python3 scripts/build_model_artifacts.py --help`.

Tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 7) API

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Or: `bash scripts/restart_moviemind.sh`

### 8) UI

```bash
streamlit run app/streamlit_app.py --server.port 8502
```

Ollama setup: `bash scripts/setup_local_ollama.sh`  
Local LLM smoke: `bash scripts/test_local_llm.sh` (needs API running)

---

## Scripts (helper)

| Script | Purpose |
|--------|---------|
| `scripts/restart_moviemind.sh` | Kill listeners on 8000/8502; start API + UI |
| `scripts/setup_local_ollama.sh` | Install/start Ollama; pull default model |
| `scripts/test_local_llm.sh` | API health + NLP smoke → `evidence/phase8/` |
| `scripts/build_model_artifacts.py` | Chain training phases |

---

## Evidence map

- `evidence/phase5b/` — DL tuning notes  
- `evidence/phase6/` — post-analysis, embeddings, plots  
- `evidence/phase7/` — API smoke logs  
- `evidence/phase8/` — UI/NLP smoke  
- `evidence/phase9_split_eval/` — 70/10/20 eval summaries  
- `evidence/runtime/` — API/UI logs from restart script  

---

## Documentation index

| Document | Audience |
|----------|----------|
| **`REVIEWER_SETUP.md`** | Clone → install → build → run → verify |
| `TOOL_AGENT_WIKI.md` | Tool agent, SSE, env vars, prompts |
| `WEBAPP_AGENT_WIKI.md` | UI ↔ API contracts |
| `LOCAL_LLM_WIKI.md` | Ollama NLP path (update if API modes change) |
| `PROGRESS_TRACKER.md` | Phase checklist |
| `CONTEXT_HANDOVER.md` | Handoff narrative |
| `AI_CONCEPTS_WIKI.md` | Course concepts |

---

## Reproducibility rules (project)

1. Modular layout: `src/`, `data/`, `models/`, `notebooks/`.  
2. Log training to console and `models/*.txt` logs.  
3. Time-based splits — no random leakage for the main pipeline.  
4. Capture evidence under `evidence/<phase>/` per milestone.  
5. Keep `PROGRESS_TRACKER.md`, `CONTEXT_HANDOVER.md`, `AI_CONCEPTS_WIKI.md` in sync when phases change.

---

## Milestone update protocol

After a major phase:

1. Save outputs in `evidence/<phase>/`.  
2. Update `PROGRESS_TRACKER.md`.  
3. Update `CONTEXT_HANDOVER.md`.  
4. Update `AI_CONCEPTS_WIKI.md` if theory changed.  
5. Update **`REVIEWER_SETUP.md`** if setup/run steps changed; update this README if replication commands changed.
