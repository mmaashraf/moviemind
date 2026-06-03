# Reviewer setup and verification guide

**This is the dedicated setup guide** for cloning, installing, building artifacts, running the app, and verifying results.

- **Project overview:** [`README.md`](README.md)  
- **Notebooks + training script order:** [`REPLICATION.md`](REPLICATION.md)  
- **Tool agent / SSE:** [`docs/AGENT.md`](docs/AGENT.md)  
- **UI ↔ API:** [`docs/APP_AND_API.md`](docs/APP_AND_API.md)  
- **Ollama NLP parse:** [`docs/OLLAMA.md`](docs/OLLAMA.md)  
- **Pack / download / restart (end-to-end):** [`docs/ARTIFACTS_AND_RUNTIME.md`](docs/ARTIFACTS_AND_RUNTIME.md)

---

## 1. What you are verifying

| Layer | What it proves |
|--------|----------------|
| **Data + features** | MovieLens 1M loads; processed CSVs exist |
| **Models** | At least one recommender artifact loads in the API |
| **API** | `/health`, `/models`, `/recommend` return sensible JSON |
| **Streamlit UI** | Manual recommendations render a table |
| **Ollama (optional)** | Local LLM parse and/or multi-step tool agent |
| **Tool agent (optional)** | Multi-turn trace, recommendations, `Turns used` > 0 |

---

## 2. What ships in git vs what you generate

Yes — others can **clone once** and choose how deep they go: read notebooks and committed results without re-running anything, **run the web app** after a one-time artifact build (or an author-supplied zip), or **optionally** re-run notebooks and the full training pipeline.

### 2.1 Already in the repo (no training required to **view**)

| Path | Purpose |
|------|---------|
| `notebooks/` | EDA and long-tail / cold-start analysis (`01_eda.ipynb`, `02_long_tail_and_cold_start.ipynb`) |
| `evidence/` | Phase logs, metrics CSVs, smoke-test outputs, presentation notes (see `evidence/README.md`) |
| `src/`, `app/`, `scripts/`, `tests/`, docs | Application and pipeline source |

Open notebooks in Jupyter/VS Code and browse `evidence/` — **no** `data/` or `models/` folder is required for that.

### 2.2 Not in git (required to **run** recommendations in the app)

These paths are **gitignored** (~160 MB on disk after a typical build):

| Path | Contents |
|------|-----------|
| `data/` | Raw MovieLens + `data/processed/*.csv` |
| `models/` | `*.pkl`, `*.pt`, training logs |

Generate them locally (§4) or unpack **artifacts from the author** (release zip, Drive link, etc.) into `moviemind/data/` and `moviemind/models/` with the same layout as §4.2–4.3.

### 2.3 Three reviewer paths

| Path | Goal | Build `data/` + `models/`? | Run notebooks? |
|------|------|----------------------------|----------------|
| **Inspect** | Trust the analysis without recompute | No | No — use committed `evidence/` and notebook source |
| **Run app** | Streamlit + FastAPI demo | Yes — §4.2 + §4.3 Path A or C (or author zip) | No |
| **Reproduce** | Re-train and refresh evidence | Yes — §4.3 Path B + README phase steps | Yes — optional, in order |

**Inspect-only example:**

```bash
git clone https://github.com/mmaashraf/moviemind.git
cd moviemind
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # only if you want to execute notebook cells
jupyter lab notebooks/01_eda.ipynb  # optional
ls evidence/phase6 evidence/phase9_split_eval
```

**Run-app example:** same clone + venv, then §4.2–4.6 (download data, `build_model_artifacts.py features` + `ml`, start API + UI).

**Reproduce example:** run notebooks, then `python3 scripts/build_model_artifacts.py all --skip-tune-dl` (or phase-by-phase commands in [`README.md`](README.md)).

### 2.4 Clone and run in ~2 minutes (pre-built artifacts)

Full diagram and script reference: [`docs/ARTIFACTS_AND_RUNTIME.md`](docs/ARTIFACTS_AND_RUNTIME.md).

Git cannot hold `data/` and `models/` at full size without a separate **release tarball**. After you publish one:

**Author (once):** `bash scripts/pack_review_artifacts.sh` → upload `moviemind-artifacts.tar.gz` to a GitHub Release.

**Reviewer:**

```bash
git clone https://github.com/mmaashraf/moviemind.git && cd moviemind
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MOVIEMIND_ARTIFACTS_URL="https://github.com/<org>/moviemind/releases/download/<tag>/moviemind-artifacts.tar.gz"
bash scripts/download_review_artifacts.sh
bash scripts/restart_moviemind.sh
```

Then open http://127.0.0.1:8502 (API on :8000). Time is dominated by clone + pip + download (~160 MB), not training.

Without a release URL, use §4.2–4.3 (15–45 min minimum build).

### 2.5 Will re-running notebooks break the app?

**No** — for the notebooks in this repo today.

| What notebooks touch | App dependency |
|----------------------|----------------|
| Read-only: `data/ml-1m/*.dat` (raw MovieLens) | API/UI use `data/processed/*.csv` and `models/*` |
| Plots stay inside the notebook (no `savefig` into `data/` or `models/`) | Unaffected |

Re-running EDA or long-tail notebooks is safe even while the app is running, as long as `data/ml-1m` is present (same files `python3 src/data_loader.py` would download).

**What can break or confuse the app**

| Action | Risk |
|--------|------|
| Re-run `scripts/build_model_artifacts.py` (features / ml / dl / post) | Overwrites `data/processed/` and `models/`; API may need a restart; pickle load errors if sklearn version differs from the one used to train |
| Delete or rename `data/processed/` or `models/*.pkl` | Recommendations fail until you rebuild or re-download artifacts |
| Edit processed CSV schemas by hand | API feature columns may not match training |

Notebooks and the training pipeline are **separate paths** until someone deliberately runs the build scripts again.

---

## 3. Prerequisites

- **Python 3.10+** (3.11–3.13 tested in development)
- **~8 GB disk** for data + models; **16 GB RAM** recommended if using Ollama locally
- **macOS / Linux** (Windows may work; Ollama install steps differ)
- Two terminal windows (API + UI)
- **Ollama** only if you test **Local LLM** or **Multi-step tool agent** (see §5)

---

## 4. Standard setup (required for all reviewers)

Run every command from the **`moviemind/`** directory (repo root for this project).

### 4.1 Clone and Python environment

```bash
cd moviemind
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4.2 Download data

```bash
python3 src/data_loader.py
```

**Pass if these exist:**

- `data/ml-1m/ratings.dat`
- `data/ml-1m/movies.dat`
- `data/ml-1m/users.dat`

### 4.3 Build features and models (pick one path)

#### Path A — Recommended minimum (~15–45 min)

Enough for **Manual** recommend and most API checks (baseline + ML models; DL optional).

```bash
python3 scripts/build_model_artifacts.py features
python3 scripts/build_model_artifacts.py ml
```

**Pass if these exist:**

- `data/processed/train_features.csv`
- `data/processed/test_features.csv`
- `models/linear_regression.pkl`
- `models/random_forest.pkl`
- `models/gradient_boosting.pkl`

#### Path B — Full artifacts (~hours; skip Optuna if `best_dl_params.txt` exists)

```bash
python3 scripts/build_model_artifacts.py all --skip-tune-dl
```

Adds DL checkpoints and Phase 6 evidence. Use when you need **NCF** models in the UI.

**Pass if additionally present (when post step completes):**

- `models/ncf_model.pt`
- `models/ncf_tuned_best.pt` (after `post` phase)

#### Path C — Smoke test only (fastest API check)

If you only need to prove the API boots:

```bash
python3 scripts/build_model_artifacts.py features
```

Then use model **`baseline_global_mean`** in the UI (no `.pkl` required). Recommendations still need processed features from the `features` step.

### 4.4 Optional unit check (sklearn pickle roundtrip)

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

---

## 5. Ollama (only for LLM / tool agent)

**Not required** for:

- **Manual** recommendations
- **Parse Query** with **API LLM** (uses guarded fallback parser in code unless you enable a real API client)

**Required** for:

- **Parse Query** with **Local LLM**
- **Multi-step tool agent (Ollama)**

### 5.1 Install and start

```bash
bash scripts/setup_local_ollama.sh
```

This installs/checks the CLI, ensures something is listening on `http://127.0.0.1:11434`, and pulls **`llama3.1:8b`** by default.

**You do not need a separate `ollama serve` terminal** if:

- The Ollama **desktop app** is running, or
- `brew services start ollama` succeeded, or
- The setup script started the daemon in the background.

**Pass check:**

```bash
curl -s http://127.0.0.1:11434/api/tags | head
```

You should see JSON listing models.

### 5.2 Optional NLP smoke script

```bash
bash scripts/test_local_ollama.sh
```

Writes a log under `evidence/phase8/` with `parsed_by=local-llm-ollama` when successful.

### 5.3 Environment (optional overrides)

| Variable | Default | Used by |
|----------|---------|---------|
| `MOVIEMIND_OLLAMA_URL` | `http://127.0.0.1:11434` | API + UI Ollama tab |
| `MOVIEMIND_OLLAMA_MODEL` | `llama3.1:8b` | NLP + tool agent |
| `MOVIEMIND_AGENT_TIMEOUT_SEC` | falls back to `120` | Tool agent (raise if timeouts) |

---

## 6. Run the application

### Option A — One script (recommended for reviewers)

Stops anything listening on the API/UI ports, then starts both in the background with logs:

```bash
cd moviemind
bash scripts/restart_moviemind.sh
```

With Ollama check/pull (for tool agent / Local LLM):

```bash
bash scripts/restart_moviemind.sh --with-ollama
```

Other flags:

```bash
bash scripts/restart_moviemind.sh --stop-only    # kill API + UI only
bash scripts/restart_moviemind.sh --start-only   # start without killing first
bash scripts/restart_moviemind.sh --foreground   # API in background, Streamlit in terminal
```

After start:

- **API:** `http://127.0.0.1:8000` (docs at `/docs`)
- **UI:** `http://127.0.0.1:8502`
- **Logs:** `evidence/runtime/api.log`, `evidence/runtime/ui.log`

Uses `.venv` automatically if present. Does **not** stop Ollama unless you use `--with-ollama` (that only ensures Ollama is running, it does not kill it).

### Option B — Two terminals (manual)

**Terminal 1 — API**

```bash
cd moviemind
source .venv/bin/activate
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Streamlit UI**

```bash
cd moviemind
source .venv/bin/activate
streamlit run app/streamlit_app.py --server.port 8502
```

### Health checks (either option)

```bash
curl -s http://127.0.0.1:8000/health
```

Expected: `{"status":"ok","service":"moviemind-api"}` (or equivalent).

```bash
curl -s http://127.0.0.1:8000/models | python3 -m json.tool | head -40
```

Expected: list of models; **`gradient_boosting`** (and others you built) show `"available": true` when artifacts exist.

Open the UI at `http://127.0.0.1:8502`.

---

## 7. Verification checklist (UI)

Use this table and tick each row when it passes.

### 7.1 System tab

| Step | Action | Expected |
|------|--------|----------|
| 1 | Open **System** | **API is healthy** + JSON from `/health` |

### 7.2 Manual recommend (no Ollama)

| Step | Action | Expected |
|------|--------|----------|
| 1 | **Recommend** → **Manual** | Model dropdown lists models with artifacts |
| 2 | Model **Gradient Boosting**, User **25**, Top **10** | — |
| 3 | **Get Recommendations** | Table with movie titles, genres, predicted ratings |
| 4 | Change Top N | Previous table can remain until you click again (session behavior) |

### 7.3 Parse Query — Local LLM (Ollama required)

| Step | Action | Expected |
|------|--------|----------|
| 1 | **Agent (NLP)**; **Multi-step tool agent** **off** | — |
| 2 | NLP Runtime **Local LLM** | — |
| 3 | Query: `top 5 action movies for user 10` | — |
| 4 | **Parse Query (NLP)** | Parsed intent panel; `parsed_by` contains `local-llm-ollama` (or clear error if Ollama is down) |
| 5 | Enable auto-recommend if shown | Recommendations table updates |

**Note:** The API accepts `runtime_mode` **`local-llm`** or **`api-llm`** only (see `NLPQueryRequest` in `src/api/schemas.py`). There is no `rule-only` mode on `/nlp/query` in the current code.

### 7.4 Tool agent (Ollama required)

| Step | Action | Expected |
|------|--------|----------|
| 1 | **Multi-step tool agent (Ollama)** **on** | — |
| 2 | **Stream agent steps (SSE)** on (default) | — |
| 3 | **Agent max turns** e.g. **8** | — |
| 4 | Query (natural): *For user 25, check which models are available, look at their taste, then recommend about 8 documentary or fiction-style movies with a bit of diversity.* | — |
| 5 | **Run tool agent** | **Agent status: running…** then **done** |
| 6 | Expand **Agent trace** | At least one **Observation** for a tool (not only JSON text in Agent reply) |
| 7 | **Agent reply** | Plain-language answer with **real movie titles** from tools |
| 8 | Caption | `Turns used: 2` or higher; model name shown |

**Fail signals:**

- **503 / Ollama** errors → §5 and §9
- Agent reply is only `{"name":"get_recommendations",...}` lines → restart API (latest code); see [`docs/AGENT.md`](docs/AGENT.md) pseudo-tool guardrail
- **404** on stream → restart API so `/agent/query/stream` is registered

### 7.5 Ollama tab

| Step | Action | Expected |
|------|--------|----------|
| 1 | Open **Ollama** tab → **Refresh Ollama snapshot** | `/api/version` reachable; `/api/tags` lists models |

### 7.6 User resource 404 (API)

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/users/999999/summary
```

Expected: **`404`** for an out-of-range user id.

---

## 8. Verification checklist (API only)

If you prefer curl over the UI:

```bash
# Recommend
curl -s -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"model_id":"gradient_boosting","user_id":25,"top_n":5,"diversity_alpha":0}' \
  | python3 -m json.tool | head -30
```

Expected: `recommendations` array with `title`, `genres`, `predicted_rating`.

```bash
# NLP parse (local-llm; needs Ollama)
curl -s -X POST http://127.0.0.1:8000/nlp/query \
  -H "Content-Type: application/json" \
  -d '{"query":"top 5 action movies for user 10","runtime_mode":"local-llm"}' \
  | python3 -m json.tool
```

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| No models in dropdown / `available: false` | Missing `models/*.pkl` or `*.pt` | Run §4.3 Path A or B |
| `FileNotFoundError` on recommend | Missing `data/processed/` | Run `features` step |
| Ollama connection error | Daemon not running | `bash scripts/setup_local_ollama.sh` or open Ollama app |
| Read timeout on tool agent | Slow GPU/CPU or low timeout | `export MOVIEMIND_AGENT_TIMEOUT_SEC=300` then restart API |
| SSE 404 | Old API process | Restart uvicorn with `--reload` on latest branch |
| Only 1 **Turn used** | Model batched tools in one round | Normal for short prompts; use longer multi-step query in [`docs/AGENT.md`](docs/AGENT.md) §8 |

---

## 10. Suggested reviewer time budget

| Goal | Time |
|------|------|
| API + Manual UI only | ~30–60 min (incl. features + ML build) |
| + Local LLM parse | +5–20 min (first Ollama load slower) |
| + Ollama + tool agent | +20–60 min (first model load is slow) |
| Full `all --skip-tune-dl` build | +1–3 hours |

---

## 11. Default branch

Use **`main`** (tool agent merged via PR #11). Confirm OpenAPI includes:

- `POST /agent/query`
- `POST /agent/query/stream`

```bash
curl -s http://127.0.0.1:8000/openapi.json | python3 -c "import sys,json; p=json.load(sys.stdin)['paths']; print([k for k in p if 'agent' in k])"
```

---

## 12. Evidence you can attach to a review

Optional proof for a written review:

- Screenshot: Manual recommendations table
- Screenshot: Tool agent trace + Agent reply
- Terminal: `curl /health` and `curl /models`
- File: `evidence/phase8/local_llm_smoke_*.txt` after `test_local_ollama.sh`

---

*Last aligned with Phase 8x (tool agent, SSE, Ollama monitor tab). Update this file when submission instructions change.*
