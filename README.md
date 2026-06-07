# MovieMind

Movie recommendations on **MovieLens 1M**: ML/DL models, **FastAPI** backend, **Streamlit** UI, optional **Ollama** NLP + tool agent.

**Local demo only** (`127.0.0.1`) — no TLS, auth, or rate limits. See [`docs/SECURITY.md`](docs/SECURITY.md).

---

## Choose what to run

| Path | Time | What you get | Guide |
|------|------|--------------|--------|
| **A — Run the app** | ~2 min (+ download) | Recommendations in the browser; API on :8000 | [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) **§2.4** |
| **B — Browse only** | ~5 min | Notebooks (read), metrics & plots in `evidence/` — no training | [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) **§2.3** (Inspect) |
| **C — Notebooks + train** | hours | Re-run EDA notebooks and/or full training pipeline | [`REPLICATION.md`](REPLICATION.md) |

Most reviewers want **Path A**.

---

## Path A — Quick start (copy-paste)

```bash
git clone https://github.com/mmaashraf/moviemind.git && cd moviemind
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export MOVIEMIND_ARTIFACTS_URL="https://github.com/mmaashraf/moviemind/releases/download/v1.0-artifacts/moviemind-artifacts.tar.gz"
bash scripts/download_review_artifacts.sh
python -m unittest discover -s tests -q
bash scripts/restart_moviemind.sh
bash scripts/verify_local_app.sh
```

Then open:

| What | URL |
|------|-----|
| **UI** (recommendations, inspectors, evidence browser) | http://127.0.0.1:8502 |
| **API** (interactive docs) | http://127.0.0.1:8000/docs |

**Try in the UI:** Recommend → **Manual** → user **1161** → model **Gradient Boosting** → Get Recommendations.

**Stop servers:** `bash scripts/stop_moviemind.sh`

First `pip install` can take several minutes (PyTorch). First API start can take up to ~90s while data loads.

Full checklists, Ollama, and tool agent: [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md).

---

## Path B — Browse without running the app

Already in git — no `data/` or `models/` download required to **read**:

- **Notebooks:** `notebooks/01_eda.ipynb`, `notebooks/02_long_tail_and_cold_start.ipynb`
- **Results:** `evidence/phase6/`, `evidence/phase9_split_eval/`, [`evidence/README.md`](evidence/README.md)

To **execute** notebook cells you still need `data/ml-1m/` (included in the release tarball from Path A, or run `python3 src/data_loader.py`). See [`REPLICATION.md`](REPLICATION.md) Tier 1.

---

## Path C — Reproduce training

Script and notebook order: [`REPLICATION.md`](REPLICATION.md) only.

---

## Helper scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_review_artifacts.sh` | Pre-built `data/` + `models/` from GitHub Release |
| `scripts/restart_moviemind.sh` | Start API + UI |
| `scripts/stop_moviemind.sh` | Stop API + UI |
| `scripts/verify_local_app.sh` | Check :8000 / :8502 |
| `scripts/setup_local_ollama.sh` | Ollama for NLP / tool agent |

Details: [`docs/ARTIFACTS_AND_RUNTIME.md`](docs/ARTIFACTS_AND_RUNTIME.md).

---

## More documentation

| Topic | File |
|-------|------|
| Setup & verification | [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) |
| Notebooks & training order | [`REPLICATION.md`](REPLICATION.md) |
| Tool agent | [`docs/AGENT.md`](docs/AGENT.md) |
| Ollama | [`docs/OLLAMA.md`](docs/OLLAMA.md) |
| UI ↔ API | [`docs/APP_AND_API.md`](docs/APP_AND_API.md) |
| All docs index | [`docs/README.md`](docs/README.md) |

---

## Project summary

- **Best offline model:** Gradient Boosting (RMSE ~0.897) — see `evidence/phase9_split_eval/`
- **API:** `/health`, `/models`, `/recommend`, `/nlp/query`, `/agent/query`
- **Optional:** Ollama on :11434 for Local LLM parse and multi-step tool agent
