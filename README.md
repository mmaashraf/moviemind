# MovieMind

Movie recommendations on **MovieLens 1M**: ML/DL models, **FastAPI** API, **Streamlit** UI, optional **Ollama** agent.

**Local demo only** (`127.0.0.1`). See [`docs/SECURITY.md`](docs/SECURITY.md).

---

## Run locally (everyone starts here)

All commands assume you cloned into a folder named `moviemind` and use **`moviemind/` as the working directory**.

```bash
git clone https://github.com/mmaashraf/moviemind.git
cd moviemind
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export MOVIEMIND_ARTIFACTS_URL="https://github.com/mmaashraf/moviemind/releases/download/v1.0-artifacts/moviemind-artifacts.tar.gz"
bash scripts/download_review_artifacts.sh
```

This creates gitignored **`data/`** and **`models/`** (~160 MB) from the [GitHub release](https://github.com/mmaashraf/moviemind/releases/tag/v1.0-artifacts). Without them you cannot run the app or training notebooks.

**Verify install:** `python -m unittest discover -s tests -q`

---

## Pick one goal

| Goal | Time | Commands / file |
|------|------|-----------------|
| **Run the web app** | ~2 min after setup | `bash scripts/restart_moviemind.sh` → open http://127.0.0.1:8502 |
| **Run the capstone notebook** | ~5–15 min | `jupyter lab notebooks/MovieMind_capstone.ipynb` or `bash scripts/verify_capstone_notebook.sh` |
| **Watch demo videos** | instant | [`evidence/demo/video/`](evidence/demo/video/) — web app + capstone notebook |
| **Full re-train** | hours | [`REPLICATION.md`](REPLICATION.md) Tier 2 |

---

## Run the web app

```bash
source .venv/bin/activate
bash scripts/restart_moviemind.sh
bash scripts/verify_local_app.sh
```

| Service | URL |
|---------|-----|
| **UI** | http://127.0.0.1:8502 |
| **API docs** | http://127.0.0.1:8000/docs |

**Try:** Recommend → Manual → user **1161** → model **Gradient Boosting** → Get Recommendations.

**Stop:** `bash scripts/stop_moviemind.sh`

First API start can take up to ~90s while models load. Details: [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md).

---

## Run the capstone notebook (course submission)

Single notebook: download → EDA → preprocessing → ML/DL train → tuning → evaluation. **No web app code** in the notebook.

```bash
source .venv/bin/activate
jupyter lab notebooks/MovieMind_capstone.ipynb
```

Headless check (same fast path as CI smoke):

```bash
bash scripts/verify_capstone_notebook.sh
```

Default skips re-training when `data/processed/` and `models/` already exist. Full re-train: see [`REPLICATION.md`](REPLICATION.md) Tier 1.5.

Other EDA notebooks (optional): `notebooks/01_eda.ipynb`, `notebooks/02_long_tail_and_cold_start.ipynb`.

---

## Demo videos

Pre-recorded walkthroughs (Playwright → WebM). Full details: [`evidence/demo/README.md`](evidence/demo/README.md).

| Video | Path | Contents |
|-------|------|----------|
| **Web app** | [`evidence/demo/video/moviemind_demo.webm`](evidence/demo/video/moviemind_demo.webm) | Manual mode — all 6 models, diversity + taste map, NLP parse + multi-step tool agent, all UI tabs |
| **Capstone notebook** | [`evidence/demo/video/moviemind_capstone_notebook.webm`](evidence/demo/video/moviemind_capstone_notebook.webm) | JupyterLab — cell-by-cell execution and outputs (fast path) |

```bash
open evidence/demo/video/moviemind_demo.webm
open evidence/demo/video/moviemind_capstone_notebook.webm
```

**Re-record** (needs `pip install -r requirements-demo.txt`, `playwright install chromium`, artifacts + Ollama for agent segments):

```bash
bash scripts/record_demo_video.sh
bash scripts/record_notebook_video.sh
```

---

## Helper scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_review_artifacts.sh` | Pre-built `data/` + `models/` |
| `scripts/restart_moviemind.sh` | Start API :8000 + UI :8502 |
| `scripts/stop_moviemind.sh` | Stop API + UI |
| `scripts/verify_local_app.sh` | Health check for app |
| `scripts/verify_capstone_notebook.sh` | Execute capstone notebook (fast path) |
| `scripts/build_model_artifacts.py` | Full training pipeline |
| `scripts/record_demo_video.sh` | Web app demo → `evidence/demo/video/moviemind_demo.webm` |
| `scripts/record_notebook_video.sh` | Capstone notebook demo → `evidence/demo/video/moviemind_capstone_notebook.webm` |

---

## For reviewers using Cursor (or other AI assistants)

When helping someone run this repo locally:

1. **Working directory:** always `moviemind/` (project root containing `src/`, `app/`, `requirements.txt`).
2. **Virtual env:** create `.venv`, activate, `pip install -r requirements.txt`.
3. **Artifacts:** run `bash scripts/download_review_artifacts.sh` before app or capstone notebook (paths are gitignored).
4. **App:** `bash scripts/restart_moviemind.sh` then `bash scripts/verify_local_app.sh`.
5. **Capstone notebook:** `bash scripts/verify_capstone_notebook.sh` or open `notebooks/MovieMind_capstone.ipynb`.
6. **Do not commit** `data/`, `models/`, `.venv/`, or `evidence/runtime/`.

Extended checklists: [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md). Training order: [`REPLICATION.md`](REPLICATION.md).

---

## More documentation

| Topic | File |
|-------|------|
| Reviewer setup & verification | [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) |
| Notebooks & training order | [`REPLICATION.md`](REPLICATION.md) |
| Scripts & runtime | [`docs/ARTIFACTS_AND_RUNTIME.md`](docs/ARTIFACTS_AND_RUNTIME.md) |
| Tool agent | [`docs/AGENT.md`](docs/AGENT.md) |
| All docs | [`docs/README.md`](docs/README.md) |
| **Course report (PDF)** | [`docs/report/README.md`](docs/report/README.md) — `cd docs/report && make pdf` |

**Best offline model:** Gradient Boosting — test RMSE **0.8981** (`evidence/phase9_split_eval/`).

---

## Reference development environment

The project was built and smoke-tested on the machine below. **These are reference specs, not hard minimums** — reviewers can run the app and capstone notebook on other macOS/Linux/Windows setups with Python 3.10+ and the steps above. Minimum disk/RAM notes: [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md).

| | |
|---|---|
| **Hardware** | Apple M1 Pro (8 cores), 16 GB RAM, `arm64` |
| **OS** | macOS 26.5.1 |
| **Python** | 3.13.2 (`python3 -m venv .venv` in project root) |
| **Key packages** (from `requirements.txt`, Jun 2026) | PyTorch 2.12, scikit-learn 1.8, pandas 3.0, NumPy 2.4, FastAPI 0.137, Streamlit 1.58, Optuna 4.9, Jupyter 1.1 |
| **Report PDF** (optional) | Tectonic 0.16.9, Pandoc 3.6.4 — `cd docs/report && make pdf` |
| **DL backend** | Apple **MPS** when training NCF locally (falls back to CPU elsewhere) |

**Rough timings on this machine (pre-built artifacts, fast paths):**

| Task | Time |
|------|------|
| `pip install -r requirements.txt` | several minutes (PyTorch) |
| `download_review_artifacts.sh` | ~1–3 min (network) |
| `restart_moviemind.sh` + first API load | up to ~90 s |
| Capstone notebook (`verify_capstone_notebook.sh`, skip re-train) | ~5–15 min |
| Full re-train ([`REPLICATION.md`](REPLICATION.md) Tier 2) | hours |

If an assistant is reproducing the repo for someone else, use the **Cursor** checklist above first; treat this table as expected environment and timing context, not a blocker.
