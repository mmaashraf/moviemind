# MovieMind replication guide

**Start here if you haven't set up yet:** [`README.md`](README.md) (clone, venv, artifact download).

This file is the **notebook and training order** reference. All commands assume **`moviemind/`** as cwd with venv active. `data/` and `models/` are **gitignored**.

- **Run the app:** [`README.md`](README.md) → `restart_moviemind.sh`
- **Capstone notebook:** Tier 1.5 below
- **EDA only:** Tier 1
- **Full re-train:** Tier 2

---

## Tier 0 — App demo (minimum)

| Step | Command | Outputs |
|------|---------|---------|
| 1 | `python3 -m venv .venv && source .venv/bin/activate` | venv |
| 2 | `pip install -r requirements.txt` | deps |
| 3 | `bash scripts/download_review_artifacts.sh` | `data/` + `models/` (reviewer fast path — see `REVIEWER_SETUP.md` §2.4) |
| 3alt | `python3 src/data_loader.py` then build steps 4–5 | local train path |
| 4 | `python3 scripts/build_model_artifacts.py features` | `data/processed/*.csv` (skip if downloaded) |
| 5 | `python3 scripts/build_model_artifacts.py ml` | `models/*.pkl` (skip if downloaded) |
| 6 | `bash scripts/restart_moviemind.sh` | API :8000, UI :8502 |

**Smoke-only variant:** step 4 only, then UI model `baseline_global_mean` (see `REVIEWER_SETUP.md` §4.3 Path C).

---

## Tier 1 — Analysis notebooks (optional, safe)

These **do not** write to `data/processed/` or `models/`. Re-running them does **not** break the app.

| Order | Notebook | Purpose |
|-------|----------|---------|
| 1 | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Distributions, sparsity, schema checks |
| 2 | [`notebooks/02_long_tail_and_cold_start.ipynb`](notebooks/02_long_tail_and_cold_start.ipynb) | Long-tail and cold-start narrative |

**Data needed:** `data/ml-1m/` — included in the [release tarball](https://github.com/mmaashraf/moviemind/releases/tag/v1.0-artifacts) (`REVIEWER_SETUP.md` §2.4), or from `python3 src/data_loader.py`.

```bash
source .venv/bin/activate
jupyter lab notebooks/01_eda.ipynb
jupyter lab notebooks/02_long_tail_and_cold_start.ipynb
```

Use the **`.venv` Python 3** kernel. Safe while the app is running (§2.5 in `REVIEWER_SETUP.md`).

---

## Tier 1.5 — Capstone submission notebook

**Single notebook** for course submission: data load → EDA → features → ML/DL train → tuning → evaluation. Orchestrates `src/*.py` modules.

| Notebook | Purpose |
|----------|---------|
| [`notebooks/MovieMind_capstone.ipynb`](notebooks/MovieMind_capstone.ipynb) | Full pipeline + results (default: skip existing artifacts) |

```bash
source .venv/bin/activate
# Interactive:
jupyter lab notebooks/MovieMind_capstone.ipynb

# Headless verify (fast path):
bash scripts/verify_capstone_notebook.sh

# Full re-train (hours):
MOVIEMIND_RUN_FULL=1 MOVIEMIND_SKIP_TUNE_DL=0 jupyter nbconvert --execute notebooks/MovieMind_capstone.ipynb
```

Regenerate notebook cells from script: `python3 scripts/build_capstone_notebook.py`

**Data needed:** `data/ml-1m/` and (for training sections) `data/processed/` + `models/` — from tarball or Tier 2 pipeline.

---

## Tier 2 — Full training pipeline (reproduce capstone artifacts)

### One-shot (recommended)

```bash
python3 src/data_loader.py
python3 scripts/build_model_artifacts.py all --skip-tune-dl
```

`--skip-tune-dl` skips Optuna when `models/best_dl_params.txt` already exists.  
Help: `python3 scripts/build_model_artifacts.py --help`

### Phase-by-phase (same order as development)

| Phase | Script / module | Command | Key outputs | Evidence folder |
|-------|-----------------|---------|-------------|-----------------|
| 1 | Data | `python3 src/data_loader.py` | `data/ml-1m/*.dat` | — |
| 2 | EDA | Tier 1 notebooks (optional) | plots in notebook | — |
| 3 | Features | `python3 src/features.py` or `build_model_artifacts.py features` | `data/processed/train_features.csv`, `test_features.csv`, `val_features.csv` | — |
| 4 | ML | `python3 src/ml_models.py` or `build_model_artifacts.py ml` | `models/*.pkl`, `models/ml_training_log.txt` | — |
| 5 | DL | `python3 src/dl_model.py` or `build_model_artifacts.py dl` | `models/ncf_model.pt` | — |
| 5b | Tune | `python3 src/tune_dl.py` or `build_model_artifacts.py tune-dl` | `models/best_dl_params.txt` | `evidence/phase5b/` |
| 6 | Post | `python3 src/post_analysis.py` or `build_model_artifacts.py post` | `models/ncf_tuned_best.pt`, plots/CSVs | `evidence/phase6/` |

Optional t-SNE (slow):

```bash
MOVIEMIND_ENABLE_TSNE=1 python3 src/post_analysis.py
```

**Tests after ML train:**

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

---

## Tier 3 — API, UI, and Ollama (Phase 7–8x)

| Step | Command | Notes |
|------|---------|--------|
| API + UI | `bash scripts/restart_moviemind.sh` | Or separate `uvicorn` / `streamlit` (see `REVIEWER_SETUP.md` §6) |
| Ollama setup | `bash scripts/setup_local_ollama.sh` | Only for Local LLM + tool agent |
| NLP smoke | `bash scripts/test_local_llm.sh` | Needs API running; writes `evidence/phase8/` |

Verify with [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) §7–9.

---

## `build_model_artifacts.py` commands

| Command | Runs |
|---------|------|
| `features` | Phase 3 — processed CSVs |
| `ml` | Phase 4 — sklearn `.pkl` |
| `dl` | Phase 5 — `ncf_model.pt` |
| `tune-dl` | Phase 5b — Optuna → `best_dl_params.txt` |
| `post` | Phase 6 — tuned NCF + `evidence/phase6/` |
| `all` | All of the above in order |

---

## What overwrites app artifacts

| Action | Effect |
|--------|--------|
| Re-run `build_model_artifacts.py` or `features` / `ml_models.py` | Replaces `data/processed/` and `models/`; restart API |
| Re-run Tier 1 notebooks | No effect on app paths |
| Different sklearn version than training | Possible pickle load errors — match `requirements.txt` |

---

## Proof already in git

You do not need to re-run training to **review** results:

- `evidence/phase5b/` — DL tuning notes  
- `evidence/phase6/` — post-analysis, embeddings, plots  
- `evidence/phase7/` — API smoke  
- `evidence/phase8/` — UI/NLP smoke, screenshots  
- `evidence/phase9_split_eval/` — 70/10/20 eval summaries  

See [`evidence/README.md`](evidence/README.md).

---

## Author / maintenance docs

Not required for reviewers: [`docs/internal/`](docs/internal/) (progress, handoff, learning wikis).
