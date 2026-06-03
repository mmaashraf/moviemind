# MovieMind replication guide

Single source for **which notebooks to run**, **which scripts in what order**, and how that maps to project phases and `evidence/`.

- **Run the app only (no training):** stop at [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) §4–6.
- **Fast artifacts (no local train):** [`docs/ARTIFACTS_AND_RUNTIME.md`](docs/ARTIFACTS_AND_RUNTIME.md) + [`REVIEWER_SETUP.md`](REVIEWER_SETUP.md) §2.4.

All commands assume you are in the **`moviemind/`** directory with venv active. `data/` and `models/` are **gitignored**.

---

## Tier 0 — App demo (minimum)

| Step | Command | Outputs |
|------|---------|---------|
| 1 | `python3 -m venv .venv && source .venv/bin/activate` | venv |
| 2 | `pip install -r requirements.txt` | deps |
| 3a | `python3 src/data_loader.py` | `data/ml-1m/*.dat` |
| 3b | *or* `bash scripts/download_review_artifacts.sh` | `data/` + `models/` |
| 4 | `python3 scripts/build_model_artifacts.py features` | `data/processed/*.csv` |
| 5 | `python3 scripts/build_model_artifacts.py ml` | `models/*.pkl` |
| 6 | `bash scripts/restart_moviemind.sh` | API :8000, UI :8502 |

**Smoke-only variant:** step 4 only, then UI model `baseline_global_mean` (see `REVIEWER_SETUP.md` §4.3 Path C).

---

## Tier 1 — Analysis notebooks (optional, safe)

These **do not** write to `data/processed/` or `models/`. Re-running them does **not** break the app.

| Order | Notebook | Purpose |
|-------|----------|---------|
| 1 | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Distributions, sparsity, schema checks |
| 2 | [`notebooks/02_long_tail_and_cold_start.ipynb`](notebooks/02_long_tail_and_cold_start.ipynb) | Long-tail and cold-start narrative |

Requires `data/ml-1m/` from `src/data_loader.py` (step 3a above).

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
