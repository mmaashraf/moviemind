#!/usr/bin/env python3
"""Generate notebooks/MovieMind_capstone.ipynb (idempotent)."""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "MovieMind_capstone.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "id": uuid.uuid4().hex[:8], "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "id": uuid.uuid4().hex[:8],
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        """# MovieMind Capstone Notebook

Single submission notebook for **data loading, exploration, preprocessing, model training, tuning, and evaluation**.

Implementation lives in `src/*.py`; this notebook orchestrates those modules with commentary and result tables.

**Runtime modes (Section 0):**
- Default (`RUN_FULL_PIPELINE=False`): skip phases when outputs already exist (~5–15 min).
- Full reproduce (`RUN_FULL_PIPELINE=True`): re-run training; add ~30–90 min without Optuna, ~2 h with Optuna.
"""
    ),
    md("## Section 0 — Setup, paths, and configuration"),
    code(
        '''import os
import sys
from pathlib import Path

# --- Resolve repo root (works from notebooks/ or project root) ---
_cwd = Path.cwd().resolve()
if (_cwd / "src" / "features.py").is_file():
    ROOT = _cwd
elif (_cwd.parent / "src" / "features.py").is_file():
    ROOT = _cwd.parent
else:
    raise FileNotFoundError(
        "Run this notebook from moviemind/ or moviemind/notebooks/. "
        "Could not find src/features.py."
    )
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# --- Execution flags (change for full re-train) ---
RUN_FULL_PIPELINE = os.environ.get("MOVIEMIND_RUN_FULL", "0").strip() == "1"
SKIP_TUNE_DL = os.environ.get("MOVIEMIND_SKIP_TUNE_DL", "1").strip() == "1"
SKIP_TUNE_ML = os.environ.get("MOVIEMIND_SKIP_TUNE_ML", "0").strip() == "1"
SKIP_POST = os.environ.get("MOVIEMIND_SKIP_POST", "0").strip() == "1"

DATA_DIR = ROOT / "data" / "ml-1m"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
EVIDENCE_DIR = ROOT / "evidence" / "phase9_split_eval"

print(f"Repo root: {ROOT}")
print(f"RUN_FULL_PIPELINE={RUN_FULL_PIPELINE}")
print(f"SKIP_TUNE_DL={SKIP_TUNE_DL}  SKIP_TUNE_ML={SKIP_TUNE_ML}  SKIP_POST={SKIP_POST}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (10, 6)
%matplotlib inline


def artifact_exists(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(path.iterdir()))


def should_run(label: str, output_path: Path, *, force: bool = False) -> bool:
    """Skip expensive phases when outputs exist unless RUN_FULL_PIPELINE or force."""
    if force or RUN_FULL_PIPELINE or not artifact_exists(output_path):
        print(f"[run] {label}")
        return True
    print(f"[skip] {label} — found {output_path}")
    return False
'''
    ),
    md("## Section 1 — Data loading"),
    code(
        '''from data_loader import main as download_data

extract_dir = download_data(data_dir=str(ROOT / "data"))
assert extract_dir and Path(extract_dir).is_dir(), (
    "MovieLens 1M missing. Run: python3 src/data_loader.py "
    "or bash scripts/download_review_artifacts.sh"
)

# Schema sanity check (optional helper)
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("schema_check", ROOT / "notebooks" / "01_schema_check.py")
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_mod.check_1m_schema(data_dir=str(DATA_DIR))
'''
    ),
    md(
        """## Section 2 — Exploratory data analysis

Insights here inform feature engineering: rating skew, sparsity, demographics, long tail, and cold start.
"""
    ),
    code(
        '''ratings_cols = ["userId", "movieId", "rating", "timestamp"]
movies_cols = ["movieId", "title", "genres"]
users_cols = ["userId", "gender", "age", "occupation", "zipCode"]

ratings = pd.read_csv(DATA_DIR / "ratings.dat", sep="::", engine="python", names=ratings_cols, encoding="latin-1")
movies = pd.read_csv(DATA_DIR / "movies.dat", sep="::", engine="python", names=movies_cols, encoding="latin-1")
users = pd.read_csv(DATA_DIR / "users.dat", sep="::", engine="python", names=users_cols, encoding="latin-1")

print(f"Ratings: {ratings.shape[0]:,}  Movies: {movies.shape[0]:,}  Users: {users.shape[0]:,}")

# Rating distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=ratings, x="rating", hue="rating", palette="viridis", legend=False)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating (stars)")
plt.ylabel("Count")
plt.show()
print("Mean rating:", round(ratings["rating"].mean(), 3), " Median:", ratings["rating"].median())

# Demographics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=users, x="gender", hue="gender", palette="Set2", ax=axes[0], legend=False)
axes[0].set_title("User gender")
sns.countplot(data=users, x="age", hue="age", palette="magma", ax=axes[1], legend=False)
axes[1].set_title("User age (binned codes)")
plt.tight_layout()
plt.show()

# Sparsity
total_possible = users.shape[0] * movies.shape[0]
actual = ratings.shape[0]
sparsity = 1 - (actual / total_possible)
print(f"Matrix sparsity: {sparsity * 100:.2f}%")
'''
    ),
    code(
        '''# Long tail + cold start (from notebooks/02_long_tail_and_cold_start.ipynb)
movie_popularity = ratings.groupby("movieId").size().reset_index(name="num_ratings")
movie_popularity = movie_popularity.sort_values("num_ratings", ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(movie_popularity.index, movie_popularity["num_ratings"], color="indigo", linewidth=2)
plt.fill_between(movie_popularity.index, movie_popularity["num_ratings"], color="indigo", alpha=0.3)
plt.axhline(y=20, color="r", linestyle="--", label="Cold-start threshold (<20 ratings)")
plt.title("Long tail of movie popularity")
plt.xlabel("Movies (ranked)")
plt.ylabel("Number of ratings")
plt.legend()
plt.show()

cold_movies = (movie_popularity["num_ratings"] < 20).sum()
print(f"Cold-start movies (<20 ratings): {cold_movies:,} / {len(movie_popularity):,}")

user_activity = ratings.groupby("userId").size().reset_index(name="num_ratings")
user_activity = user_activity.sort_values("num_ratings", ascending=False).reset_index(drop=True)
casual = (user_activity["num_ratings"] < 50).sum()
print(f"Casual users (<50 ratings): {casual:,} / {len(user_activity):,}")

top_movies = pd.merge(movie_popularity.head(10), movies, on="movieId")
plt.figure(figsize=(10, 6))
sns.barplot(data=top_movies, x="num_ratings", y="title", hue="title", palette="rocket", legend=False)
plt.title("Top 10 most rated movies")
plt.tight_layout()
plt.show()
'''
    ),
    md(
        """## Section 3 — Feature engineering (preprocessing)

Chronological **70/10/20** train/val/test split on `timestamp` (leakage-safe). Features include user/movie aggregates, demographics, and release year.
"""
    ),
    code(
        '''train_csv = PROCESSED_DIR / "train_features.csv"
if should_run("Feature engineering", train_csv):
    from features import main as run_features
    run_features()

for name in ("train_features.csv", "val_features.csv", "test_features.csv"):
    p = PROCESSED_DIR / name
    df = pd.read_csv(p, nrows=3)
    print(f"{name}: exists, sample columns = {list(df.columns[:8])} ...")

from ml_models import FEATURE_COLS
print("ML feature columns:", FEATURE_COLS)
'''
    ),
    md(
        """## Section 4 — ML model training

Baseline global mean, Linear Regression, Random Forest, and Gradient Boosting on tabular features.
"""
    ),
    code(
        '''gb_pkl = MODELS_DIR / "gradient_boosting.pkl"
if should_run("ML training", gb_pkl):
    from ml_models import main as run_ml_models
    run_ml_models()

log_path = MODELS_DIR / "ml_training_log.txt"
if log_path.is_file():
    print(log_path.read_text())
else:
    print("No ml_training_log.txt yet.")
'''
    ),
    md(
        """## Section 5 — ML hyperparameter tuning

Grid search on Random Forest and Gradient Boosting (fast-pass subsample for search; final refit on full train). Not part of `build_model_artifacts.py` — called explicitly here.
"""
    ),
    code(
        '''tuned_csv = EVIDENCE_DIR / "ml_tuned_val_test_metrics_70_10_20.csv"
if SKIP_TUNE_ML and tuned_csv.is_file() and not RUN_FULL_PIPELINE:
    print(f"[skip] ML tuning — found {tuned_csv}")
elif should_run("ML tuning", tuned_csv):
    from tune_ml import main as run_tune_ml
    run_tune_ml()

if tuned_csv.is_file():
    from IPython.display import display
    display(pd.read_csv(tuned_csv))
'''
    ),
    md("## Section 6 — Deep learning (NCF baseline)"),
    code(
        '''ncf_pt = MODELS_DIR / "ncf_model.pt"
if should_run("DL baseline NCF", ncf_pt):
    from dl_model import main as run_dl_model
    run_dl_model()

params_file = MODELS_DIR / "best_dl_params.txt"
print("best_dl_params.txt exists:", params_file.is_file())
if params_file.is_file():
    print(params_file.read_text()[:500])
'''
    ),
    md(
        """## Section 7 — DL hyperparameter tuning (Optuna)

**Slow:** ~16 min for 50 trials. Default skips when `models/best_dl_params.txt` exists (`SKIP_TUNE_DL=True`).
"""
    ),
    code(
        '''params_file = MODELS_DIR / "best_dl_params.txt"
if SKIP_TUNE_DL and params_file.is_file() and not RUN_FULL_PIPELINE:
    print(f"[skip] Optuna tune_dl — using {params_file}")
    print(params_file.read_text())
elif should_run("Optuna DL tuning", params_file):
    from tune_dl import main as run_tune_dl
    run_tune_dl()
else:
    print("No best_dl_params.txt; set SKIP_TUNE_DL=False or download review artifacts.")
'''
    ),
    md("## Section 8 — Post-analysis (tuned NCF + XAI artifacts)"),
    code(
        '''tuned_ncf = MODELS_DIR / "ncf_tuned_best.pt"
if SKIP_POST and tuned_ncf.is_file() and not RUN_FULL_PIPELINE:
    print(f"[skip] post_analysis — found {tuned_ncf}")
elif should_run("Post-analysis", tuned_ncf):
    from post_analysis import main as run_post_analysis
    run_post_analysis()

phase6 = ROOT / "evidence" / "phase6"
if (phase6 / "post_analysis_summary.txt").is_file():
    print((phase6 / "post_analysis_summary.txt").read_text()[:800])
'''
    ),
    md("## Section 9 — Evaluation summary"),
    code(
        '''from evaluation import rmse, mae, evaluate_model

master_md = EVIDENCE_DIR / "evaluation_master_summary_70_10_20.md"
if master_md.is_file():
    from IPython.display import Markdown, display
    display(Markdown(master_md.read_text()))
else:
    print("Run tune_ml / training to generate phase9 evidence.")

for csv_name in (
    "ml_default_val_test_metrics_70_10_20.csv",
    "ml_tuned_val_test_metrics_70_10_20.csv",
):
    p = EVIDENCE_DIR / csv_name
    if p.is_file():
        print(f"\\n=== {csv_name} ===")
        display(pd.read_csv(p))

print("\\nChampion (documented): Gradient Boosting (raw) — test RMSE 0.8981, MAE 0.7062")
'''
    ),
    md(
        """## Section 10 — Conclusion (notebook)

- **Best tabular model:** raw Gradient Boosting on engineered features (RMSE ~0.90 on chronological test split).
- **DL:** NCF + Optuna improves over raw NCF but still trails GB on RMSE for this rating-regression setup.
- **Key design choices:** time-based split, popularity/demographic features for cold-start mitigation, hybrid ML + embedding approach.
- **Future work:** ranking metrics at scale, two-tower retrieval, and production agent/API layer (see app docs).
"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
