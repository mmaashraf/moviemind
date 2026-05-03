import json
import os
import time
from itertools import product
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from evaluation import mae, rmse
from ml_models import FEATURE_COLS, TARGET_COL, PROCESSED_DIR

EVIDENCE_DIR = os.path.join("evidence", "phase9_split_eval")
MODELS_DIR = "models"
TUNE_SAMPLE_ROWS = 200_000


def _load_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features.csv"), low_memory=False)
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val_features.csv"), low_memory=False)
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test_features.csv"), low_memory=False)
    return train, val, test


def _sample_for_fast_tuning(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).sort_values("timestamp")


def _xy(df: pd.DataFrame):
    x = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]
    return x, y


def _eval(name: str, y_true, y_pred) -> Dict[str, float]:
    return {"model": name, "rmse": float(rmse(y_true, y_pred)), "mae": float(mae(y_true, y_pred))}


def _run_default_models(x_train, y_train, x_val, y_val, x_test, y_test) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    # Baseline global mean
    base_pred_val = [float(y_train.mean())] * len(y_val)
    base_pred_test = [float(y_train.mean())] * len(y_test)
    rows.append({**_eval("Baseline (Global Mean) - VAL", y_val, base_pred_val), "phase": "raw"})
    rows.append({**_eval("Baseline (Global Mean) - TEST", y_test, base_pred_test), "phase": "raw"})

    defaults = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        ),
    ]
    for name, model in defaults:
        start = time.time()
        model.fit(x_train, y_train)
        fit_sec = time.time() - start
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
        rows.append({**_eval(f"{name} - VAL", y_val, val_pred), "phase": "raw", "fit_sec": round(fit_sec, 2)})
        rows.append({**_eval(f"{name} - TEST", y_test, test_pred), "phase": "raw", "fit_sec": round(fit_sec, 2)})
    return rows


def _grid_params_rf() -> List[Dict[str, int]]:
    grid = {
        "n_estimators": [80, 120],
        "max_depth": [10, 14],
        "min_samples_leaf": [1],
    }
    keys = list(grid.keys())
    combos = []
    for vals in product(*[grid[k] for k in keys]):
        combo = {k: v for k, v in zip(keys, vals)}
        combo["random_state"] = 42
        combo["n_jobs"] = 4
        combos.append(combo)
    return combos


def _grid_params_gb() -> List[Dict[str, float]]:
    grid = {
        "n_estimators": [120, 180],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4],
    }
    keys = list(grid.keys())
    combos = []
    for vals in product(*[grid[k] for k in keys]):
        combo = {k: v for k, v in zip(keys, vals)}
        combo["random_state"] = 42
        combos.append(combo)
    return combos


def _tune_family(
    family_name: str,
    param_grid: List[Dict],
    builder,
    x_train,
    y_train,
    x_val,
    y_val,
) -> Tuple[Dict, float]:
    best_params = None
    best_rmse = float("inf")
    for idx, params in enumerate(param_grid, start=1):
        model = builder(**params)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        score = float(rmse(y_val, pred))
        if score < best_rmse:
            best_rmse = score
            best_params = params
        print(f"[{family_name}] trial {idx}/{len(param_grid)} rmse={score:.4f} params={params}", flush=True)
    return best_params, best_rmse


def main() -> None:
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df, val_df, test_df = _load_split_data()
    tune_train_df = _sample_for_fast_tuning(train_df, TUNE_SAMPLE_ROWS, seed=42)
    tune_val_df = _sample_for_fast_tuning(val_df, min(TUNE_SAMPLE_ROWS // 2, len(val_df)), seed=42)

    x_train, y_train = _xy(train_df)
    x_val, y_val = _xy(val_df)
    x_test, y_test = _xy(test_df)
    x_tune_train, y_tune_train = _xy(tune_train_df)
    x_tune_val, y_tune_val = _xy(tune_val_df)

    log_lines = []
    log_lines.append("=== ML tuning on train/val/test (70/10/20) ===")
    log_lines.append(f"train_rows={len(train_df)} val_rows={len(val_df)} test_rows={len(test_df)}")
    log_lines.append(
        f"fast_tuning_sample_rows train={len(tune_train_df)} val={len(tune_val_df)} "
        f"(for speed; final tuned test eval remains on full test split)"
    )

    # Raw/default metrics
    default_rows = _run_default_models(x_train, y_train, x_val, y_val, x_test, y_test)
    default_df = pd.DataFrame(default_rows)
    default_csv = os.path.join(EVIDENCE_DIR, "ml_default_val_test_metrics_70_10_20.csv")
    default_df.to_csv(default_csv, index=False)
    log_lines.append(f"saved_default_metrics={default_csv}")

    # Tune Random Forest
    rf_grid = _grid_params_rf()
    best_rf_params, best_rf_val_rmse = _tune_family(
        "RandomForest",
        rf_grid,
        RandomForestRegressor,
        x_tune_train,
        y_tune_train,
        x_tune_val,
        y_tune_val,
    )
    rf_model = RandomForestRegressor(**best_rf_params)
    start_rf = time.time()
    rf_model.fit(x_train, y_train)
    rf_fit_sec = time.time() - start_rf
    rf_val = _eval("Random Forest Tuned - VAL", y_val, rf_model.predict(x_val))
    rf_test = _eval("Random Forest Tuned - TEST", y_test, rf_model.predict(x_test))
    rf_val["fit_sec"] = round(rf_fit_sec, 2)
    rf_test["fit_sec"] = round(rf_fit_sec, 2)

    # Tune Gradient Boosting
    gb_grid = _grid_params_gb()
    best_gb_params, best_gb_val_rmse = _tune_family(
        "GradientBoosting",
        gb_grid,
        GradientBoostingRegressor,
        x_tune_train,
        y_tune_train,
        x_tune_val,
        y_tune_val,
    )
    gb_model = GradientBoostingRegressor(**best_gb_params)
    start_gb = time.time()
    gb_model.fit(x_train, y_train)
    gb_fit_sec = time.time() - start_gb
    gb_val = _eval("Gradient Boosting Tuned - VAL", y_val, gb_model.predict(x_val))
    gb_test = _eval("Gradient Boosting Tuned - TEST", y_test, gb_model.predict(x_test))
    gb_val["fit_sec"] = round(gb_fit_sec, 2)
    gb_test["fit_sec"] = round(gb_fit_sec, 2)

    tuned_df = pd.DataFrame([rf_val, rf_test, gb_val, gb_test])
    tuned_csv = os.path.join(EVIDENCE_DIR, "ml_tuned_val_test_metrics_70_10_20.csv")
    tuned_df.to_csv(tuned_csv, index=False)

    best_params = {
        "random_forest": {"best_val_rmse": best_rf_val_rmse, "params": best_rf_params},
        "gradient_boosting": {"best_val_rmse": best_gb_val_rmse, "params": best_gb_params},
    }
    best_params_path = os.path.join(MODELS_DIR, "best_ml_params_70_10_20.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    log_lines.append(f"saved_tuned_metrics={tuned_csv}")
    log_lines.append(f"saved_best_params={best_params_path}")
    log_lines.append(f"rf_best_val_rmse={best_rf_val_rmse:.4f}")
    log_lines.append(f"gb_best_val_rmse={best_gb_val_rmse:.4f}")

    log_path = os.path.join(EVIDENCE_DIR, "ml_tuning_run_70_10_20.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print("\n".join(log_lines))
    print("\n--- Tuned model metrics ---")
    print(tuned_df.to_string(index=False))


if __name__ == "__main__":
    main()
