import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Allow imports from the src/ directory when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from evaluation import evaluate_model

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# These are the numeric columns we will use as input features for ML models.
# They come from the feature engineering step (src/features.py).
FEATURE_COLS = [
    'user_rating_count',
    'user_avg_rating',
    'movie_rating_count',
    'movie_avg_rating',
    'age',
    'occupation',
    'release_year',
]

TARGET_COL = "rating"

# joblib + numpy on some Python 3.13 builds can fail when pickling large tree state
# (AttributeError: module 'pickle' has no attribute 'PickleBuffer' with protocol 5).
# Protocol 4 via stdlib pickle avoids that path; joblib.load still loads these files.
def save_sklearn_estimator(
    model,
    filename: str,
    *,
    models_dir: str | None = None,
) -> str:
    base = models_dir if models_dir is not None else MODELS_DIR
    path = os.path.join(base, filename)
    os.makedirs(base, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=4)
    return path


def load_processed_data():
    """Loads the train and test CSVs produced by features.py."""
    train = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_features.csv'))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, 'test_features.csv'))
    return train, test


def prepare_xy(df):
    """Extracts feature matrix X and target vector y from a DataFrame."""
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]
    return X, y


def train_baseline(train_df, test_df):
    """
    Baseline Model: Predicts the global mean rating for every movie.
    This is the simplest possible model. Any real model must beat this.
    """
    start_time = time.time()
    global_mean = train_df[TARGET_COL].mean()
    y_pred = np.full(len(test_df), global_mean)
    duration = time.time() - start_time
    
    res = evaluate_model(test_df[TARGET_COL], y_pred, model_name="Baseline (Global Mean)")
    res['time_sec'] = round(duration, 2)
    print(f"  Time: {res['time_sec']}s")
    return res


def train_linear_regression(X_train, y_train, X_test, y_test):
    """Linear Regression: Finds the best straight-line fit through the features."""
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    duration = time.time() - start_time

    save_sklearn_estimator(model, "linear_regression.pkl")

    res = evaluate_model(y_test, y_pred, model_name="Linear Regression")
    res['time_sec'] = round(duration, 2)
    print(f"  Time: {res['time_sec']}s")
    return res


def train_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest: An ensemble of decision trees that votes on the prediction."""
    start_time = time.time()
    
    # Hyperparameter Rationale:
    # n_estimators=100: The default. 100 trees provide a good balance between accuracy and training time.
    # max_depth=10: We explicitly restrict this from the default (None/unlimited). With 800K rows, unlimited depth would massively overfit and create a gigabyte-sized model.
    # n_jobs=-1: Use all available CPU cores on the M1 Mac to train trees in parallel.
    # random_state=42: Ensures reproducibility across runs.
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    duration = time.time() - start_time

    save_sklearn_estimator(model, "random_forest.pkl")

    res = evaluate_model(y_test, y_pred, model_name="Random Forest")
    res['time_sec'] = round(duration, 2)
    print(f"  Time: {res['time_sec']}s")
    return res


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Gradient Boosting: Builds trees sequentially, each one correcting the previous."""
    start_time = time.time()
    
    # Hyperparameter Rationale:
    # n_estimators=200: Increased from default (100). Since GB builds "weak" trees sequentially, more trees generally improve accuracy if learning rate is controlled.
    # max_depth=5: Increased from default (3). With 800K rows, a depth of 3 is too simple to capture complex user/movie interactions. Depth 5 allows for richer feature combinations while still preventing overfitting.
    # learning_rate=0.1: The default. It strikes a proven balance between converging steadily and not overshooting the minimum loss.
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    duration = time.time() - start_time

    save_sklearn_estimator(model, "gradient_boosting.pkl")

    res = evaluate_model(y_test, y_pred, model_name="Gradient Boosting")
    res['time_sec'] = round(duration, 2)
    print(f"  Time: {res['time_sec']}s")
    return res


def main():
    print("--- Phase 4: ML Model Training ---\n")

    train_df, test_df = load_processed_data()
    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    print(f"Training set: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(f"Test set:     {X_test.shape[0]:,} rows, {X_test.shape[1]} features\n")

    results = []

    # 1. Baseline
    results.append(train_baseline(train_df, test_df))

    # 2. Linear Regression
    results.append(train_linear_regression(X_train, y_train, X_test, y_test))

    # 3. Random Forest
    results.append(train_random_forest(X_train, y_train, X_test, y_test))

    # 4. Gradient Boosting
    results.append(train_gradient_boosting(X_train, y_train, X_test, y_test))

    # Summary table
    print("\n--- Model Comparison ---")
    summary = pd.DataFrame(results)
    table_str = summary.to_string(index=False)
    print(table_str)
    
    # Log to file
    with open(os.path.join(MODELS_DIR, 'ml_training_log.txt'), 'w') as f:
        f.write("--- Model Comparison ---\n")
        f.write(table_str + "\n")
    print(f"\nML results saved to {MODELS_DIR}/ml_training_log.txt")


if __name__ == "__main__":
    main()
