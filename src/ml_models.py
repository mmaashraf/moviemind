import pandas as pd
import numpy as np
import joblib
import os
import sys
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

TARGET_COL = 'rating'


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
    global_mean = train_df[TARGET_COL].mean()
    y_pred = np.full(len(test_df), global_mean)
    return evaluate_model(test_df[TARGET_COL], y_pred, model_name="Baseline (Global Mean)")


def train_linear_regression(X_train, y_train, X_test, y_test):
    """Linear Regression: Finds the best straight-line fit through the features."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, 'linear_regression.pkl'))

    return evaluate_model(y_test, y_pred, model_name="Linear Regression")


def train_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest: An ensemble of decision trees that votes on the prediction."""
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, 'random_forest.pkl'))

    return evaluate_model(y_test, y_pred, model_name="Random Forest")


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Gradient Boosting: Builds trees sequentially, each one correcting the previous."""
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, 'gradient_boosting.pkl'))

    return evaluate_model(y_test, y_pred, model_name="Gradient Boosting")


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
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
