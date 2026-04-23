import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    """Root Mean Squared Error - penalizes large errors more."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error - average magnitude of errors."""
    return mean_absolute_error(y_true, y_pred)


def precision_at_k(actual_relevant, predicted_list, k):
    """
    Precision@K: Of the top-K items we recommended, how many were actually relevant?

    Args:
        actual_relevant: set of movieIds the user actually liked (e.g., rated >= 4)
        predicted_list: ordered list of recommended movieIds (best first)
        k: number of top recommendations to consider
    """
    top_k = predicted_list[:k]
    relevant_in_top_k = len(set(top_k) & set(actual_relevant))
    return relevant_in_top_k / k if k > 0 else 0.0


def recall_at_k(actual_relevant, predicted_list, k):
    """
    Recall@K: Of all the items the user actually liked, how many did we find in top-K?

    Args:
        actual_relevant: set of movieIds the user actually liked
        predicted_list: ordered list of recommended movieIds
        k: number of top recommendations to consider
    """
    top_k = predicted_list[:k]
    relevant_in_top_k = len(set(top_k) & set(actual_relevant))
    return relevant_in_top_k / len(actual_relevant) if len(actual_relevant) > 0 else 0.0


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Prints a clean summary of RMSE and MAE for a model."""
    r = rmse(y_true, y_pred)
    m = mae(y_true, y_pred)
    print(f"--- {model_name} ---")
    print(f"  RMSE: {r:.4f}")
    print(f"  MAE:  {m:.4f}")
    return {"model": model_name, "rmse": r, "mae": m}
