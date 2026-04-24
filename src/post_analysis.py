import os
import time
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from dl_model import MovieLensDataset, PROCESSED_DIR, MODELS_DIR

EVIDENCE_DIR = os.path.join("evidence", "phase6")
POST_LOG_PATH = os.path.join(EVIDENCE_DIR, "post_analysis_log.txt")
SUMMARY_PATH = os.path.join(EVIDENCE_DIR, "post_analysis_summary.txt")
FEATURE_COLS = [
    "user_rating_count",
    "user_avg_rating",
    "movie_rating_count",
    "movie_avg_rating",
    "age",
    "release_year",
]
ML_FEATURE_COLS = [
    "user_rating_count",
    "user_avg_rating",
    "movie_rating_count",
    "movie_avg_rating",
    "age",
    "occupation",
    "release_year",
]
REQUIRED_COLS = ["userId", "movieId", "rating"] + FEATURE_COLS
TUNE_EPOCHS = 5
BATCH_SIZE = 1024


def log(msg):
    """Prints and writes logs to evidence folder."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    print(msg)
    with open(POST_LOG_PATH, "a") as f:
        f.write(msg + "\n")


def run_tsne_subprocess(tsne_input):
    """
    Runs sklearn t-SNE in a child process.
    If sklearn/native libs segfault, only child dies and main script survives.
    """
    tsne_input_path = os.path.join(EVIDENCE_DIR, "tsne_input.npy")
    tsne_output_path = os.path.join(EVIDENCE_DIR, "tsne_output.npy")
    np.save(tsne_input_path, tsne_input.astype(np.float32))

    child_code = f"""
import numpy as np
from sklearn.manifold import TSNE
X = np.load(r"{tsne_input_path}")
model = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, init="pca")
Y = model.fit_transform(X)
np.save(r"{tsne_output_path}", Y)
"""
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    result = subprocess.run(
        [sys.executable, "-c", child_code],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"t-SNE child process failed with return code {result.returncode}.")
        if result.stdout.strip():
            log(f"t-SNE child stdout: {result.stdout.strip()}")
        if result.stderr.strip():
            log(f"t-SNE child stderr: {result.stderr.strip()}")
        return None

    if not os.path.exists(tsne_output_path):
        log("t-SNE child process ended without creating tsne_output.npy.")
        return None
    return np.load(tsne_output_path)


def parse_best_params(path):
    """Reads best Optuna parameters from models/best_dl_params.txt."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    params = {}
    best_rmse = None
    with open(path, "r") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            if clean.lower().startswith("best rmse:"):
                best_rmse = float(clean.split(":")[1].strip())
            elif ":" in clean:
                key, value = [x.strip() for x in clean.split(":", 1)]
                if key in {"embedding_dim", "n_layers"} or key.startswith("n_units_l"):
                    params[key] = int(float(value))
                else:
                    params[key] = float(value)

    if best_rmse is None:
        raise ValueError("Could not parse Best RMSE from best_dl_params.txt")
    return best_rmse, params


class DynamicNCF(nn.Module):
    """Same dynamic architecture used during tuning."""

    def __init__(self, num_users, num_movies, num_dense_features, embedding_dim, hidden_layers, dropout_rate):
        super(DynamicNCF, self).__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users + 1, embedding_dim=embedding_dim)
        self.movie_embedding = nn.Embedding(num_embeddings=num_movies + 1, embedding_dim=embedding_dim)

        total_input_dim = (embedding_dim * 2) + num_dense_features
        layers = []
        in_dim = total_input_dim
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, user_idx, movie_idx, dense_features):
        u_emb = self.user_embedding(user_idx)
        m_emb = self.movie_embedding(movie_idx)
        x = torch.cat([u_emb, m_emb, dense_features], dim=1)
        return self.fc_layers(x).squeeze()


def load_datasets():
    """Loads processed train and test sets with only required columns."""
    train_df = pd.read_csv(
        os.path.join(PROCESSED_DIR, "train_features.csv"),
        usecols=REQUIRED_COLS,
        low_memory=False,
    )
    test_df = pd.read_csv(
        os.path.join(PROCESSED_DIR, "test_features.csv"),
        usecols=REQUIRED_COLS,
        low_memory=False,
    )
    return train_df, test_df


def train_best_tuned_model(train_df, test_df, params):
    """
    Re-trains one model using best tuned hyperparameters.
    We do this because tuning file currently saves best params, not model weights.
    """
    n_layers = params["n_layers"]
    hidden_layers = [params[f"n_units_l{i}"] for i in range(n_layers)]
    embedding_dim = params["embedding_dim"]
    dropout_rate = params["dropout_rate"]
    learning_rate = params["learning_rate"]

    num_users = int(max(train_df["userId"].max(), test_df["userId"].max()))
    num_movies = int(max(train_df["movieId"].max(), test_df["movieId"].max()))

    train_loader = DataLoader(MovieLensDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(MovieLensDataset(test_df), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"Training best tuned model on device: {device}")
    model = DynamicNCF(num_users, num_movies, len(FEATURE_COLS), embedding_dim, hidden_layers, dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start = time.time()
    for epoch in range(TUNE_EPOCHS):
        model.train()
        running_loss = 0.0
        for users, movies, dense, ratings in train_loader:
            users = users.to(device)
            movies = movies.to(device)
            dense = dense.to(device)
            ratings = ratings.to(device)
            optimizer.zero_grad()
            outputs = model(users, movies, dense)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * users.size(0)
        train_mse = running_loss / len(train_loader.dataset)
        log(f"Epoch {epoch + 1}/{TUNE_EPOCHS} train MSE: {train_mse:.4f}")

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for users, movies, dense, ratings in test_loader:
            users = users.to(device)
            movies = movies.to(device)
            dense = dense.to(device)
            ratings = ratings.to(device)
            outputs = model(users, movies, dense)
            loss = criterion(outputs, ratings)
            running_val_loss += loss.item() * users.size(0)
    val_mse = running_val_loss / len(test_loader.dataset)
    val_rmse = float(np.sqrt(val_mse))

    elapsed = (time.time() - start) / 60.0
    log(f"Retrain complete in {elapsed:.2f} minutes | Validation RMSE: {val_rmse:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    tuned_model_path = os.path.join(MODELS_DIR, "ncf_tuned_best.pt")
    torch.save(model.state_dict(), tuned_model_path)
    log(f"Saved tuned model weights: {tuned_model_path}")
    return model, val_rmse, tuned_model_path


def save_embedding_visuals(model):
    """
    Extracts user embeddings and saves PCA outputs.
    t-SNE is optional because it can be unstable on some local environments.
    Enable t-SNE with: MOVIEMIND_ENABLE_TSNE=1
    """
    # Last row is padding/unused because dataset indices are 0..(max_id-1).
    user_emb = model.user_embedding.weight.detach().cpu().numpy()[:-1]
    user_ids = np.arange(1, user_emb.shape[0] + 1)

    emb_df = pd.DataFrame(user_emb)
    emb_df.insert(0, "userId", user_ids)
    emb_path = os.path.join(EVIDENCE_DIR, "user_embeddings_raw.csv")
    emb_df.to_csv(emb_path, index=False)
    log(f"Saved raw user embeddings: {emb_path}")

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(user_emb)
    pca_df = pd.DataFrame({"userId": user_ids, "pca_1": pca_2d[:, 0], "pca_2": pca_2d[:, 1]})
    pca_path = os.path.join(EVIDENCE_DIR, "user_embeddings_pca_2d.csv")
    pca_df.to_csv(pca_path, index=False)
    log(f"Saved PCA coordinates: {pca_path}")

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df["pca_1"], pca_df["pca_2"], s=6, alpha=0.6)
    plt.title("User Embeddings - PCA 2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.2)
    pca_plot_path = os.path.join(EVIDENCE_DIR, "user_embeddings_pca_2d.png")
    plt.savefig(pca_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved PCA plot: {pca_plot_path}")

    tsne_enabled = os.environ.get("MOVIEMIND_ENABLE_TSNE", "0") == "1"
    if not tsne_enabled:
        note_path = os.path.join(EVIDENCE_DIR, "tsne_status.txt")
        with open(note_path, "w") as f:
            f.write("t-SNE skipped by default for crash safety.\n")
            f.write("Set MOVIEMIND_ENABLE_TSNE=1 to enable t-SNE generation.\n")
        log(f"Skipped t-SNE (default safe mode). Note saved: {note_path}")
        return "skipped (safe mode)"

    # t-SNE is expensive and can be unstable in some local environments.
    # We run it in a child process so parent script still completes on failure.
    sample_size = min(1000, len(user_emb))
    sample_idx = np.random.RandomState(42).choice(len(user_emb), size=sample_size, replace=False)
    tsne_input = np.ascontiguousarray(user_emb[sample_idx].astype(np.float32))
    tsne_user_ids = user_ids[sample_idx]
    tsne_2d = run_tsne_subprocess(tsne_input)
    if tsne_2d is None:
        note_path = os.path.join(EVIDENCE_DIR, "tsne_status.txt")
        with open(note_path, "w") as f:
            f.write("t-SNE failed in isolated child process (likely environment-level instability).\n")
            f.write("Core Phase 6 outputs are still complete (PCA + feature importance + summary).\n")
        log(f"t-SNE failed safely. Note saved: {note_path}")
        return "failed safely (child process)"

    tsne_df = pd.DataFrame({"userId": tsne_user_ids, "tsne_1": tsne_2d[:, 0], "tsne_2": tsne_2d[:, 1]})
    tsne_path = os.path.join(EVIDENCE_DIR, "user_embeddings_tsne_2d_sample.csv")
    tsne_df.to_csv(tsne_path, index=False)
    log(f"Saved t-SNE coordinates (sample): {tsne_path}")

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_df["tsne_1"], tsne_df["tsne_2"], s=8, alpha=0.7)
    plt.title("User Embeddings - t-SNE 2D (Sample)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.2)
    tsne_plot_path = os.path.join(EVIDENCE_DIR, "user_embeddings_tsne_2d_sample.png")
    plt.savefig(tsne_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved t-SNE plot: {tsne_plot_path}")
    return "completed"


def save_gb_feature_importance():
    """Loads Gradient Boosting model and saves feature importance evidence."""
    gb_path = os.path.join(MODELS_DIR, "gradient_boosting.pkl")
    if not os.path.exists(gb_path):
        raise FileNotFoundError(f"Gradient Boosting model not found: {gb_path}")

    gb_model = joblib.load(gb_path)
    importances = gb_model.feature_importances_
    fi_df = pd.DataFrame({"feature": ML_FEATURE_COLS, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)

    fi_csv_path = os.path.join(EVIDENCE_DIR, "gradient_boosting_feature_importance.csv")
    fi_df.to_csv(fi_csv_path, index=False)
    log(f"Saved feature importance table: {fi_csv_path}")

    plt.figure(figsize=(9, 5))
    plt.barh(fi_df["feature"], fi_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Gradient Boosting Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    fi_plot_path = os.path.join(EVIDENCE_DIR, "gradient_boosting_feature_importance.png")
    plt.savefig(fi_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved feature importance plot: {fi_plot_path}")
    return fi_df


def write_summary(best_tuned_rmse_from_file, retrained_rmse, fi_df, tsne_status):
    """Writes a small summary with key observations."""
    top_features = ", ".join(fi_df["feature"].head(3).tolist())
    lines = [
        "Phase 6 Post-Modeling Analysis Summary",
        "=====================================",
        f"Best tuned RMSE from best_dl_params.txt: {best_tuned_rmse_from_file:.4f}",
        f"RMSE from retraining best tuned config ({TUNE_EPOCHS} epochs): {retrained_rmse:.4f}",
        f"t-SNE status: {tsne_status}",
        f"Top 3 Gradient Boosting features: {top_features}",
        "",
        "Note: Retrained RMSE may differ from tuning best due to random initialization and limited epochs.",
        "All artifacts are saved under evidence/phase6/.",
    ]
    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"Saved summary: {SUMMARY_PATH}")


def main():
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(POST_LOG_PATH, "w") as f:
        f.write("--- Phase 6 Post Analysis Log ---\n")

    log("--- Starting Phase 6: Post-Modeling Analysis & XAI ---")
    best_rmse_file, params = parse_best_params(os.path.join(MODELS_DIR, "best_dl_params.txt"))
    log(f"Loaded best tuned RMSE from file: {best_rmse_file:.4f}")
    log(f"Loaded tuned params keys: {', '.join(sorted(params.keys()))}")

    train_df, test_df = load_datasets()
    model, retrained_rmse, _ = train_best_tuned_model(train_df, test_df, params)
    tsne_status = save_embedding_visuals(model)
    fi_df = save_gb_feature_importance()
    write_summary(best_rmse_file, retrained_rmse, fi_df, tsne_status)
    log("--- Phase 6 analysis complete ---")


if __name__ == "__main__":
    main()
