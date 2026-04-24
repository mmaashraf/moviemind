import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from dl_model import MovieLensDataset, PROCESSED_DIR, MODELS_DIR

# Keep epochs small for tuning so it doesn't take 3 hours
TUNE_EPOCHS = 3
N_TRIALS = 50
BATCH_SIZE = 1024
EVIDENCE_DIR = os.path.join("evidence", "phase5b")
REQUIRED_COLS = [
    "userId",
    "movieId",
    "rating",
    "user_rating_count",
    "user_avg_rating",
    "movie_rating_count",
    "movie_avg_rating",
    "age",
    "release_year",
]

# We load these once in main() so each Optuna trial is faster.
TRAIN_LOADER = None
TEST_LOADER = None
NUM_USERS = None
NUM_MOVIES = None
DEVICE = None

def log_to_file(msg, filepath=os.path.join(MODELS_DIR, 'tune_dl_log.txt')):
    """Appends a message to the tuning log file."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(msg + '\n')
    with open(os.path.join(EVIDENCE_DIR, "tune_dl_observations.txt"), 'a') as f:
        f.write(msg + '\n')
    print(msg)


def initialize_tuning_data():
    """
    Loads and prepares data once for all trials.
    This avoids re-reading large CSVs inside every objective() call.
    """
    global TRAIN_LOADER, TEST_LOADER, NUM_USERS, NUM_MOVIES, DEVICE

    train_path = os.path.join(PROCESSED_DIR, "train_features.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_features.csv")

    train_df = pd.read_csv(train_path, usecols=REQUIRED_COLS, low_memory=False)
    test_df = pd.read_csv(test_path, usecols=REQUIRED_COLS, low_memory=False)

    NUM_USERS = int(max(train_df["userId"].max(), test_df["userId"].max()))
    NUM_MOVIES = int(max(train_df["movieId"].max(), test_df["movieId"].max()))

    TRAIN_LOADER = DataLoader(MovieLensDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    TEST_LOADER = DataLoader(MovieLensDataset(test_df), batch_size=BATCH_SIZE, shuffle=False)

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log_to_file(f"Data initialized once. Train rows: {len(train_df):,}, Test rows: {len(test_df):,}")
    log_to_file(f"Using device: {DEVICE}")

class DynamicNCF(nn.Module):
    """A version of our NCF model where architecture is dynamically built by Optuna."""
    def __init__(self, num_users, num_movies, num_dense_features, embedding_dim, hidden_layers, dropout_rate):
        super(DynamicNCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_embeddings=num_users + 1, embedding_dim=embedding_dim)
        self.movie_embedding = nn.Embedding(num_embeddings=num_movies + 1, embedding_dim=embedding_dim)
        
        total_input_dim = (embedding_dim * 2) + num_dense_features
        
        # Dynamically create layers based on Optuna's suggestions
        layers = []
        in_dim = total_input_dim
        for out_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim
            
        # Final output layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, user_idx, movie_idx, dense_features):
        u_emb = self.user_embedding(user_idx)
        m_emb = self.movie_embedding(movie_idx)
        x = torch.cat([u_emb, m_emb, dense_features], dim=1)
        return self.fc_layers(x).squeeze()

def objective(trial):
    """Optuna objective function. Suggests hyperparameters, trains, and returns validation RMSE."""
    global TRAIN_LOADER, TEST_LOADER, NUM_USERS, NUM_MOVIES, DEVICE
    
    # 1. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
    # Suggest network depth and width
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_layers = []
    for i in range(n_layers):
        hidden_layers.append(trial.suggest_int(f"n_units_l{i}", 32, 256, step=32))

    # 2. Setup Model
    model = DynamicNCF(NUM_USERS, NUM_MOVIES, 6, embedding_dim, hidden_layers, dropout_rate).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 3. Train for a few epochs
    for epoch in range(TUNE_EPOCHS):
        model.train()
        for users, movies, dense, ratings in TRAIN_LOADER:
            users, movies, dense, ratings = users.to(DEVICE), movies.to(DEVICE), dense.to(DEVICE), ratings.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(users, movies, dense), ratings)
            loss.backward()
            optimizer.step()
            
    # 4. Evaluate (This is what Optuna uses to score the trial)
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for users, movies, dense, ratings in TEST_LOADER:
            users, movies, dense, ratings = users.to(DEVICE), movies.to(DEVICE), dense.to(DEVICE), ratings.to(DEVICE)
            loss = criterion(model(users, movies, dense), ratings)
            running_val_loss += loss.item() * users.size(0)
            
    val_mse = running_val_loss / len(TEST_LOADER.dataset)
    val_rmse = np.sqrt(val_mse)
    
    return val_rmse

def main():
    run_start = time.time()
    log_to_file("--- Starting Optuna Hyperparameter Tuning ---")
    log_to_file(f"Running {N_TRIALS} trials (each trial trains for {TUNE_EPOCHS} epochs)")
    initialize_tuning_data()
    
    # Create study to MINIMIZE the RMSE
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    
    log_to_file("\n=== TUNING COMPLETE ===")
    log_to_file(f"Best trial RMSE: {study.best_trial.value:.4f}")
    log_to_file("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        log_to_file(f"    {key}: {value}")
        
    # Save the best parameters to a text file
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'best_dl_params.txt'), 'w') as f:
        f.write(f"Best RMSE: {study.best_trial.value:.4f}\n")
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")
    elapsed_min = (time.time() - run_start) / 60.0
    log_to_file(f"Total tuning runtime: {elapsed_min:.2f} minutes")
    log_to_file("Saved artifacts: models/best_dl_params.txt, models/tune_dl_log.txt, evidence/phase5b/tune_dl_observations.txt")

if __name__ == "__main__":
    main()
