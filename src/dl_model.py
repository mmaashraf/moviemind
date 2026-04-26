import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# Hyperparameters
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.001
EMBEDDING_DIM = 32

class MovieLensDataset(Dataset):
    def __init__(self, df):
        # We need the user and movie IDs for embeddings. 
        # Note: PyTorch embeddings need 0-indexed integer IDs.
        # MovieLens IDs start at 1, so we subtract 1.
        self.users = torch.tensor(df['userId'].values - 1, dtype=torch.long)
        self.movies = torch.tensor(df['movieId'].values - 1, dtype=torch.long)
        
        # Dense numerical features from Phase 3
        feature_cols = ['user_rating_count', 'user_avg_rating', 'movie_rating_count', 'movie_avg_rating', 'age', 'release_year']
        self.dense_features = torch.tensor(df[feature_cols].fillna(0).values, dtype=torch.float32)
        
        # Target
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.dense_features[idx], self.ratings[idx]

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_movies, num_dense_features):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # Embeddings for Collaborative Filtering
        self.user_embedding = nn.Embedding(num_embeddings=num_users + 1, embedding_dim=EMBEDDING_DIM)
        self.movie_embedding = nn.Embedding(num_embeddings=num_movies + 1, embedding_dim=EMBEDDING_DIM)
        
        # Dense network for combining embeddings and content features
        total_input_dim = (EMBEDDING_DIM * 2) + num_dense_features
        
        self.fc_layers = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Predicts a single rating value
        )

    def forward(self, user_idx, movie_idx, dense_features):
        u_emb = self.user_embedding(user_idx)
        m_emb = self.movie_embedding(movie_idx)
        
        # Concatenate embeddings and numerical features
        x = torch.cat([u_emb, m_emb, dense_features], dim=1)
        
        # Pass through dense layers
        output = self.fc_layers(x)
        return output.squeeze()

def plot_loss_curve(train_losses, val_losses):
    """Saves the loss curve plot as requested by Rule #10."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(val_losses, label='Validation Loss (MSE)')
    plt.title('Deep Learning Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    plot_path = os.path.join(MODELS_DIR, 'dl_loss_curve.png')
    plt.savefig(plot_path)
    log_msg = f"Loss curve saved to {plot_path}"
    print(log_msg)
    log_to_file(log_msg)

def log_to_file(msg, filepath=os.path.join(MODELS_DIR, 'dl_training_log.txt')):
    """Appends a message to the training log file."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(msg + '\n')

def main():
    log_to_file("--- Phase 5: Deep Learning (Neural Collaborative Filtering) ---\n")
    print("--- Phase 5: Deep Learning (Neural Collaborative Filtering) ---\n")
    
    # 1. Load Data
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_features.csv'))
    val_path = os.path.join(PROCESSED_DIR, 'val_features.csv')
    if os.path.exists(val_path):
        eval_df = pd.read_csv(val_path)
        eval_label = "val"
    else:
        # Backward-compatible fallback for old 2-way split artifacts.
        eval_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test_features.csv'))
        eval_label = "test"
    
    num_users = max(train_df['userId'].max(), eval_df['userId'].max())
    num_movies = max(train_df['movieId'].max(), eval_df['movieId'].max())
    
    train_dataset = MovieLensDataset(train_df)
    eval_dataset = MovieLensDataset(eval_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    # Determine device (Use Apple Silicon MPS if available, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    msg_dev = f"Training on device: {device}\n"
    print(msg_dev)
    log_to_file(msg_dev)
    
    model = NeuralCollaborativeFiltering(num_users=num_users, num_movies=num_movies, num_dense_features=6)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop with Timers and Logging (Rule #10)
    train_losses = []
    val_losses = []
    
    total_start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        running_train_loss = 0.0
        for users, movies, dense, ratings in train_loader:
            users, movies, dense, ratings = users.to(device), movies.to(device), dense.to(device), ratings.to(device)
            
            optimizer.zero_grad()
            outputs = model(users, movies, dense)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * users.size(0)
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for users, movies, dense, ratings in eval_loader:
                users, movies, dense, ratings = users.to(device), movies.to(device), dense.to(device), ratings.to(device)
                outputs = model(users, movies, dense)
                loss = criterion(outputs, ratings)
                running_val_loss += loss.item() * users.size(0)
                
        epoch_val_loss = running_val_loss / len(eval_dataset)
        val_losses.append(epoch_val_loss)
        
        epoch_duration = time.time() - epoch_start_time
        msg_epoch = (
            f"Epoch {epoch+1:02d}/{EPOCHS} | Time: {epoch_duration:.1f}s | "
            f"Train Loss (MSE): {epoch_train_loss:.4f} | "
            f"{eval_label.capitalize()} Loss (MSE): {epoch_val_loss:.4f} | "
            f"{eval_label.capitalize()} RMSE: {np.sqrt(epoch_val_loss):.4f}"
        )
        print(msg_epoch)
        log_to_file(msg_epoch)
        
    total_duration = time.time() - total_start_time
    msg_total = f"\nTraining Complete in {total_duration / 60:.2f} minutes."
    print(msg_total)
    log_to_file(msg_total)
    
    # 4. Save Artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'ncf_model.pt')
    torch.save(model.state_dict(), model_path)
    msg_save = f"Model saved to {model_path}"
    print(msg_save)
    log_to_file(msg_save)
    
    plot_loss_curve(train_losses, val_losses)

if __name__ == "__main__":
    main()
