import pandas as pd
import numpy as np
import os
import re

DATA_DIR = "data/ml-1m"
PROCESSED_DIR = "data/processed"

def load_data():
    """Loads raw MovieLens 1M data."""
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), sep='::', engine='python', 
                          names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin-1')
    movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.dat'), sep='::', engine='python', 
                         names=['movieId', 'title', 'genres'], encoding='latin-1')
    users = pd.read_csv(os.path.join(DATA_DIR, 'users.dat'), sep='::', engine='python', 
                        names=['userId', 'gender', 'age', 'occupation', 'zipCode'], encoding='latin-1')
    return ratings, movies, users

def extract_movie_year(title):
    """Extracts the 4-digit release year from the movie title."""
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else 1900 # Default to 1900 if missing

def engineer_features(ratings, movies, users):
    """
    Computes statistical features and merges all data into a single DataFrame.
    """
    print("1. Extracting Movie Year...")
    movies['release_year'] = movies['title'].apply(extract_movie_year)
    
    print("2. Computing User Features (Activity & Average)...")
    user_stats = ratings.groupby('userId').agg(
        user_rating_count=('rating', 'count'),
        user_avg_rating=('rating', 'mean')
    ).reset_index()
    
    print("3. Computing Movie Features (Popularity & Average)...")
    movie_stats = ratings.groupby('movieId').agg(
        movie_rating_count=('rating', 'count'),
        movie_avg_rating=('rating', 'mean')
    ).reset_index()
    
    print("4. Merging everything into a single dataset...")
    # Merge ratings with user stats and movie stats
    df = ratings.merge(user_stats, on='userId', how='left')
    df = df.merge(movie_stats, on='movieId', how='left')
    
    # Merge with original demographics and movie metadata
    df = df.merge(users, on='userId', how='left')
    df = df.merge(movies, on='movieId', how='left')
    
    return df

def time_based_split(df, test_size=0.2):
    """
    Splits the data based on timestamp to simulate real-world prediction (future).
    Prevents data leakage.
    """
    print(f"5. Performing Time-Based Train/Test Split (Test Size: {test_size * 100}%)...")
    df_sorted = df.sort_values('timestamp')
    
    split_index = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]
    
    return train_df, test_df

def main():
    print("--- Starting Feature Engineering Phase ---")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    ratings, movies, users = load_data()
    
    # Build features
    full_dataset = engineer_features(ratings, movies, users)
    
    # Split
    train_df, test_df = time_based_split(full_dataset)
    
    print("6. Saving processed datasets...")
    # Saving as CSV. Note: For massive datasets, Parquet is better, but CSV is universally easy to debug.
    train_path = os.path.join(PROCESSED_DIR, 'train_features.csv')
    test_path = os.path.join(PROCESSED_DIR, 'test_features.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Done! Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")
    print(f"Data saved to {PROCESSED_DIR}/")

if __name__ == "__main__":
    main()
