import pandas as pd
import os

def check_1m_schema(data_dir="data/ml-1m"):
    """
    Checks the schema of the MovieLens 1M dataset.
    The 1M dataset does not have headers and uses '::' as a separator.
    """
    print("--- MovieLens 1M Schema Check ---\n")
    
    # 1. Ratings
    ratings_file = os.path.join(data_dir, "ratings.dat")
    if os.path.exists(ratings_file):
        ratings = pd.read_csv(
            ratings_file, 
            sep='::', 
            engine='python', 
            names=['userId', 'movieId', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        print(f"Ratings Shape: {ratings.shape}")
        print("Ratings Head:")
        print(ratings.head(), "\n")
        
    # 2. Movies
    movies_file = os.path.join(data_dir, "movies.dat")
    if os.path.exists(movies_file):
        movies = pd.read_csv(
            movies_file, 
            sep='::', 
            engine='python', 
            names=['movieId', 'title', 'genres'],
            encoding='latin-1'
        )
        print(f"Movies Shape: {movies.shape}")
        print("Movies Head:")
        print(movies.head(), "\n")
        
    # 3. Users
    users_file = os.path.join(data_dir, "users.dat")
    if os.path.exists(users_file):
        users = pd.read_csv(
            users_file, 
            sep='::', 
            engine='python', 
            names=['userId', 'gender', 'age', 'occupation', 'zipCode'],
            encoding='latin-1'
        )
        print(f"Users Shape: {users.shape}")
        print("Users Head:")
        print(users.head(), "\n")

if __name__ == "__main__":
    check_1m_schema()
