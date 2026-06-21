#!/usr/bin/env python3
"""Generate EDA figures for docs/report/ from data/ml-1m/. Run from moviemind/ root."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "ml-1m"
OUT_DIR = Path(__file__).resolve().parent / "figures"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10


def load_data():
    ratings = pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        DATA_DIR / "users.dat",
        sep="::",
        engine="python",
        names=["userId", "gender", "age", "occupation", "zipCode"],
        encoding="latin-1",
    )
    return ratings, movies, users


def plot_rating_distribution(ratings: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=ratings, x="rating", hue="rating", palette="viridis", legend=False, ax=ax)
    ax.set_title("Distribution of Movie Ratings")
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rating_distribution.png")
    plt.close(fig)


def plot_demographics(users: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(data=users, x="gender", hue="gender", palette="Set2", legend=False, ax=axes[0])
    axes[0].set_title("User gender")
    sns.countplot(data=users, x="age", hue="age", palette="magma", legend=False, ax=axes[1])
    axes[1].set_title("User age (binned codes)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "user_demographics.png")
    plt.close(fig)


def plot_long_tail(ratings: pd.DataFrame) -> None:
    movie_pop = ratings.groupby("movieId").size().reset_index(name="num_ratings")
    movie_pop = movie_pop.sort_values("num_ratings", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(movie_pop.index, movie_pop["num_ratings"], color="indigo", linewidth=2)
    ax.fill_between(movie_pop.index, movie_pop["num_ratings"], color="indigo", alpha=0.3)
    ax.axhline(y=20, color="r", linestyle="--", label="Cold-start threshold (<20 ratings)")
    ax.set_title("Long Tail of Movie Popularity")
    ax.set_xlabel("Movies (ranked by popularity)")
    ax.set_ylabel("Number of ratings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "long_tail.png")
    plt.close(fig)


def plot_user_activity(ratings: pd.DataFrame) -> None:
    user_act = ratings.groupby("userId").size().reset_index(name="num_ratings")
    user_act = user_act.sort_values("num_ratings", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(user_act.index, user_act["num_ratings"], color="teal", linewidth=2)
    ax.fill_between(user_act.index, user_act["num_ratings"], color="teal", alpha=0.3)
    ax.axhline(y=50, color="r", linestyle="--", label="Casual users (<50 ratings)")
    ax.set_title("User Activity Distribution")
    ax.set_xlabel("Users (ranked by activity)")
    ax.set_ylabel("Number of ratings submitted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "user_activity.png")
    plt.close(fig)


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Missing {DATA_DIR}. Run data_loader or download_review_artifacts.sh")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ratings, movies, users = load_data()

    plot_rating_distribution(ratings)
    plot_demographics(users)
    plot_long_tail(ratings)
    plot_user_activity(ratings)

    sparsity = 1 - len(ratings) / (len(users) * len(movies))
    movie_counts = ratings.groupby("movieId").size()
    user_counts = ratings.groupby("userId").size()
    cold_movies = (movie_counts < 20).sum()
    casual_users = (user_counts < 50).sum()
    n_rated_movies = movie_counts.shape[0]

    print(f"ratings={len(ratings):,} users={len(users):,} movies={len(movies):,}")
    print(f"mean_rating={ratings['rating'].mean():.3f} median={ratings['rating'].median():.0f}")
    print(f"sparsity_pct={sparsity * 100:.2f}")
    print(f"cold_movies={cold_movies} ({100 * cold_movies / n_rated_movies:.1f}%)")
    print(f"casual_users={casual_users} ({100 * casual_users / len(users):.1f}%)")
    print(f"Wrote PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
