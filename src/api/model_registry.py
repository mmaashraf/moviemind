import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..dl_model import EMBEDDING_DIM, NeuralCollaborativeFiltering
from ..ml_models import FEATURE_COLS as ML_FEATURE_COLS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ml-1m")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RATING_MIN = 1.0
RATING_MAX = 5.0
RECOMMEND_DL_BATCH_SIZE = 512

DL_FEATURE_COLS = [
    "user_rating_count",
    "user_avg_rating",
    "movie_rating_count",
    "movie_avg_rating",
    "age",
    "release_year",
]

MODEL_DEFS = {
    "baseline_global_mean": {"display_name": "Baseline (Global Mean)", "family": "baseline", "artifact": None},
    "linear_regression": {
        "display_name": "Linear Regression",
        "family": "ml",
        "artifact": os.path.join(MODELS_DIR, "linear_regression.pkl"),
    },
    "random_forest": {
        "display_name": "Random Forest",
        "family": "ml",
        "artifact": os.path.join(MODELS_DIR, "random_forest.pkl"),
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting",
        "family": "ml",
        "artifact": os.path.join(MODELS_DIR, "gradient_boosting.pkl"),
    },
    "ncf_baseline": {
        "display_name": "NCF Baseline",
        "family": "dl",
        "artifact": os.path.join(MODELS_DIR, "ncf_model.pt"),
    },
    "ncf_tuned": {
        "display_name": "NCF Tuned Best",
        "family": "dl_tuned",
        "artifact": os.path.join(MODELS_DIR, "ncf_tuned_best.pt"),
    },
}


class DynamicNCF(nn.Module):
    def __init__(self, num_users: int, num_movies: int, num_dense_features: int, embedding_dim: int, hidden_layers: List[int], dropout_rate: float):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users + 1, embedding_dim=embedding_dim)
        self.movie_embedding = nn.Embedding(num_embeddings=num_movies + 1, embedding_dim=embedding_dim)
        total_input_dim = (embedding_dim * 2) + num_dense_features
        layers = []
        in_dim = total_input_dim
        for out_dim in hidden_layers:
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor, dense_features: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_embedding(user_idx)
        m_emb = self.movie_embedding(movie_idx)
        x = torch.cat([u_emb, m_emb, dense_features], dim=1)
        return self.fc_layers(x).squeeze()


@dataclass
class ModelRuntime:
    model_id: str
    display_name: str
    family: str
    artifact_path: Optional[str]
    available: bool
    estimator: Any = None
    params: Optional[Dict[str, Any]] = None


class ModelRegistry:
    def __init__(self) -> None:
        # Registry loads data and model artifacts once so API requests stay fast.
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.train_df, self.test_df = self._load_processed_frames()
        self.full_df = pd.concat([self.train_df, self.test_df], ignore_index=True)
        self.global_mean = float(self.train_df["rating"].mean())
        self.user_stats, self.movie_stats = self._build_stats(self.full_df)
        self.user_meta = self.full_df.groupby("userId")[["age", "occupation"]].first()
        self.movie_meta = self._load_movie_metadata()
        self.user_seen_movies = self.full_df.groupby("userId")["movieId"].apply(set).to_dict()
        self.train_user_stats = self.train_df.groupby("userId").agg(
            rating_count_train=("rating", "count"),
            avg_rating_train=("rating", "mean"),
            first_timestamp_train=("timestamp", "min"),
            last_timestamp_train=("timestamp", "max"),
        )
        self.max_user_id = int(self.full_df["userId"].max())
        self.max_movie_id = int(self.full_df["movieId"].max())
        self._models: Dict[str, ModelRuntime] = {}
        self._initialize_models()

    def _load_processed_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads processed train/test feature files used for online inference stats."""
        train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features.csv"), low_memory=False)
        test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test_features.csv"), low_memory=False)
        return train_df, test_df

    def _build_stats(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Precomputes user/movie aggregates so request-time feature building is fast."""
        user_stats = frame.groupby("userId").agg(user_rating_count=("rating", "count"), user_avg_rating=("rating", "mean"))
        movie_stats = frame.groupby("movieId").agg(movie_rating_count=("rating", "count"), movie_avg_rating=("rating", "mean"))
        return user_stats, movie_stats

    def _load_movie_metadata(self) -> pd.DataFrame:
        """Loads movie title/genre/year metadata for recommendation output."""
        movies_path = os.path.join(RAW_DATA_DIR, "movies.dat")
        movies = pd.read_csv(movies_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding="latin-1")
        year_extract = movies["title"].str.extract(r"\((\d{4})\)")
        movies["release_year"] = pd.to_numeric(year_extract[0], errors="coerce").fillna(1900).astype(int)
        return movies.set_index("movieId")

    def _parse_best_params(self) -> Dict[str, Any]:
        """Reads tuned DL params so tuned model can be reconstructed for inference."""
        params_path = os.path.join(MODELS_DIR, "best_dl_params.txt")
        if not os.path.exists(params_path):
            return {}
        params: Dict[str, Any] = {}
        with open(params_path, "r") as f:
            for line in f:
                clean = line.strip()
                if ":" not in clean:
                    continue
                key, value = [x.strip() for x in clean.split(":", 1)]
                if key.lower() == "best rmse":
                    params["best_rmse"] = float(value)
                elif key in {"embedding_dim", "n_layers"} or key.startswith("n_units_l"):
                    params[key] = int(float(value))
                else:
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value
        return params

    def _initialize_models(self) -> None:
        """Loads all model runtimes once and records availability per model."""
        tuned_params = self._parse_best_params()
        for model_id, cfg in MODEL_DEFS.items():
            artifact = cfg["artifact"]
            runtime = ModelRuntime(
                model_id=model_id,
                display_name=cfg["display_name"],
                family=cfg["family"],
                artifact_path=artifact,
                available=(artifact is None or os.path.exists(artifact)),
                params={},
            )
            try:
                if model_id == "baseline_global_mean":
                    runtime.params = {"global_mean_rating": self.global_mean}
                elif runtime.available and runtime.family in {"ml"}:
                    # sklearn pickle versions can differ between train/runtime envs;
                    # we keep loading logic defensive to avoid hard API crashes.
                    runtime.estimator = joblib.load(artifact)
                    runtime.params = {"class_name": runtime.estimator.__class__.__name__}
                elif runtime.available and model_id == "ncf_baseline":
                    model = NeuralCollaborativeFiltering(self.max_user_id, self.max_movie_id, len(DL_FEATURE_COLS)).to(self.device)
                    model.load_state_dict(torch.load(artifact, map_location=self.device))
                    model.eval()
                    runtime.estimator = model
                    runtime.params = {"embedding_dim": EMBEDDING_DIM}
                elif runtime.available and model_id == "ncf_tuned":
                    n_layers = int(tuned_params.get("n_layers", 3))
                    hidden_layers = [int(tuned_params.get(f"n_units_l{i}", 128)) for i in range(n_layers)]
                    embedding_dim = int(tuned_params.get("embedding_dim", 64))
                    dropout_rate = float(tuned_params.get("dropout_rate", 0.2))
                    model = DynamicNCF(self.max_user_id, self.max_movie_id, len(DL_FEATURE_COLS), embedding_dim, hidden_layers, dropout_rate).to(self.device)
                    model.load_state_dict(torch.load(artifact, map_location=self.device))
                    model.eval()
                    runtime.estimator = model
                    runtime.params = tuned_params
            except Exception as exc:
                # Model-level failure should not crash the full API. We mark it unavailable
                # so UI/clients can still use other models.
                runtime.available = False
                runtime.estimator = None
                runtime.params = {"load_error": str(exc)}
            self._models[model_id] = runtime

    def list_models(self) -> List[ModelRuntime]:
        """Returns model summaries in fixed display order."""
        return [self._models[k] for k in MODEL_DEFS.keys()]

    def get_model(self, model_id: str) -> ModelRuntime:
        """Fetches one model runtime by ID with explicit not-found handling."""
        if model_id not in self._models:
            raise KeyError(f"Unknown model_id: {model_id}")
        return self._models[model_id]

    def _build_feature_row(self, user_id: int, movie_id: int) -> Dict[str, float]:
        # Cold-start fallback: if user/movie is unseen, use safe defaults from global stats.
        user_count = float(self.user_stats.loc[user_id, "user_rating_count"]) if user_id in self.user_stats.index else 0.0
        user_avg = float(self.user_stats.loc[user_id, "user_avg_rating"]) if user_id in self.user_stats.index else self.global_mean
        movie_count = float(self.movie_stats.loc[movie_id, "movie_rating_count"]) if movie_id in self.movie_stats.index else 0.0
        movie_avg = float(self.movie_stats.loc[movie_id, "movie_avg_rating"]) if movie_id in self.movie_stats.index else self.global_mean
        age = float(self.user_meta.loc[user_id, "age"]) if user_id in self.user_meta.index else 0.0
        occupation = float(self.user_meta.loc[user_id, "occupation"]) if user_id in self.user_meta.index else 0.0
        release_year = float(self.movie_meta.loc[movie_id, "release_year"]) if movie_id in self.movie_meta.index else 1900.0
        return {
            "user_rating_count": user_count,
            "user_avg_rating": user_avg,
            "movie_rating_count": movie_count,
            "movie_avg_rating": movie_avg,
            "age": age,
            "occupation": occupation,
            "release_year": release_year,
        }

    def _predict_with_dl(self, runtime: ModelRuntime, user_id: int, movie_id: int, row: Dict[str, float]) -> float:
        """Single-item DL prediction path used by /predict endpoint."""
        model = runtime.estimator
        if model is None:
            raise RuntimeError(f"Model not loaded: {runtime.model_id}")
        user_idx = torch.tensor([max(user_id - 1, 0)], dtype=torch.long, device=self.device)
        movie_idx = torch.tensor([max(movie_id - 1, 0)], dtype=torch.long, device=self.device)
        dense = torch.tensor(
            [[row["user_rating_count"], row["user_avg_rating"], row["movie_rating_count"], row["movie_avg_rating"], row["age"], row["release_year"]]],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            pred = model(user_idx, movie_idx, dense).item()
        return float(pred)

    def _build_feature_frame(self, user_id: int, movie_ids: List[int]) -> pd.DataFrame:
        """Builds one dataframe for bulk scoring instead of per-movie dict work."""
        user_count = float(self.user_stats.loc[user_id, "user_rating_count"]) if user_id in self.user_stats.index else 0.0
        user_avg = float(self.user_stats.loc[user_id, "user_avg_rating"]) if user_id in self.user_stats.index else self.global_mean
        age = float(self.user_meta.loc[user_id, "age"]) if user_id in self.user_meta.index else 0.0
        occupation = float(self.user_meta.loc[user_id, "occupation"]) if user_id in self.user_meta.index else 0.0

        movie_df = self.movie_meta.loc[movie_ids, ["release_year"]].copy()
        movie_df["movieId"] = movie_df.index.astype(int)

        stats = self.movie_stats.reindex(movie_ids)
        movie_df["movie_rating_count"] = stats["movie_rating_count"].fillna(0.0).to_numpy()
        movie_df["movie_avg_rating"] = stats["movie_avg_rating"].fillna(self.global_mean).to_numpy()
        movie_df["user_rating_count"] = user_count
        movie_df["user_avg_rating"] = user_avg
        movie_df["age"] = age
        movie_df["occupation"] = occupation
        movie_df = movie_df.reset_index(drop=True)
        return movie_df

    def _predict_with_dl_batch(
        self,
        runtime: ModelRuntime,
        user_id: int,
        feature_df: pd.DataFrame,
        batch_size: int = RECOMMEND_DL_BATCH_SIZE,
    ) -> np.ndarray:
        """Scores candidate movies in mini-batches for better latency."""
        model = runtime.estimator
        if model is None:
            raise RuntimeError(f"Model not loaded: {runtime.model_id}")

        user_idx_full = np.full(len(feature_df), max(user_id - 1, 0), dtype=np.int64)
        movie_idx_full = np.maximum(feature_df["movieId"].to_numpy(dtype=np.int64) - 1, 0)
        dense_full = feature_df[
            ["user_rating_count", "user_avg_rating", "movie_rating_count", "movie_avg_rating", "age", "release_year"]
        ].to_numpy(dtype=np.float32)

        preds = []
        with torch.no_grad():
            for i in range(0, len(feature_df), batch_size):
                j = i + batch_size
                users = torch.tensor(user_idx_full[i:j], dtype=torch.long, device=self.device)
                movies = torch.tensor(movie_idx_full[i:j], dtype=torch.long, device=self.device)
                dense = torch.tensor(dense_full[i:j], dtype=torch.float32, device=self.device)
                batch_pred = model(users, movies, dense).detach().cpu().numpy()
                preds.append(batch_pred)
        if not preds:
            return np.array([], dtype=np.float32)
        return np.concatenate(preds, axis=0)

    def _clip_rating(self, value: float) -> float:
        """Clips predictions to the canonical MovieLens rating scale."""
        return float(np.clip(value, RATING_MIN, RATING_MAX))

    def predict(self, model_id: str, user_id: int, movie_id: int) -> Tuple[float, bool]:
        """Predicts one rating and returns value plus whether clipping was applied."""
        runtime = self.get_model(model_id)
        if not runtime.available:
            raise FileNotFoundError(f"Model artifact missing for: {model_id}")

        row = self._build_feature_row(user_id=user_id, movie_id=movie_id)
        if runtime.family == "baseline":
            raw = self.global_mean
        elif runtime.family == "ml":
            features = pd.DataFrame([[row[c] for c in ML_FEATURE_COLS]], columns=ML_FEATURE_COLS)
            raw = float(runtime.estimator.predict(features)[0])
        else:
            raw = self._predict_with_dl(runtime, user_id, movie_id, row)

        # Keep outputs on MovieLens scale so downstream UI/ranking stays consistent.
        clipped = self._clip_rating(raw)
        return clipped, abs(clipped - raw) > 1e-9

    def _genres_set(self, genres_text: str) -> set:
        return {g.strip() for g in str(genres_text).split("|") if g.strip()}

    def _diversity_rerank(self, scored: pd.DataFrame, top_n: int, diversity_alpha: float) -> pd.DataFrame:
        """Greedy rerank that trades predicted score vs genre overlap with selected items."""
        if diversity_alpha <= 0 or scored.empty:
            base = scored.nlargest(top_n, "predicted_rating").copy()
            base["overlap_penalty"] = 0.0
            base["adjusted_score"] = base["predicted_rating"]
            base["overlap_genres"] = [[] for _ in range(len(base))]
            return base

        pool = scored.nlargest(min(250, len(scored)), "predicted_rating").copy()
        selected_rows: List[pd.Series] = []
        selected_genres: set = set()

        while len(selected_rows) < top_n and not pool.empty:
            best_idx = None
            best_score = -1e9
            best_overlap = 0.0
            best_overlap_genres: List[str] = []
            for idx, row in pool.iterrows():
                gset = row["genre_set"]
                overlap = 0.0
                if selected_genres and gset:
                    overlap = len(gset & selected_genres) / max(len(gset), 1)
                adjusted = float(row["predicted_rating"]) - diversity_alpha * overlap
                if adjusted > best_score:
                    best_score = adjusted
                    best_idx = idx
                    best_overlap = overlap
                    best_overlap_genres = sorted(list(gset & selected_genres))
            chosen = pool.loc[best_idx]
            chosen = chosen.copy()
            chosen["overlap_penalty"] = float(diversity_alpha * best_overlap)
            chosen["adjusted_score"] = float(best_score)
            chosen["overlap_genres"] = best_overlap_genres
            selected_rows.append(chosen)
            selected_genres.update(chosen["genre_set"])
            pool = pool.drop(index=best_idx)

        return pd.DataFrame(selected_rows)

    def recommend(
        self,
        model_id: str,
        user_id: int,
        top_n: int,
        diversity_alpha: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Returns top-N unseen movies for user by batched model scoring with optional diversity rerank."""
        runtime = self.get_model(model_id)
        if not runtime.available:
            raise FileNotFoundError(f"Model artifact missing for: {model_id}")

        seen = self.user_seen_movies.get(user_id, set())
        # Recommend only unseen movies for that user (simple candidate generation).
        candidates = [mid for mid in self.movie_meta.index.tolist() if mid not in seen]
        if not candidates:
            return []

        feature_df = self._build_feature_frame(user_id=user_id, movie_ids=candidates)
        if runtime.family == "baseline":
            raw_preds = np.full(len(feature_df), self.global_mean, dtype=np.float32)
        elif runtime.family == "ml":
            ml_input = feature_df[ML_FEATURE_COLS]
            raw_preds = runtime.estimator.predict(ml_input)
        else:
            raw_preds = self._predict_with_dl_batch(runtime, user_id=user_id, feature_df=feature_df)

        clipped_preds = np.clip(raw_preds, RATING_MIN, RATING_MAX)
        scored = feature_df[["movieId"]].copy()
        scored["predicted_rating"] = clipped_preds
        scored["title"] = scored["movieId"].map(lambda m: str(self.movie_meta.loc[int(m), "title"]))
        scored["genres"] = scored["movieId"].map(lambda m: str(self.movie_meta.loc[int(m), "genres"]))
        scored["genre_set"] = scored["genres"].map(self._genres_set)
        top = self._diversity_rerank(scored, top_n=top_n, diversity_alpha=diversity_alpha)

        user_top_counts = self._user_top_genre_counts(user_id=user_id, top_k=5)
        user_top = set(user_top_counts.keys())

        scores = []
        for _, row in top.iterrows():
            movie_id = int(row["movieId"])
            title = str(row["title"])
            genres = str(row["genres"])
            gset = row["genre_set"]
            overlap = sorted(list(gset & user_top))
            if overlap:
                reason = f"High predicted rating; overlaps your top genres: {', '.join(overlap[:2])}."
            else:
                reason = "High predicted rating; adds some genre novelty."
            scores.append(
                {
                    "movie_id": movie_id,
                    "title": title,
                    "genres": genres,
                    "predicted_rating": round(float(row["predicted_rating"]), 4),
                    "reason": reason,
                    "predicted_rating_raw": round(float(row["predicted_rating"]), 4),
                    "overlap_penalty": round(float(row.get("overlap_penalty", 0.0)), 4),
                    "adjusted_score": round(float(row.get("adjusted_score", row["predicted_rating"])), 4),
                    "overlap_genres": row.get("overlap_genres", []),
                }
            )
        return scores

    def _extract_metrics(self, model_id: str) -> Dict[str, Any]:
        """Builds lightweight metrics payload used in model inspector UI."""
        metrics: Dict[str, Any] = {}
        ml_log = os.path.join(MODELS_DIR, "ml_training_log.txt")
        if model_id in {"baseline_global_mean", "linear_regression", "random_forest", "gradient_boosting"} and os.path.exists(ml_log):
            with open(ml_log, "r") as f:
                text = f.read()
            label_map = {
                "baseline_global_mean": "Baseline (Global Mean)",
                "linear_regression": "Linear Regression",
                "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting",
            }
            model_label = label_map[model_id]
            line_match = re.search(rf"{re.escape(model_label)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text)
            if line_match:
                metrics = {
                    "rmse": float(line_match.group(1)),
                    "mae": float(line_match.group(2)),
                    "time_sec": float(line_match.group(3)),
                }
        if model_id in {"ncf_baseline", "ncf_tuned"}:
            if model_id == "ncf_tuned":
                params = self._parse_best_params()
                if "best_rmse" in params:
                    metrics["best_rmse"] = params["best_rmse"]
            dl_log = os.path.join(MODELS_DIR, "dl_training_log.txt")
            if os.path.exists(dl_log):
                metrics["log_path"] = dl_log
        return metrics

    def _user_top_genre_counts(self, user_id: int, top_k: int = 5) -> Dict[str, int]:
        if "genres" not in self.train_df.columns:
            return {}
        user_rows = self.train_df[self.train_df["userId"] == user_id]
        if user_rows.empty:
            return {}
        exploded = (
            user_rows["genres"]
            .dropna()
            .astype(str)
            .str.split("|")
            .explode()
            .str.strip()
        )
        if exploded.empty:
            return {}
        counts = exploded.value_counts().head(top_k)
        return {str(k): int(v) for k, v in counts.items()}

    def model_info(self, model_id: str) -> Dict[str, Any]:
        """Returns moderate inspector payload (params/metrics/family-specific details)."""
        runtime = self.get_model(model_id)
        inspector: Dict[str, Any] = {}
        if model_id == "gradient_boosting" and runtime.estimator is not None:
            importances = runtime.estimator.feature_importances_
            pairs = sorted(zip(ML_FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
            inspector["top_feature_importances"] = [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs[:5]]
        if model_id.startswith("ncf") and runtime.estimator is not None:
            inspector["embedding_dimensions"] = int(runtime.estimator.user_embedding.embedding_dim)
            inspector["parameter_count"] = int(sum(p.numel() for p in runtime.estimator.parameters()))
            inspector["device"] = str(self.device)
        return {
            "model_id": runtime.model_id,
            "display_name": runtime.display_name,
            "family": runtime.family,
            "artifact_path": runtime.artifact_path,
            "available": runtime.available,
            "params": runtime.params or {},
            "metrics": self._extract_metrics(model_id),
            "inspector": inspector,
        }

    def user_summary(self, user_id: int) -> Dict[str, Any]:
        """Returns training-data summary for selected user ID."""
        found = user_id in self.train_user_stats.index
        if found:
            stats = self.train_user_stats.loc[user_id]
            rating_count_train = int(stats["rating_count_train"])
            avg_rating_train = float(stats["avg_rating_train"])
            first_ts = int(stats["first_timestamp_train"])
            last_ts = int(stats["last_timestamp_train"])
        else:
            rating_count_train = 0
            avg_rating_train = 0.0
            first_ts = None
            last_ts = None

        demo_row = self.full_df[self.full_df["userId"] == user_id][["age", "occupation", "gender"]].head(1)
        age = int(demo_row.iloc[0]["age"]) if not demo_row.empty and pd.notna(demo_row.iloc[0]["age"]) else None
        occupation = (
            int(demo_row.iloc[0]["occupation"]) if not demo_row.empty and pd.notna(demo_row.iloc[0]["occupation"]) else None
        )
        gender = str(demo_row.iloc[0]["gender"]) if not demo_row.empty and pd.notna(demo_row.iloc[0]["gender"]) else None

        top_genre_counts = self._user_top_genre_counts(user_id=user_id, top_k=5) if found else {}
        top_genres: List[str] = list(top_genre_counts.keys())

        return {
            "user_id": user_id,
            "found_in_training": found,
            "rating_count_train": rating_count_train,
            "avg_rating_train": round(avg_rating_train, 4),
            "age": age,
            "occupation": occupation,
            "gender": gender,
            "first_timestamp_train": first_ts,
            "last_timestamp_train": last_ts,
            "top_genres_train": top_genres,
            "top_genre_counts": top_genre_counts,
        }

