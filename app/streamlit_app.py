import os
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from sklearn import tree as sktree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Ensure imports like `src.*` work when Streamlit runs from `app/`.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.model_registry import DynamicNCF
from src.dl_model import EMBEDDING_DIM, NeuralCollaborativeFiltering
from src.ml_models import FEATURE_COLS as ML_FEATURE_COLS

from torchinfo import summary as torchinfo_summary
from torchviz import make_dot

st.set_page_config(page_title="MovieMind", page_icon="🎬", layout="wide")

API_BASE_URL = os.environ.get("MOVIEMIND_API_URL", "http://127.0.0.1:8000")
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
RUNTIME_OPTIONS = {
    "Rule-only": "rule-only",
    "Local LLM": "local-llm",
    "API LLM": "api-llm",
}

OCCUPATION_LEGEND = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}


def api_get(path: str) -> Dict[str, Any]:
    resp = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def safe_api_get(path: str) -> Dict[str, Any]:
    try:
        return {"ok": True, "data": api_get(path)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def safe_api_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return {"ok": True, "data": api_post(path, payload)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@st.cache_data(show_spinner=False)
def _load_train_frame() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _local_user_summary(user_id: int) -> Dict[str, Any]:
    """Fallback summary from local training dataset when API summary is unavailable."""
    train_df = _load_train_frame()
    if train_df.empty:
        return {
            "user_id": user_id,
            "found_in_training": False,
            "rating_count_train": 0,
            "avg_rating_train": 0.0,
            "age": None,
            "occupation": None,
            "gender": None,
            "first_timestamp_train": None,
            "last_timestamp_train": None,
            "top_genres_train": [],
        }

    user_rows = train_df[train_df["userId"] == user_id]
    found = not user_rows.empty
    if not found:
        return {
            "user_id": user_id,
            "found_in_training": False,
            "rating_count_train": 0,
            "avg_rating_train": 0.0,
            "age": None,
            "occupation": None,
            "gender": None,
            "first_timestamp_train": None,
            "last_timestamp_train": None,
            "top_genres_train": [],
        }

    top_counts = (
        user_rows["genres"]
        .dropna()
        .astype(str)
        .str.split("|")
        .explode()
        .str.strip()
        .value_counts()
        .head(5)
        if "genres" in user_rows.columns
        else pd.Series(dtype=int)
    )
    top_genres = top_counts.index.tolist()
    age_val = user_rows["age"].dropna().iloc[0] if "age" in user_rows.columns and user_rows["age"].dropna().shape[0] > 0 else None
    occ_val = (
        user_rows["occupation"].dropna().iloc[0]
        if "occupation" in user_rows.columns and user_rows["occupation"].dropna().shape[0] > 0
        else None
    )
    gender_val = user_rows["gender"].dropna().iloc[0] if "gender" in user_rows.columns and user_rows["gender"].dropna().shape[0] > 0 else None
    first_ts = int(user_rows["timestamp"].min()) if "timestamp" in user_rows.columns else None
    last_ts = int(user_rows["timestamp"].max()) if "timestamp" in user_rows.columns else None
    return {
        "user_id": user_id,
        "found_in_training": True,
        "rating_count_train": int(user_rows.shape[0]),
        "avg_rating_train": round(float(user_rows["rating"].mean()), 4),
        "age": int(age_val) if pd.notna(age_val) else None,
        "occupation": int(occ_val) if pd.notna(occ_val) else None,
        "gender": str(gender_val) if pd.notna(gender_val) else None,
        "first_timestamp_train": first_ts,
        "last_timestamp_train": last_ts,
        "top_genres_train": top_genres,
        "top_genre_counts": {str(k): int(v) for k, v in top_counts.items()},
    }


@st.cache_data(show_spinner=False)
def _user_id_range() -> Dict[str, int]:
    train_df = _load_train_frame()
    if train_df.empty or "userId" not in train_df.columns:
        return {"min_user_id": 1, "max_user_id": 1}
    return {
        "min_user_id": int(train_df["userId"].min()),
        "max_user_id": int(train_df["userId"].max()),
    }


def _genre_counts_from_recommendations(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in items:
        for g in str(row.get("genres", "")).split("|"):
            g = g.strip()
            if not g:
                continue
            counts[g] = counts.get(g, 0) + 1
    return counts


def _render_taste_radar(user_counts: Dict[str, int], reco_counts: Dict[str, int]) -> None:
    labels = sorted(set(list(user_counts.keys()) + list(reco_counts.keys())))
    if not labels:
        st.info("No genre data available yet for taste map.")
        return

    user_vals = np.array([user_counts.get(k, 0) for k in labels], dtype=float)
    reco_vals = np.array([reco_counts.get(k, 0) for k in labels], dtype=float)
    user_max = user_vals.max() if user_vals.max() > 0 else 1.0
    reco_max = reco_vals.max() if reco_vals.max() > 0 else 1.0
    user_norm = user_vals / user_max
    reco_norm = reco_vals / reco_max

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    user_plot = np.concatenate([user_norm, [user_norm[0]]])
    reco_plot = np.concatenate([reco_norm, [reco_norm[0]]])

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw={"projection": "polar"})
    ax.plot(angles, user_plot, linewidth=2, label="User Taste (train)")
    ax.fill(angles, user_plot, alpha=0.2)
    ax.plot(angles, reco_plot, linewidth=2, label="Recommended Mix")
    ax.fill(angles, reco_plot, alpha=0.15)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_title("Taste Map (normalized genre profile)", pad=14, fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.10, 1.08), fontsize=7)
    fig.tight_layout(pad=0.8)
    st.pyplot(fig, clear_figure=True, use_container_width=False, width=420)


def _render_diversity_impact_panel(items: List[Dict[str, Any]], diversity_alpha: float) -> None:
    st.markdown("### Diversity Impact & Formula")
    st.latex(r"\mathrm{adjusted\_score}_i = \hat{r}_i - \alpha \cdot \mathrm{overlap}_i")
    st.caption(
        "Where r_hat is predicted rating, alpha is the diversity slider, and overlap is genre overlap with "
        "already selected recommendations during greedy reranking."
    )

    if not items:
        st.info("Run recommendations to compute diversity impact stats.")
        return

    penalties = [float(row.get("overlap_penalty") or 0.0) for row in items]
    adjusted = [float(row.get("adjusted_score") or row.get("predicted_rating") or 0.0) for row in items]
    raw = [float(row.get("predicted_rating_raw") or row.get("predicted_rating") or 0.0) for row in items]
    overlap_non_empty = sum(1 for row in items if row.get("overlap_genres"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha (diversity)", f"{diversity_alpha:.2f}")
    c2.metric("Avg Penalty", f"{(sum(penalties) / max(len(penalties), 1)):.4f}")
    c3.metric("Overlap Hits", f"{overlap_non_empty}/{len(items)}")
    c4.metric("Avg Adjusted Score", f"{(sum(adjusted) / max(len(adjusted), 1)):.4f}")

    impact_rows = [
        {
            "movie_id": row.get("movie_id"),
            "title": row.get("title"),
            "raw_score": row.get("predicted_rating_raw", row.get("predicted_rating")),
            "penalty": row.get("overlap_penalty", 0.0),
            "adjusted_score": row.get("adjusted_score", row.get("predicted_rating")),
            "overlap_genres": ", ".join(row.get("overlap_genres", [])),
        }
        for row in items
    ]
    with st.expander("Show Diversity Impact Table", expanded=False):
        st.dataframe(impact_rows, use_container_width=True)

    with st.expander("Worked Examples (term-by-term breakdown)", expanded=False):
        if not impact_rows:
            st.info("Run recommendations first to generate a real datapoint example.")
        else:
            st.markdown(
                """
**Terms in this formula**
- `raw score (r_hat)`: model’s original predicted rating before diversity adjustment.
- `alpha`: diversity strength set by slider (`0.00` to `1.00`).
- `overlap`: how much this movie’s genres overlap with already selected recommendation genres.
- `penalty`: `alpha * overlap`.
- `adjusted score`: final rerank value = `raw score - penalty`.
"""
            )
            st.latex(r"\text{adjusted\_score}=\hat{r}-\alpha\cdot overlap")

            # Show multiple datapoints so users can compare behavior.
            sample_count = min(3, len(impact_rows))
            st.markdown(f"**Examples from current result set (showing {sample_count})**")
            for i in range(sample_count):
                ex = impact_rows[i]
                raw_score = float(ex.get("raw_score") or 0.0)
                penalty = float(ex.get("penalty") or 0.0)
                adjusted_score = float(ex.get("adjusted_score") or 0.0)
                # overlap = penalty / alpha, guarded when alpha is zero.
                overlap = (penalty / diversity_alpha) if diversity_alpha > 0 else 0.0
                overlap_genres = ex.get("overlap_genres") or "None"
                st.markdown(
                    f"""
**Example {i+1}:** `{ex.get("title")}` (`id={ex.get("movie_id")}`)

- raw score (r_hat): `{raw_score:.4f}`
- alpha: `{diversity_alpha:.2f}`
- overlap (derived): `{overlap:.4f}`
- penalty (alpha * overlap): `{penalty:.4f}`
- overlap genres: `{overlap_genres}`
- adjusted score: `{adjusted_score:.4f}`
"""
                )


def render_header() -> None:
    st.title("MovieMind Web App")
    st.caption("Phase 8 UI: API-backed recommendations, explainability, model inspector, and NLP mode toggle.")
    st.write(f"API endpoint: `{API_BASE_URL}`")


def load_models() -> List[Dict[str, Any]]:
    result = safe_api_get("/models")
    if not result["ok"]:
        st.error(f"Could not load models from API: {result['error']}")
        return []
    return result["data"].get("models", [])


def recommend_page(models: List[Dict[str, Any]]) -> None:
    st.subheader("Recommend")
    if not models:
        st.info("No models available from API.")
        return

    model_map = {f"{m['display_name']} ({m['model_id']})": m["model_id"] for m in models}
    id_range = _user_id_range()

    cols = st.columns(4)
    selected_label = cols[0].selectbox("Model", list(model_map.keys()))
    user_id = cols[1].number_input(
        "User ID",
        min_value=id_range["min_user_id"],
        max_value=id_range["max_user_id"],
        value=id_range["min_user_id"],
        step=1,
    )
    top_n = cols[2].slider("Top N", min_value=1, max_value=50, value=10)
    runtime_label = cols[3].selectbox("NLP Runtime", list(RUNTIME_OPTIONS.keys()), index=1)
    with st.expander("Advanced Recommendation Controls", expanded=False):
        diversity_alpha = st.slider(
            "Diversity",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help=(
                "Alpha in adjusted_score = predicted_rating - alpha * overlap. "
                "0 = accuracy focus, higher alpha = stronger novelty/diversification."
            ),
        )
        show_diversity_debug = st.checkbox("Show Diversity Math Debug", value=False)
    st.caption(f"Available userId range in training data: `{id_range['min_user_id']} - {id_range['max_user_id']}`")
    with st.expander("New User & Field Legends", expanded=False):
        st.markdown(
            """
Current UI supports known training user IDs only.

Planned optional extension for new users:
- Collect quick onboarding preferences (top genres or sample movie likes).
- Build a temporary profile vector from demographics + preferences.
- Use popularity/diversity fallback until enough interactions are logged.
"""
        )
        st.markdown("**Gender Legend**")
        st.write({"M": "male", "F": "female"})
        st.markdown("**Occupation Legend (MovieLens 1M codes)**")
        st.dataframe(
            [{"code": code, "occupation": label} for code, label in OCCUPATION_LEGEND.items()],
            use_container_width=True,
        )

    query = st.text_input("Optional natural-language request", placeholder="e.g., top 5 action movies for user 120 with tuned model")

    # Show quick user context from training data to guide interpretation.
    user_summary = safe_api_get(f"/users/{int(user_id)}/summary")
    s = user_summary["data"] if user_summary["ok"] else _local_user_summary(int(user_id))
    if not user_summary["ok"]:
        st.info("User summary API unavailable; showing local training-data summary fallback.")
    info_cols = st.columns(4)
    info_cols[0].metric("User in Training", "Yes" if s["found_in_training"] else "No")
    info_cols[1].metric("Train Ratings", s["rating_count_train"])
    info_cols[2].metric("Avg Rating (Train)", s["avg_rating_train"])
    info_cols[3].metric("Age / Occupation", f"{s.get('age', '-')}/{s.get('occupation', '-')}")
    with st.expander("User Summary Details", expanded=False):
        st.json(s)
    if s.get("top_genre_counts"):
        st.markdown("**Taste Profile (Top Genres in Training)**")
        genre_df = pd.DataFrame(
            {"genre": list(s["top_genre_counts"].keys()), "count": list(s["top_genre_counts"].values())}
        ).set_index("genre")
        st.bar_chart(genre_df)

    st.markdown("### Actions")
    rec_btn, nlp_btn = st.columns([1.4, 1])
    with rec_btn:
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            payload = {
                "model_id": model_map[selected_label],
                "user_id": int(user_id),
                "top_n": int(top_n),
                "diversity_alpha": float(diversity_alpha),
            }
            result = safe_api_post("/recommend", payload)
            if not result["ok"]:
                st.error(result["error"])
            else:
                items = result["data"]["recommendations"]
                st.session_state["last_recommendations"] = items
                st.success(f"Returned {len(items)} recommendations.")
                display_rows = [
                    {
                        "movie_id": row.get("movie_id"),
                        "title": row.get("title"),
                        "genres": row.get("genres"),
                        "predicted_rating": row.get("predicted_rating"),
                        "reason": row.get("reason"),
                    }
                    for row in items
                ]
                st.dataframe(display_rows, use_container_width=True)
                if show_diversity_debug and items:
                    st.markdown("**Diversity Debug (score adjustment per selected item)**")
                    debug_rows = []
                    for row in items:
                        raw = float(row.get("predicted_rating_raw") or row.get("predicted_rating") or 0.0)
                        penalty = float(row.get("overlap_penalty") or 0.0)
                        adjusted = float(row.get("adjusted_score") or row.get("predicted_rating") or 0.0)
                        overlap_genres = row.get("overlap_genres") or []
                        movie_genres = [g.strip() for g in str(row.get("genres", "")).split("|") if g.strip()]
                        overlap_ratio = (penalty / diversity_alpha) if diversity_alpha > 0 else 0.0
                        debug_rows.append(
                            {
                                "movie_id": row.get("movie_id"),
                                "title": row.get("title"),
                                "predicted_rating_raw": round(raw, 4),
                                "diversity_alpha": round(float(diversity_alpha), 4),
                                "overlap_ratio": round(float(overlap_ratio), 4),
                                "overlap_count": len(overlap_genres),
                                "movie_genre_count": len(movie_genres),
                                "overlap_penalty": round(penalty, 4),
                                "adjusted_score": round(adjusted, 4),
                                "calc_check": round(raw - penalty, 4),
                                "overlap_genres": overlap_genres,
                            }
                        )
                    st.dataframe(debug_rows, use_container_width=True)

    with nlp_btn:
        if st.button("Parse Query (NLP)", use_container_width=True) and query.strip():
            payload = {"query": query, "runtime_mode": RUNTIME_OPTIONS[runtime_label]}
            result = safe_api_post("/nlp/query", payload)
            if not result["ok"]:
                st.error(result["error"])
            else:
                data = result["data"]
                st.write("Parsed intent")
                st.json(data)
                parsed_model = data.get("model_hint") or model_map[selected_label]
                parsed_user = int(data.get("filters", {}).get("user_id", user_id))
                parsed_top_n = int(data.get("filters", {}).get("top_n", top_n))
                rec_result = safe_api_post(
                    "/recommend",
                    {
                        "model_id": parsed_model,
                        "user_id": parsed_user,
                        "top_n": parsed_top_n,
                        "diversity_alpha": float(diversity_alpha),
                    },
                )
                if rec_result["ok"]:
                    st.write("Recommendations from parsed intent")
                    parsed_items = rec_result["data"]["recommendations"]
                    st.session_state["last_recommendations"] = parsed_items
                    parsed_display = [
                        {
                            "movie_id": row.get("movie_id"),
                            "title": row.get("title"),
                            "genres": row.get("genres"),
                            "predicted_rating": row.get("predicted_rating"),
                            "reason": row.get("reason"),
                        }
                        for row in parsed_items
                    ]
                    st.dataframe(parsed_display, use_container_width=True)
                    if show_diversity_debug and parsed_items:
                        st.markdown("**Diversity Debug (score adjustment per selected item)**")
                        debug_rows = []
                        for row in parsed_items:
                            raw = float(row.get("predicted_rating_raw") or row.get("predicted_rating") or 0.0)
                            penalty = float(row.get("overlap_penalty") or 0.0)
                            adjusted = float(row.get("adjusted_score") or row.get("predicted_rating") or 0.0)
                            overlap_genres = row.get("overlap_genres") or []
                            movie_genres = [g.strip() for g in str(row.get("genres", "")).split("|") if g.strip()]
                            overlap_ratio = (penalty / diversity_alpha) if diversity_alpha > 0 else 0.0
                            debug_rows.append(
                                {
                                    "movie_id": row.get("movie_id"),
                                    "title": row.get("title"),
                                    "predicted_rating_raw": round(raw, 4),
                                    "diversity_alpha": round(float(diversity_alpha), 4),
                                    "overlap_ratio": round(float(overlap_ratio), 4),
                                    "overlap_count": len(overlap_genres),
                                    "movie_genre_count": len(movie_genres),
                                    "overlap_penalty": round(penalty, 4),
                                    "adjusted_score": round(adjusted, 4),
                                    "calc_check": round(raw - penalty, 4),
                                    "overlap_genres": overlap_genres,
                                }
                            )
                        st.dataframe(debug_rows, use_container_width=True)
                else:
                    st.error(rec_result["error"])

    with st.expander("Taste Map (Game-Style Radar)", expanded=False):
        st.caption(
            "Normalization note: radar values are scaled to [0,1] per profile. "
            "1.0 means strongest genre in that profile, not absolute rating/count."
        )
        st.caption(
            "Interpretation note: this radar is genre-frequency based (from training history and "
            "current recommendations), not an embedding-space visualization."
        )
        user_counts = s.get("top_genre_counts", {}) or {}
        reco_counts = _genre_counts_from_recommendations(st.session_state.get("last_recommendations", []))
        _render_taste_radar(user_counts=user_counts, reco_counts=reco_counts)
    _render_diversity_impact_panel(st.session_state.get("last_recommendations", []), diversity_alpha)


def inspector_page(models: List[Dict[str, Any]]) -> None:
    st.subheader("Model Inspector")
    if not models:
        st.info("No models available from API.")
        return
    model_map = {f"{m['display_name']} ({m['model_id']})": m["model_id"] for m in models}
    selected = st.selectbox("Inspect model", list(model_map.keys()), key="inspector_model")
    info_result = safe_api_get(f"/models/{model_map[selected]}/info")
    if not info_result["ok"]:
        st.error(info_result["error"])
        return
    info = info_result["data"]
    left, right = st.columns(2)
    with left:
        st.metric("Model Family", info["family"])
        st.metric("Availability", "Yes" if info["available"] else "No")
        st.write("Artifact path")
        st.code(str(info.get("artifact_path")))
    with right:
        st.write("Training metrics")
        st.json(info.get("metrics", {}))
        st.write("Inspector fields")
        st.json(info.get("inspector", {}))
    st.write("Parameters")
    st.json(info.get("params", {}))


def system_page() -> None:
    st.subheader("System")
    health = safe_api_get("/health")
    if health["ok"]:
        st.success("API is healthy")
        st.json(health["data"])
    else:
        st.error(f"API health check failed: {health['error']}")
    st.markdown(
        """
        **Run commands**
        - API: `source .venv/bin/activate && uvicorn src.api.app:app --host 127.0.0.1 --port 8000`
        - UI: `source .venv/bin/activate && streamlit run app/streamlit_app.py`
        """
    )


def embedding_space_page() -> None:
    st.subheader("Embedding Space")
    st.caption("Visualizes learned user embedding structure from Phase 6 analysis artifacts.")
    pca_plot = EVIDENCE_DIR / "phase6" / "user_embeddings_pca_2d.png"
    tsne_plot = EVIDENCE_DIR / "phase6" / "user_embeddings_tsne_2d_sample.png"
    pca_csv = EVIDENCE_DIR / "phase6" / "user_embeddings_pca_2d.csv"
    raw_csv = EVIDENCE_DIR / "phase6" / "user_embeddings_raw.csv"
    tsne_status = EVIDENCE_DIR / "phase6" / "tsne_status.txt"

    left, right = st.columns(2)
    with left:
        st.markdown("**PCA (always generated in Phase 6)**")
        if pca_plot.exists():
            st.image(str(pca_plot), use_container_width=True)
        else:
            st.warning("PCA plot not found. Run `python src/post_analysis.py` to generate it.")
    with right:
        st.markdown("**t-SNE (optional/safe-mode dependent)**")
        if tsne_plot.exists():
            st.image(str(tsne_plot), use_container_width=True)
        else:
            st.info("t-SNE plot not found. This is normal if safe mode skipped t-SNE.")
            if tsne_status.exists():
                st.code(tsne_status.read_text())

    st.markdown("**Artifact availability**")
    st.write(
        {
            "pca_csv": pca_csv.exists(),
            "raw_embeddings_csv": raw_csv.exists(),
            "tsne_plot": tsne_plot.exists(),
            "tsne_status_file": tsne_status.exists(),
        }
    )

    st.markdown("### How to Read These Plots")
    st.info(
        "Each point represents one user embedding learned by the DL model. "
        "Points closer together generally indicate more similar learned preference patterns."
    )
    st.markdown(
        """
- **PCA plot**: preserves broad/global structure; good for stable high-level view.
- **t-SNE plot**: emphasizes local neighborhoods/clusters; useful for fine-grained grouping intuition.
- **Important caveat**: 2D projection is an approximation of high-dimensional space.
- **What this supports**: qualitative evidence that embeddings capture user taste structure.
- **What it does not replace**: quantitative evaluation metrics like RMSE/MAE/Precision@K.
"""
    )


def _render_markdown_file(path: Path, missing_msg: str) -> None:
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.warning(missing_msg)


def _file_status_row(path: Path, label: str) -> Dict[str, Any]:
    return {
        "artifact": label,
        "path": str(path.relative_to(PROJECT_ROOT)) if path.exists() else str(path.relative_to(PROJECT_ROOT)),
        "exists": path.exists(),
    }


def _notebook_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"notebook": str(path.relative_to(PROJECT_ROOT)), "exists": False}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cells = raw.get("cells", [])
        code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
        markdown_cells = sum(1 for c in cells if c.get("cell_type") == "markdown")
        return {
            "notebook": str(path.relative_to(PROJECT_ROOT)),
            "exists": True,
            "total_cells": len(cells),
            "code_cells": code_cells,
            "markdown_cells": markdown_cells,
        }
    except Exception as exc:
        return {
            "notebook": str(path.relative_to(PROJECT_ROOT)),
            "exists": True,
            "parse_error": str(exc),
        }


def _parse_best_dl_params(path: Path) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if not path.exists():
        return params
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if ":" not in clean:
            continue
        key, value = [x.strip() for x in clean.split(":", 1)]
        if key.lower() == "best rmse":
            params["best_rmse"] = value
        elif key in {"embedding_dim", "n_layers"} or key.startswith("n_units_l"):
            try:
                params[key] = int(float(value))
            except ValueError:
                params[key] = value
        else:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value
    return params


def _load_user_movie_dims() -> Dict[str, int]:
    train_path = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test_features.csv"
    if not train_path.exists() or not test_path.exists():
        return {}
    train_df = pd.read_csv(train_path, usecols=["userId", "movieId"])
    test_df = pd.read_csv(test_path, usecols=["userId", "movieId"])
    return {
        "num_users": int(max(train_df["userId"].max(), test_df["userId"].max())),
        "num_movies": int(max(train_df["movieId"].max(), test_df["movieId"].max())),
    }


def _build_raw_ncf_model():
    model_path = PROJECT_ROOT / "models" / "ncf_model.pt"
    dims = _load_user_movie_dims()
    if not model_path.exists() or not dims:
        return None
    model = NeuralCollaborativeFiltering(
        num_users=dims["num_users"],
        num_movies=dims["num_movies"],
        num_dense_features=6,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def _build_tuned_ncf_model():
    model_path = PROJECT_ROOT / "models" / "ncf_tuned_best.pt"
    dims = _load_user_movie_dims()
    params = _parse_best_dl_params(PROJECT_ROOT / "models" / "best_dl_params.txt")
    if not model_path.exists() or not dims or not params:
        return None
    state_dict = torch.load(model_path, map_location="cpu")

    # First attempt: build from best_dl_params.txt (expected path).
    n_layers = int(params.get("n_layers", 3))
    hidden_layers = [int(params.get(f"n_units_l{i}", 128)) for i in range(n_layers)]
    embedding_dim = int(params.get("embedding_dim", 64))
    dropout_rate = float(params.get("dropout_rate", 0.2))

    def _build_model(emb_dim: int, hidden: List[int], drop: float):
        return DynamicNCF(
            num_users=dims["num_users"],
            num_movies=dims["num_movies"],
            num_dense_features=6,
            embedding_dim=emb_dim,
            hidden_layers=hidden,
            dropout_rate=drop,
        )

    model = _build_model(embedding_dim, hidden_layers, dropout_rate)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Fallback: infer architecture from checkpoint weights if params file and checkpoint drift.
        emb_from_ckpt = int(state_dict["user_embedding.weight"].shape[1])
        linear_keys = sorted(
            [k for k in state_dict.keys() if k.startswith("fc_layers.") and k.endswith(".weight")],
            key=lambda x: int(x.split(".")[1]),
        )
        if not linear_keys:
            raise
        inferred_hidden = [int(state_dict[k].shape[0]) for k in linear_keys[:-1]]
        model = _build_model(emb_from_ckpt, inferred_hidden, dropout_rate)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def _render_torchinfo_summary(model, model_label: str) -> None:
    st.markdown(f"**{model_label} Layer Summary (`torchinfo`)**")
    user_idx = torch.tensor([0], dtype=torch.long)
    movie_idx = torch.tensor([0], dtype=torch.long)
    dense = torch.tensor(np.zeros((1, 6), dtype=np.float32))
    try:
        summary_obj = torchinfo_summary(model, input_data=(user_idx, movie_idx, dense), verbose=0)
        st.code(str(summary_obj))
    except Exception as exc:
        st.warning(f"Could not generate torchinfo summary: {exc}")


def _draw_ncf_architecture_schematic(model_name: str, embedding_dim: int, hidden_layers: List[int]) -> None:
    """Draw a readable layer-flow diagram for NCF architecture."""
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    labels = [
        "User ID",
        f"User Embedding ({embedding_dim})",
        "Movie ID",
        f"Movie Embedding ({embedding_dim})",
        "Dense Features (6)",
        f"Concat ({embedding_dim * 2 + 6})",
    ] + [f"Dense ({u})" for u in hidden_layers] + ["Rating Output (1)"]

    x_positions = np.linspace(0.04, 0.96, len(labels))
    y = 0.5
    for i, (x, label) in enumerate(zip(x_positions, labels)):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f2f2f2", edgecolor="#666666"),
        )
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.02, y),
                xytext=(x + 0.02, y),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1),
            )
    ax.set_title(f"{model_name} Architecture Flow", fontsize=10)
    st.pyplot(fig, clear_figure=True)


def _render_torchviz_graph(model, model_label: str) -> None:
    """Renders computation graph image from live forward pass."""
    st.markdown(f"**{model_label} Computation Graph (`torchviz`)**")
    user_idx = torch.tensor([0], dtype=torch.long)
    movie_idx = torch.tensor([0], dtype=torch.long)
    dense = torch.tensor(np.zeros((1, 6), dtype=np.float32))
    try:
        output = model(user_idx, movie_idx, dense)
        dot = make_dot(output, params=dict(model.named_parameters()))
        png_bytes = dot.pipe(format="png")
        # Keep graph readable in demos without taking over the whole page.
        st.image(png_bytes, width=900)
    except Exception as exc:
        st.info(
            "Torchviz graph could not be rendered on this environment. "
            "Fallback schematic is shown above."
        )
        st.code(str(exc))


def _first_markdown_heading(path: Path) -> str:
    if not path.exists():
        return "No summary heading available."
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return "No markdown heading found."


def lifecycle_evidence_page() -> None:
    st.subheader("Lifecycle Evidence & Decisions")
    st.caption("Exact file-backed evidence and decisions across all completed phases.")

    st.markdown("### Human-Readable Evidence Summary")
    phase_dirs = sorted([p for p in EVIDENCE_DIR.glob("phase*") if p.is_dir()], key=lambda p: p.name)
    if phase_dirs:
        summary_rows = []
        for phase_dir in phase_dirs:
            md_files = sorted(phase_dir.glob("*.md"))
            all_files = sorted([p for p in phase_dir.iterdir() if p.is_file()])
            summary_rows.append(
                {
                    "phase": phase_dir.name,
                    "artifact_count": len(all_files),
                    "markdown_notes": len(md_files),
                    "top_note": _first_markdown_heading(md_files[0]) if md_files else "No markdown note",
                }
            )
        st.dataframe(summary_rows, use_container_width=True)
    else:
        st.info("No phase evidence folders found.")

    st.markdown("### Canonical Project Documents (Exact Content)")
    doc_paths = [
        ("README.md", PROJECT_ROOT / "README.md"),
        ("CONTEXT_HANDOVER.md", PROJECT_ROOT / "CONTEXT_HANDOVER.md"),
        ("AI_CONCEPTS_WIKI.md", PROJECT_ROOT / "AI_CONCEPTS_WIKI.md"),
        ("WEBAPP_AGENT_WIKI.md", PROJECT_ROOT / "WEBAPP_AGENT_WIKI.md"),
        ("PROGRESS_TRACKER.md", PROJECT_ROOT / "PROGRESS_TRACKER.md"),
    ]
    for title, path in doc_paths:
        with st.expander(title, expanded=False):
            _render_markdown_file(path, f"{title} not found.")

    st.markdown("### Evidence by Phase (All Found in Repo)")
    if not phase_dirs:
        st.warning("No phase directories found under evidence/.")
    for phase_dir in phase_dirs:
        with st.expander(f"{phase_dir.name}", expanded=False):
            md_files = sorted(phase_dir.glob("*.md"))
            other_files = sorted([p for p in phase_dir.iterdir() if p.is_file() and p.suffix.lower() != ".md"])
            if md_files:
                st.markdown("**Markdown Notes (Exact Content)**")
                for md_path in md_files:
                    st.markdown(f"#### `{md_path.name}`")
                    _render_markdown_file(md_path, f"{md_path.name} missing.")
            else:
                st.info("No markdown notes found in this phase folder.")
            if other_files:
                st.markdown("**Other Artifacts in Folder**")
                st.dataframe(
                    [
                        {
                            "file": p.name,
                            "path": str(p.relative_to(PROJECT_ROOT)),
                            "size_kb": round(p.stat().st_size / 1024.0, 2),
                        }
                        for p in other_files
                    ],
                    use_container_width=True,
                )

    st.markdown("### Other Evidence (Cross-Phase)")
    other_evidence_md = sorted(
        [
            p
            for p in EVIDENCE_DIR.glob("*.md")
            if p.is_file()
        ],
        key=lambda p: p.name,
    )
    if other_evidence_md:
        for md_path in other_evidence_md:
            with st.expander(md_path.name, expanded=False):
                _render_markdown_file(md_path, f"{md_path.name} not found.")
    else:
        st.info("No root-level markdown evidence files found.")


def model_visualizers_page() -> None:
    st.subheader("Model Visualizers")
    st.caption("Visual summary of neural and traditional ML model behavior from training and post-analysis artifacts.")
    graph_view_mode = st.radio(
        "NN Graph Detail",
        options=["Compact", "Full"],
        horizontal=True,
        help="Compact keeps demo clean (schematic + layer summary). Full also renders torchviz computation graph.",
    )

    st.markdown("### Neural Network Visualizer (NCF Raw + Tuned)")
    dl_loss_plot = PROJECT_ROOT / "models" / "dl_loss_curve.png"
    tuned_params_path = PROJECT_ROOT / "models" / "best_dl_params.txt"

    col_nn_top_left, col_nn_top_right = st.columns(2)
    with col_nn_top_left:
        st.markdown("**DL Training Curve**")
        if dl_loss_plot.exists():
            st.image(str(dl_loss_plot), use_container_width=True)
        else:
            st.warning("DL loss curve not found. Run `python src/dl_model.py` to generate it.")
    with col_nn_top_right:
        st.markdown("**Tuned Architecture Snapshot**")
        tuned_params = _parse_best_dl_params(tuned_params_path)
        if tuned_params:
            n_layers = int(tuned_params.get("n_layers", 0))
            hidden_layers = [tuned_params.get(f"n_units_l{i}") for i in range(n_layers)]
            architecture = {
                "embedding_dim": tuned_params.get("embedding_dim"),
                "hidden_layers": hidden_layers,
                "dropout_rate": tuned_params.get("dropout_rate"),
                "learning_rate": tuned_params.get("learning_rate"),
                "best_rmse": tuned_params.get("best_rmse"),
            }
            st.json(architecture)
        else:
            st.warning("Tuned params file not found. Run `python src/tune_dl.py` to generate it.")

    raw_model = _build_raw_ncf_model()
    tuned_model = _build_tuned_ncf_model()

    col_raw, col_tuned = st.columns(2)
    with col_raw:
        st.markdown("#### Raw NCF")
        if raw_model is None:
            st.warning("Raw NCF model artifact not found or dimensions unavailable.")
        else:
            raw_hidden = [128, 64, 32]
            st.write(
                {
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_layers": raw_hidden,
                    "source": "models/ncf_model.pt",
                }
            )
            st.latex(
                r"\hat{r}_{ui}=f_{\theta}\left([\mathrm{Emb}_u,\mathrm{Emb}_i,\mathrm{DenseFeatures}]\right)"
            )
            _draw_ncf_architecture_schematic("Raw NCF", EMBEDDING_DIM, raw_hidden)
            _render_torchinfo_summary(raw_model, "Raw NCF")
            if graph_view_mode == "Full":
                _render_torchviz_graph(raw_model, "Raw NCF")
    with col_tuned:
        st.markdown("#### Tuned NCF")
        if tuned_model is None:
            st.warning("Tuned NCF model artifact/params not found.")
        else:
            tuned_params = _parse_best_dl_params(tuned_params_path)
            tuned_hidden = [tuned_params.get(f"n_units_l{i}") for i in range(int(tuned_params.get("n_layers", 0)))]
            st.write(
                {
                    "embedding_dim": tuned_params.get("embedding_dim"),
                    "n_layers": tuned_params.get("n_layers"),
                    "hidden_layers": tuned_hidden,
                    "source": "models/ncf_tuned_best.pt",
                }
            )
            st.latex(
                r"\hat{r}_{ui}=f_{\theta}^{tuned}\left([\mathrm{Emb}_u,\mathrm{Emb}_i,\mathrm{DenseFeatures}]\right)"
            )
            _draw_ncf_architecture_schematic(
                "Tuned NCF",
                int(tuned_params.get("embedding_dim", 64)),
                [int(x) for x in tuned_hidden if x is not None],
            )
            _render_torchinfo_summary(tuned_model, "Tuned NCF")
            if graph_view_mode == "Full":
                _render_torchviz_graph(tuned_model, "Tuned NCF")

    st.markdown("### Traditional ML Visualizers")
    gb_plot = EVIDENCE_DIR / "phase6" / "gradient_boosting_feature_importance.png"
    gb_csv = EVIDENCE_DIR / "phase6" / "gradient_boosting_feature_importance.csv"
    ml_log = PROJECT_ROOT / "models" / "ml_training_log.txt"
    lr_path = PROJECT_ROOT / "models" / "linear_regression.pkl"
    rf_path = PROJECT_ROOT / "models" / "random_forest.pkl"

    col_ml_a, col_ml_b = st.columns(2)
    with col_ml_a:
        st.markdown("**Gradient Boosting Feature Importance**")
        if gb_plot.exists():
            st.image(str(gb_plot), use_container_width=True)
        else:
            st.warning("GB feature importance plot not found. Run `python src/post_analysis.py` to generate it.")
    with col_ml_b:
        st.markdown("**Feature Importance Table**")
        if gb_csv.exists():
            st.dataframe(
                [
                    {"feature": row.split(",")[0], "importance": row.split(",")[1]}
                    for row in gb_csv.read_text(encoding="utf-8").splitlines()[1:]
                    if "," in row
                ],
                use_container_width=True,
            )
        else:
            st.info("Feature-importance CSV not found yet.")

    st.markdown("### Other ML Models (Linear Regression + Random Forest)")
    col_other_ml_left, col_other_ml_right = st.columns(2)
    with col_other_ml_left:
        st.markdown("**Linear Regression Coefficients**")
        if lr_path.exists():
            try:
                lr_model = joblib.load(lr_path)
                coef_rows = [
                    {"feature": feat, "coefficient": float(coef)}
                    for feat, coef in zip(ML_FEATURE_COLS, lr_model.coef_)
                ]
                st.dataframe(coef_rows, use_container_width=True)
                st.bar_chart(pd.DataFrame(coef_rows).set_index("feature"))
                eq_terms = [f"({float(c):+.4f}*{f})" for f, c in zip(ML_FEATURE_COLS, lr_model.coef_)]
                eq_text = f"rating_hat = {float(lr_model.intercept_):+.4f} " + " ".join(eq_terms)
                st.code(eq_text)
                st.latex(
                    r"\hat{y}=\beta_0+\sum_{j=1}^{p}\beta_jx_j"
                )
            except Exception as exc:
                st.warning(f"Could not load linear regression model: {exc}")
        else:
            st.info("Linear regression artifact not found.")

    with col_other_ml_right:
        st.markdown("**Random Forest Feature Importance**")
        if rf_path.exists():
            try:
                rf_model = joblib.load(rf_path)
                fi_rows = [
                    {"feature": feat, "importance": float(val)}
                    for feat, val in zip(ML_FEATURE_COLS, rf_model.feature_importances_)
                ]
                fi_rows = sorted(fi_rows, key=lambda x: x["importance"], reverse=True)
                st.dataframe(fi_rows, use_container_width=True)
                st.bar_chart(pd.DataFrame(fi_rows).set_index("feature"))
                st.markdown("**Random Forest Tree Snapshot (single tree from ensemble)**")
                fig, ax = plt.subplots(figsize=(11, 6))
                sktree.plot_tree(
                    rf_model.estimators_[0],
                    max_depth=2,
                    feature_names=ML_FEATURE_COLS,
                    filled=True,
                    rounded=True,
                    fontsize=7,
                    ax=ax,
                )
                st.pyplot(fig, clear_figure=True)
            except Exception as exc:
                st.warning(f"Could not load random forest model: {exc}")
        else:
            st.info("Random forest artifact not found.")

    st.markdown("**ML Model Comparison Log (RMSE/MAE/Time)**")
    if ml_log.exists():
        st.code(ml_log.read_text(encoding="utf-8"))
    else:
        st.info("ML training log not found. Run `python src/ml_models.py` to generate it.")


def ai_concepts_page() -> None:
    st.subheader("AI Concepts")
    st.caption("Theory and math tied directly to MovieMind objectives.")

    st.markdown("### Objective 1: Accurate Rating Prediction")
    st.write("We estimate user u's rating for movie i.")
    st.latex(r"\hat{r}_{ui}")
    st.write("For linear models:")
    st.latex(r"\hat{y}=\beta_0+\sum_{j=1}^{p}\beta_jx_j")
    st.write("For embedding-based NCF:")
    st.latex(r"\hat{r}_{ui}=f_{\theta}\left([\mathrm{Emb}_u,\mathrm{Emb}_i,\mathrm{DenseFeatures}]\right)")

    st.markdown("### Objective 2: Measure Prediction Quality")
    st.write("MAE (average absolute error):")
    st.latex(r"\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}\left|y_i-\hat{y}_i\right|")
    st.write("RMSE (penalizes large errors more):")
    st.latex(r"\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}")

    st.markdown("### Objective 3: Ranking Quality for Recommendations")
    st.write("For top-K recommendation quality:")
    st.write("Precision@K")
    st.latex(r"\mathrm{Precision@K}=\frac{|Rel \cap TopK|}{K}")
    st.write("Recall@K")
    st.latex(r"\mathrm{Recall@K}=\frac{|Rel \cap TopK|}{|Rel|}")
    st.write("where Rel is set of truly relevant items.")

    st.markdown("### Objective 4: Train Robust Neural Model")
    st.write("NCF training minimizes MSE loss:")
    st.latex(r"\mathcal{L}_{\mathrm{MSE}}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2")
    st.write("Parameters are updated with Adam optimizer:")
    st.latex(r"\theta \leftarrow \theta - \alpha \cdot \nabla_{\theta}\mathcal{L}")
    st.write("with adaptive moments in Adam for stable convergence.")

    st.markdown("### Objective 5: Leakage-Safe Evaluation")
    st.markdown(
        """
MovieMind uses **time-based split** (past -> train, future -> test), not random split.

Why this matters:
- Prevents training on future interactions.
- Keeps champion model comparison realistic.
- Aligns offline evaluation with real deployment behavior.
"""
    )

    st.markdown("### Objective 6: Explainability and Representation")
    st.write("For tree models (GB/RF), feature importance estimates contribution strength:")
    st.latex(r"\mathrm{Importance}(x_j)\propto \text{total impurity/loss reduction from splits using }x_j")
    st.write("For embeddings, dimensionality reduction helps interpretation:")
    st.write("- PCA: linear projection maximizing variance.")
    st.write("- t-SNE: nonlinear neighborhood-preserving map for cluster intuition.")

    with st.expander("Open Full Learning Wiki (AI_CONCEPTS_WIKI.md)", expanded=False):
        concepts_path = PROJECT_ROOT / "AI_CONCEPTS_WIKI.md"
        if concepts_path.exists():
            st.markdown(concepts_path.read_text(encoding="utf-8"))
        else:
            st.warning("AI_CONCEPTS_WIKI.md not found.")


def main() -> None:
    render_header()
    models = load_models()
    tab_reco, tab_inspect, tab_embed, tab_visual, tab_evidence, tab_concepts, tab_system = st.tabs(
        [
            "Recommend",
            "Model Inspector",
            "Embedding Space",
            "Model Visualizers",
            "Lifecycle Evidence",
            "AI Concepts",
            "System",
        ]
    )
    with tab_reco:
        recommend_page(models)
    with tab_inspect:
        inspector_page(models)
    with tab_embed:
        embedding_space_page()
    with tab_visual:
        model_visualizers_page()
    with tab_evidence:
        lifecycle_evidence_page()
    with tab_concepts:
        ai_concepts_page()
    with tab_system:
        system_page()


if __name__ == "__main__":
    main()

