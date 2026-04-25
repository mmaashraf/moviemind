import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request

from .model_registry import ModelRegistry
from .nlp import parse_query
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    ModelsResponse,
    NLPQueryRequest,
    NLPQueryResponse,
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    UserSummaryResponse,
)

app = FastAPI(title="MovieMind API", version="0.1.0")
registry = ModelRegistry()
logger = logging.getLogger("moviemind.api")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    # Generic middleware for traceability on every API request.
    # Adds request metadata to response headers and logs latency.
    request_id = str(uuid.uuid4())[:8]
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    response = await call_next(request)
    ended_at = datetime.now(timezone.utc).isoformat()
    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Started-At"] = started_at
    response.headers["X-Processed-At"] = ended_at
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    logger.info(
        "request id=%s path=%s method=%s status=%s started_at=%s ended_at=%s latency_ms=%.2f",
        request_id,
        request.url.path,
        request.method,
        response.status_code,
        started_at,
        ended_at,
        latency_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Simple liveness probe used by UI/system checks.
    logger.info("health_check called")
    return HealthResponse(status="ok", service="moviemind-api")


@app.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    # Lists all model options so UI can populate the selector.
    logger.info("models_list called")
    return ModelsResponse(
        models=[
            {
                "model_id": m.model_id,
                "display_name": m.display_name,
                "family": m.family,
                "artifact_path": m.artifact_path,
                "available": m.available,
            }
            for m in registry.list_models()
        ]
    )


@app.get("/models/{model_id}/info", response_model=ModelInfoResponse)
def model_info(model_id: str) -> ModelInfoResponse:
    # Returns moderate inspector payload for one model.
    logger.info("model_info called model_id=%s", model_id)
    try:
        return ModelInfoResponse(**registry.model_info(model_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/users/{user_id}/summary", response_model=UserSummaryResponse)
def user_summary(user_id: int) -> UserSummaryResponse:
    # Returns user profile and training interaction summary for UI context.
    logger.info("user_summary called user_id=%s", user_id)
    return UserSummaryResponse(**registry.user_summary(user_id))


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    # Predicts one user-movie rating using selected model.
    logger.info(
        "predict called model_id=%s user_id=%s movie_id=%s",
        payload.model_id,
        payload.user_id,
        payload.movie_id,
    )
    try:
        value, clipped = registry.predict(payload.model_id, payload.user_id, payload.movie_id)
        return PredictResponse(
            model_id=payload.model_id,
            user_id=payload.user_id,
            movie_id=payload.movie_id,
            predicted_rating=round(float(value), 4),
            clipped_to_rating_scale=clipped,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    # Main recommendation endpoint used by Streamlit Recommend tab.
    logger.info(
        "recommend called model_id=%s user_id=%s top_n=%s diversity_alpha=%.2f",
        payload.model_id,
        payload.user_id,
        payload.top_n,
        payload.diversity_alpha,
    )
    try:
        recs = registry.recommend(payload.model_id, payload.user_id, payload.top_n, payload.diversity_alpha)
        return RecommendResponse(
            model_id=payload.model_id,
            user_id=payload.user_id,
            top_n=payload.top_n,
            recommendations=recs,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/nlp/query", response_model=NLPQueryResponse)
def nlp_query(payload: NLPQueryRequest) -> NLPQueryResponse:
    # Parses natural-language input into structured intent/filters.
    # Recommendation retrieval is done in a separate /recommend call.
    logger.info(
        "nlp_query called runtime_mode=%s query_preview=%s",
        payload.runtime_mode,
        payload.query[:80].replace("\n", " "),
    )
    mode, parsed = parse_query(payload.query, payload.runtime_mode)
    logger.info(
        "nlp_query parsed runtime_mode=%s parsed_by=%s confidence=%.2f intent=%s model_hint=%s",
        mode,
        parsed.get("parsed_by"),
        float(parsed.get("confidence", 0.0)),
        parsed.get("intent"),
        parsed.get("model_hint"),
    )
    return NLPQueryResponse(runtime_mode=mode, **parsed)

