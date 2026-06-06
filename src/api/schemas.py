from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class ModelSummary(BaseModel):
    model_id: str
    display_name: str
    family: str
    artifact_path: Optional[str] = None
    available: bool


class ModelsResponse(BaseModel):
    models: List[ModelSummary]


class ModelInfoResponse(BaseModel):
    model_id: str
    display_name: str
    family: str
    artifact_path: Optional[str] = None
    available: bool
    params: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    inspector: Dict[str, Any] = Field(default_factory=dict)


class UserSummaryResponse(BaseModel):
    user_id: int
    found_in_training: bool
    rating_count_train: int
    avg_rating_train: float
    age: Optional[int] = None
    occupation: Optional[int] = None
    gender: Optional[str] = None
    first_timestamp_train: Optional[int] = None
    last_timestamp_train: Optional[int] = None
    top_genres_train: List[str] = Field(default_factory=list)
    top_genre_counts: Dict[str, int] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    model_id: str = Field(..., description="Model key from /models")
    user_id: int = Field(..., ge=1)
    movie_id: int = Field(..., ge=1)


class PredictResponse(BaseModel):
    model_id: str
    user_id: int
    movie_id: int
    predicted_rating: float
    clipped_to_rating_scale: bool


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    predicted_rating: float
    reason: Optional[str] = None
    predicted_rating_raw: Optional[float] = None
    overlap_penalty: Optional[float] = None
    adjusted_score: Optional[float] = None
    overlap_genres: List[str] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    model_id: str
    user_id: int = Field(..., ge=1)
    top_n: int = Field(10, ge=1, le=100)
    diversity_alpha: float = Field(0.0, ge=0.0, le=1.0)


class RecommendResponse(BaseModel):
    model_id: str
    user_id: int
    top_n: int
    recommendations: List[RecommendationItem]


class NLPQueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    runtime_mode: str = Field("local-llm", pattern="^(local-llm|api-llm)$")


class NLPQueryResponse(BaseModel):
    runtime_mode: str
    parsed_by: str
    confidence: float
    intent: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    model_hint: Optional[str] = None
    explanation: str


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    max_turns: int = Field(8, ge=1, le=24)
    user_id: Optional[int] = Field(None, ge=1)
    model_id: Optional[str] = None
    top_n: Optional[int] = Field(None, ge=1, le=100)
    stop_after_recommendations: bool = True


class AgentQueryResponse(BaseModel):
    """Multi-step Ollama tool agent: chains list_models / user_summary / recommendations."""

    final_message: str
    recommendations: Optional[List[Dict[str, Any]]] = None
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    turns_used: int = 0
    model: Optional[str] = None
    error: Optional[str] = None

