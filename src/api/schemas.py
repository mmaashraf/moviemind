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


class RecommendRequest(BaseModel):
    model_id: str
    user_id: int = Field(..., ge=1)
    top_n: int = Field(10, ge=1, le=100)


class RecommendResponse(BaseModel):
    model_id: str
    user_id: int
    top_n: int
    recommendations: List[RecommendationItem]


class NLPQueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    runtime_mode: str = Field("rule-only", pattern="^(rule-only|local-llm|api-llm)$")


class NLPQueryResponse(BaseModel):
    runtime_mode: str
    parsed_by: str
    confidence: float
    intent: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    model_hint: Optional[str] = None
    explanation: str

