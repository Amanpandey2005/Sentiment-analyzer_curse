"""Pydantic request and response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictRequest(BaseModel):
    review_text: str = Field(..., min_length=1, max_length=10_000)


class PredictionResponse(BaseModel):
    review_text: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    summary: str


class BatchPredictRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=1_000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    database: str
    redis: str
    model_loaded: bool


class AnalyticsResponse(BaseModel):
    total_predictions: int
    sentiment_distribution: dict[str, int]
    average_confidence: float
    daily_trends: list[dict[str, int | str]]
    top_positive_reviews: list[PredictionResponse]
    top_negative_reviews: list[PredictionResponse]
    generated_at: datetime
