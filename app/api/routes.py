"""API route definitions."""

from __future__ import annotations

import io

import pandas as pd
import redis
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.security import create_access_token, get_current_subject
from app.core_config import get_settings
from app.database.session import get_db
from app.middleware.rate_limit import RateLimiter
from app.models.schemas import (
    AnalyticsResponse,
    BatchPredictionResponse,
    BatchPredictRequest,
    HealthResponse,
    PredictionResponse,
    PredictRequest,
    TokenResponse,
)
from app.monitoring.metrics import metrics_response
from app.services.model_service import SentimentModelService, get_model_service
from app.services.prediction_service import get_analytics, predict_and_store, predict_dataframe

router = APIRouter()
rate_limiter = RateLimiter()


def enforce_rate_limit(request: Request) -> None:
    rate_limiter.check(request)


@router.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def token(username: str = "demo") -> TokenResponse:
    """Issue a demo JWT token for portfolio deployments."""

    return TokenResponse(access_token=create_access_token(username))


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(db: Session = Depends(get_db)) -> HealthResponse:
    """Return dependency health."""

    database = "ok"
    redis_status = "ok"
    settings = get_settings()
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        database = "error"
    try:
        redis.Redis.from_url(settings.redis_url).ping()
    except redis.RedisError:
        redis_status = "error"
    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        database=database,
        redis=redis_status,
        model_loaded=settings.model_path.exists(),
    )


@router.get("/metrics", include_in_schema=False)
def metrics():
    return metrics_response()


@router.post("/predict", response_model=PredictionResponse, tags=["predictions"])
def predict(
    payload: PredictRequest,
    request: Request,
    db: Session = Depends(get_db),
    model: SentimentModelService = Depends(get_model_service),
    _: str | None = Depends(get_current_subject),
) -> PredictionResponse:
    enforce_rate_limit(request)
    return predict_and_store(payload.review_text, model, db)


@router.post("/batch-predict", response_model=BatchPredictionResponse, tags=["predictions"])
def batch_predict(
    payload: BatchPredictRequest,
    request: Request,
    db: Session = Depends(get_db),
    model: SentimentModelService = Depends(get_model_service),
    _: str | None = Depends(get_current_subject),
) -> BatchPredictionResponse:
    enforce_rate_limit(request)
    predictions = [predict_and_store(review, model, db, source="batch") for review in payload.reviews]
    return BatchPredictionResponse(predictions=predictions)


@router.post("/upload-csv", response_model=BatchPredictionResponse, tags=["predictions"])
async def upload_csv(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    model: SentimentModelService = Depends(get_model_service),
    _: str | None = Depends(get_current_subject),
) -> BatchPredictionResponse:
    enforce_rate_limit(request)
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload a CSV file")
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    try:
        predictions = predict_dataframe(df, model, db)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return BatchPredictionResponse(predictions=predictions)


@router.get("/analytics", response_model=AnalyticsResponse, tags=["analytics"])
def analytics(db: Session = Depends(get_db)) -> dict[str, object]:
    return get_analytics(db)
