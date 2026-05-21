"""Prediction persistence and analytics."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.prediction import PredictionRecord
from app.models.schemas import PredictionResponse
from app.monitoring.metrics import PREDICTION_COUNT
from app.services.model_service import SentimentModelService


def predict_and_store(
    review_text: str,
    model_service: SentimentModelService,
    db: Session,
    source: str = "api",
) -> PredictionResponse:
    """Run inference and persist the result."""

    prediction = model_service.predict(review_text)
    record = PredictionRecord(
        review_text=review_text,
        sentiment=str(prediction["sentiment"]),
        confidence=float(prediction["confidence"]),
        summary=str(prediction["summary"]),
        source=source,
    )
    db.add(record)
    db.commit()
    PREDICTION_COUNT.labels(record.sentiment).inc()
    return PredictionResponse(**prediction)


def predict_dataframe(df: pd.DataFrame, model_service: SentimentModelService, db: Session) -> list[PredictionResponse]:
    """Predict all reviews in a dataframe that includes review_text."""

    if "review_text" not in df.columns:
        raise ValueError("CSV must contain a review_text column")
    responses: list[PredictionResponse] = []
    for review_text in df["review_text"].dropna().astype(str).tolist():
        if review_text.strip():
            responses.append(predict_and_store(review_text, model_service, db, source="csv"))
    return responses


def get_analytics(db: Session) -> dict[str, object]:
    """Aggregate prediction analytics for API and dashboard."""

    total = db.scalar(select(func.count(PredictionRecord.id))) or 0
    distribution_rows = db.execute(
        select(PredictionRecord.sentiment, func.count(PredictionRecord.id)).group_by(PredictionRecord.sentiment)
    ).all()
    distribution = {sentiment: int(count) for sentiment, count in distribution_rows}
    avg_confidence = db.scalar(select(func.avg(PredictionRecord.confidence))) or 0.0
    trend_rows = db.execute(
        select(
            func.date(PredictionRecord.created_at).label("day"),
            PredictionRecord.sentiment,
            func.count(PredictionRecord.id),
        )
        .group_by("day", PredictionRecord.sentiment)
        .order_by("day")
    ).all()
    trends = [{"date": str(day), "sentiment": sentiment, "count": int(count)} for day, sentiment, count in trend_rows]

    top_positive = _top_reviews(db, "positive")
    top_negative = _top_reviews(db, "negative")
    return {
        "total_predictions": total,
        "sentiment_distribution": distribution,
        "average_confidence": float(avg_confidence),
        "daily_trends": trends,
        "top_positive_reviews": top_positive,
        "top_negative_reviews": top_negative,
        "generated_at": datetime.now(UTC),
    }


def _top_reviews(db: Session, sentiment: str) -> list[PredictionResponse]:
    rows = db.scalars(
        select(PredictionRecord)
        .where(PredictionRecord.sentiment == sentiment)
        .order_by(PredictionRecord.confidence.desc())
        .limit(5)
    ).all()
    return [
        PredictionResponse(
            review_text=row.review_text,
            sentiment=row.sentiment,
            confidence=row.confidence,
            probabilities={row.sentiment: row.confidence},
            summary=row.summary,
        )
        for row in rows
    ]
