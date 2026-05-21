"""Backend helpers for the Streamlit sentiment dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .model import SentimentModel, load_reviews_csv

DEFAULT_MODEL_PATH = Path("models/sentiment_model.json")
REQUIRED_TEXT_COLUMN = "review_text"


def ensure_model(training_data_path: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH) -> SentimentModel:
    """Load a saved model, or train one from sample data if it does not exist."""

    model_file = Path(model_path)
    if model_file.exists():
        return SentimentModel.load(model_file)

    texts, labels = load_reviews_csv(training_data_path)
    model = SentimentModel().fit(texts, labels)
    model.save(model_file)
    return model


def load_input_data(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Load review data from a CSV path or uploaded file object."""

    df = pd.read_csv(source)
    if REQUIRED_TEXT_COLUMN not in df.columns:
        raise ValueError(f"CSV must include a '{REQUIRED_TEXT_COLUMN}' column.")

    df = df.copy()
    df[REQUIRED_TEXT_COLUMN] = df[REQUIRED_TEXT_COLUMN].fillna("").astype(str)
    df = df[df[REQUIRED_TEXT_COLUMN].str.strip().astype(bool)]

    if df.empty:
        raise ValueError("No non-empty reviews found in the uploaded CSV.")

    return df


def analyze_dataframe(df: pd.DataFrame, model: SentimentModel) -> pd.DataFrame:
    """Add sentiment predictions and confidence scores to a review dataframe."""

    analyzed = df.copy()
    predictions: list[str] = []
    confidences: list[float] = []

    for review_text in analyzed[REQUIRED_TEXT_COLUMN]:
        probabilities = model.predict_proba(review_text)
        sentiment = max(probabilities, key=probabilities.get)
        predictions.append(sentiment)
        confidences.append(probabilities[sentiment])

    analyzed["predicted_sentiment"] = predictions
    analyzed["confidence"] = confidences
    return analyzed


def summarize_predictions(df: pd.DataFrame) -> dict[str, object]:
    """Create dashboard summary metrics for predicted sentiment data."""

    counts = df["predicted_sentiment"].value_counts().to_dict()
    return {
        "total_reviews": int(len(df)),
        "positive_count": int(counts.get("positive", 0)),
        "neutral_count": int(counts.get("neutral", 0)),
        "negative_count": int(counts.get("negative", 0)),
        "average_confidence": float(df["confidence"].mean()),
    }


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return sentiment counts in a stable order for charting."""

    counts = df["predicted_sentiment"].value_counts()
    rows = [
        {"sentiment": label.title(), "reviews": int(counts.get(label, 0))}
        for label in ("positive", "neutral", "negative")
    ]
    return pd.DataFrame(rows)
