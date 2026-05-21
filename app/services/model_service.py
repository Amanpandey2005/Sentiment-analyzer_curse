"""Model loading and inference service."""

from __future__ import annotations

import logging
from functools import lru_cache

import joblib

from app.core_config import get_settings
from ml.preprocessing.text import normalize_text
from ml.training.train import LABELS, train_model

logger = logging.getLogger(__name__)


class SentimentModelService:
    """Load, warm, and query the sentiment model artifact."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.artifact = self._load_or_train()
        self.pipeline = self.artifact["pipeline"]
        self.labels: list[str] = list(self.artifact.get("labels", LABELS))

    def _load_or_train(self) -> dict[str, object]:
        if not self.settings.model_path.exists():
            logger.info("Model artifact missing; training from bundled dataset")
            train_model(self.settings.training_data_path, self.settings.model_path)
        return joblib.load(self.settings.model_path)

    def predict(self, review_text: str) -> dict[str, object]:
        """Predict sentiment and class probabilities for one review."""

        normalized = normalize_text(review_text)
        probabilities = self.pipeline.predict_proba([normalized])[0]
        classes = list(self.pipeline.classes_)
        probability_map = {label: 0.0 for label in self.labels}
        probability_map.update({label: float(prob) for label, prob in zip(classes, probabilities, strict=False)})
        sentiment = max(probability_map, key=probability_map.get)
        return {
            "review_text": review_text,
            "sentiment": sentiment,
            "confidence": probability_map[sentiment],
            "probabilities": probability_map,
            "summary": summarize_review(review_text, sentiment),
        }


def summarize_review(review_text: str, sentiment: str) -> str:
    """Generate a short extractive summary without external LLM calls."""

    text = " ".join(review_text.split())
    snippet = text[:180].rstrip()
    suffix = "..." if len(text) > 180 else ""
    return f"{sentiment.title()} review summary: {snippet}{suffix}"


@lru_cache
def get_model_service() -> SentimentModelService:
    return SentimentModelService()
