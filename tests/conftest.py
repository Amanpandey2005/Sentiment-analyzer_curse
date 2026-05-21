from __future__ import annotations

import os
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:///./test_sentiment.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"
os.environ["JWT_SECRET_KEY"] = "test-secret"
os.environ["MODEL_PATH"] = "ml/saved_models/test_sentiment_model.joblib"
os.environ["TRAINING_DATA_PATH"] = "data/sample_reviews.csv"


def pytest_sessionfinish(session, exitstatus):  # type: ignore[no-untyped-def]
    for path in [
        Path("test_sentiment.db"),
        Path("ml/saved_models/test_sentiment_model.joblib"),
    ]:
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                pass
