from __future__ import annotations

from pathlib import Path

import joblib

from ml.training.train import train_model


def test_training_pipeline_creates_joblib_model(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    report = train_model("data/sample_reviews.csv", model_path)
    artifact = joblib.load(model_path)

    assert model_path.exists()
    assert "pipeline" in artifact
    assert "accuracy" in report
