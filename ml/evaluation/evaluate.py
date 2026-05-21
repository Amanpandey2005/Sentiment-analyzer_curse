"""Evaluate a serialized sentiment model."""

from __future__ import annotations

import argparse

import joblib
from sklearn.metrics import classification_report

from ml.training.train import load_training_data


def evaluate_model(data_path: str, model_path: str) -> dict[str, object]:
    """Return a classification report for a saved model."""

    artifact = joblib.load(model_path)
    model = artifact["pipeline"]
    df = load_training_data(data_path)
    predictions = model.predict(df["review_text"])
    return classification_report(df["sentiment"], predictions, output_dict=True, zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentiment model")
    parser.add_argument("--data", default="data/sample_reviews.csv")
    parser.add_argument("--model", default="ml/saved_models/sentiment_model.joblib")
    args = parser.parse_args()
    print(evaluate_model(args.data, args.model))


if __name__ == "__main__":
    main()
