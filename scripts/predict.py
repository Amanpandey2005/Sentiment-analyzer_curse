"""Run one local prediction against a serialized model."""

from __future__ import annotations

import argparse

import joblib

from ml.preprocessing.text import normalize_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment for one review")
    parser.add_argument("--model", default="ml/saved_models/sentiment_model.joblib")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    pipeline = artifact["pipeline"]
    text = normalize_text(args.text)
    probabilities = pipeline.predict_proba([text])[0]
    probability_map = dict(zip(pipeline.classes_, probabilities))
    sentiment = max(probability_map, key=probability_map.get)
    print({"sentiment": sentiment, "probabilities": probability_map})


if __name__ == "__main__":
    main()
