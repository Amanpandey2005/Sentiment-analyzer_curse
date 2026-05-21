"""Train and serialize the sentiment model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.preprocessing.text import normalize_text

LABELS = ["negative", "neutral", "positive"]


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load a labeled review CSV and validate required columns."""

    df = pd.read_csv(path)
    required = {"review_text", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")
    df = df.dropna(subset=["review_text", "sentiment"]).copy()
    df["review_text"] = df["review_text"].map(normalize_text)
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    return df[df["sentiment"].isin(LABELS)]


def build_pipeline() -> Pipeline:
    """Create the TF-IDF + Logistic Regression classifier."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=25_000,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(data_path: str | Path, model_out: str | Path) -> dict[str, object]:
    """Train the classifier and persist it as a joblib artifact."""

    df = load_training_data(data_path)
    if df.empty:
        raise ValueError("Training data contains no valid labeled reviews.")

    pipeline = build_pipeline()
    stratify = df["sentiment"] if df["sentiment"].value_counts().min() >= 2 else None
    train_x, test_x, train_y, test_y = train_test_split(
        df["review_text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(test_x)
    report = classification_report(test_y, predictions, output_dict=True, zero_division=0)

    output_path = Path(model_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "labels": LABELS, "report": report}, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--data", default="data/sample_reviews.csv")
    parser.add_argument("--model-out", default="ml/saved_models/sentiment_model.joblib")
    args = parser.parse_args()
    report = train_model(args.data, args.model_out)
    print(report)


if __name__ == "__main__":
    main()
