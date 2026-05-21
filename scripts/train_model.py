from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment_analyzer.metrics import classification_report
from sentiment_analyzer.model import SentimentModel, load_reviews_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an e-commerce sentiment model.")
    parser.add_argument("--data", default="data/sample_reviews.csv", help="CSV file with review_text and sentiment columns.")
    parser.add_argument("--model-out", default="models/sentiment_model.json", help="Path for the trained model JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts, labels = load_reviews_csv(args.data)
    model = SentimentModel().fit(texts, labels)
    predictions = model.predict_batch(texts)
    report = classification_report(labels, predictions)
    model.save(args.model_out)

    print(f"Trained on {len(texts)} reviews.")
    print(f"Labels: {', '.join(model.labels)}")
    print(f"Training accuracy: {report['accuracy']:.2%}")
    print(f"Saved model: {args.model_out}")


if __name__ == "__main__":
    main()
