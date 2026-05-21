from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment_analyzer.metrics import classification_report
from sentiment_analyzer.model import SentimentModel, load_reviews_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an e-commerce sentiment model.")
    parser.add_argument("--data", default="data/sample_reviews.csv", help="CSV file with review_text and sentiment columns.")
    parser.add_argument("--model", default="models/sentiment_model.json", help="Path to trained model JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts, labels = load_reviews_csv(args.data)
    model = SentimentModel.load(args.model)
    predictions = model.predict_batch(texts)
    report = classification_report(labels, predictions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
