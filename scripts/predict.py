from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment_analyzer.model import SentimentModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict sentiment for one e-commerce review.")
    parser.add_argument("--model", default="models/sentiment_model.json", help="Path to trained model JSON.")
    parser.add_argument("--text", required=True, help="Review text to classify.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = SentimentModel.load(args.model)
    result = {
        "text": args.text,
        "sentiment": model.predict(args.text),
        "probabilities": model.predict_proba(args.text),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
