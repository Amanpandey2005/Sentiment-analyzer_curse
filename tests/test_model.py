from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sentiment_analyzer.metrics import classification_report
from sentiment_analyzer.model import SentimentModel
from sentiment_analyzer.preprocessing import tokenize


class SentimentModelTests(unittest.TestCase):
    def test_tokenize_normalizes_review_text(self) -> None:
        self.assertEqual(tokenize("Amazing!!! The QUALITY is great."), ["amazing", "quality", "great"])

    def test_model_trains_predicts_and_reloads(self) -> None:
        texts = [
            "excellent premium delightful",
            "poor broken defective",
            "average okay expected",
        ]
        labels = ["positive", "negative", "neutral"]
        model = SentimentModel().fit(texts, labels)

        self.assertEqual(model.predict("excellent delightful"), "positive")
        self.assertEqual(model.predict("broken and poor"), "negative")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "sentiment_model.json"
            model.save(model_path)
            reloaded = SentimentModel.load(model_path)

        self.assertEqual(reloaded.predict("excellent delightful"), "positive")

    def test_classification_report_contains_macro_f1(self) -> None:
        report = classification_report(["positive", "negative"], ["positive", "positive"])
        self.assertIn("macro_avg", report)
        self.assertIn("f1", report["macro_avg"])


if __name__ == "__main__":
    unittest.main()
