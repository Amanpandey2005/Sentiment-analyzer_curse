"""Trainable sentiment classifier for e-commerce reviews."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .preprocessing import tokenize


@dataclass
class SentimentModel:
    """Multinomial Naive Bayes sentiment classifier."""

    alpha: float = 1.0
    labels: list[str] = field(default_factory=list)
    class_doc_counts: dict[str, int] = field(default_factory=dict)
    token_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    total_tokens: dict[str, int] = field(default_factory=dict)
    vocabulary: set[str] = field(default_factory=set)

    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> "SentimentModel":
        class_doc_counts: Counter[str] = Counter()
        token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        total_tokens: Counter[str] = Counter()
        vocabulary: set[str] = set()

        for text, label in zip(texts, labels):
            label = label.strip().lower()
            if not label:
                continue

            class_doc_counts[label] += 1
            tokens = tokenize(text)
            token_counts[label].update(tokens)
            total_tokens[label] += len(tokens)
            vocabulary.update(tokens)

        if not class_doc_counts:
            raise ValueError("Cannot train model without labeled examples.")

        self.labels = sorted(class_doc_counts)
        self.class_doc_counts = dict(class_doc_counts)
        self.token_counts = {label: dict(counts) for label, counts in token_counts.items()}
        self.total_tokens = dict(total_tokens)
        self.vocabulary = vocabulary
        return self

    def predict(self, text: str) -> str:
        scores = self.predict_log_proba(text)
        return max(scores, key=scores.get)

    def predict_batch(self, texts: Iterable[str]) -> list[str]:
        return [self.predict(text) for text in texts]

    def predict_log_proba(self, text: str) -> dict[str, float]:
        if not self.labels:
            raise ValueError("Model is not trained.")

        tokens = tokenize(text)
        doc_count = sum(self.class_doc_counts.values())
        vocab_size = len(self.vocabulary) or 1
        scores: dict[str, float] = {}

        for label in self.labels:
            prior = self.class_doc_counts[label] / doc_count
            score = math.log(prior)
            denominator = self.total_tokens.get(label, 0) + self.alpha * vocab_size
            counts = self.token_counts.get(label, {})

            for token in tokens:
                numerator = counts.get(token, 0) + self.alpha
                score += math.log(numerator / denominator)

            scores[label] = score

        return scores

    def predict_proba(self, text: str) -> dict[str, float]:
        log_scores = self.predict_log_proba(text)
        max_score = max(log_scores.values())
        exp_scores = {label: math.exp(score - max_score) for label, score in log_scores.items()}
        total = sum(exp_scores.values()) or 1.0
        return {label: score / total for label, score in exp_scores.items()}

    def save(self, path: str | Path) -> None:
        payload = {
            "alpha": self.alpha,
            "labels": self.labels,
            "class_doc_counts": self.class_doc_counts,
            "token_counts": self.token_counts,
            "total_tokens": self.total_tokens,
            "vocabulary": sorted(self.vocabulary),
        }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SentimentModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            alpha=payload["alpha"],
            labels=payload["labels"],
            class_doc_counts=payload["class_doc_counts"],
            token_counts=payload["token_counts"],
            total_tokens=payload["total_tokens"],
            vocabulary=set(payload["vocabulary"]),
        )


def load_reviews_csv(path: str | Path, text_column: str = "review_text", label_column: str = "sentiment") -> tuple[list[str], list[str]]:
    """Load review text and labels from a CSV file."""

    texts: list[str] = []
    labels: list[str] = []

    with Path(path).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {text_column, label_column}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

        for row in reader:
            text = (row.get(text_column) or "").strip()
            label = (row.get(label_column) or "").strip().lower()
            if text and label:
                texts.append(text)
                labels.append(label)

    return texts, labels
