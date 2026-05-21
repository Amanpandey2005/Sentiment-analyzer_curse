"""Small classification metrics module without third-party dependencies."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    true_values = list(y_true)
    pred_values = list(y_pred)
    if not true_values:
        return 0.0
    correct = sum(actual == predicted for actual, predicted in zip(true_values, pred_values))
    return correct / len(true_values)


def classification_report(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, object]:
    true_values = list(y_true)
    pred_values = list(y_pred)
    labels = sorted(set(true_values) | set(pred_values))
    report: dict[str, object] = {"accuracy": accuracy(true_values, pred_values), "labels": {}}

    label_metrics = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in labels:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(true_values, pred_values))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(true_values, pred_values))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(true_values, pred_values))
        support = sum(actual == label for actual in true_values)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        label_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    label_count = len(labels) or 1
    report["labels"] = label_metrics
    report["macro_avg"] = {
        "precision": macro_precision / label_count,
        "recall": macro_recall / label_count,
        "f1": macro_f1 / label_count,
    }
    report["confusion_matrix"] = confusion_matrix(true_values, pred_values, labels)
    return report


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1

    return {
        actual: {predicted: matrix[actual][predicted] for predicted in labels}
        for actual in labels
    }
