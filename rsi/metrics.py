"""Metrics used by the harness; mirrors the official evaluation scripts."""
from __future__ import annotations

import numpy as np


def classification_metrics(probabilities: np.ndarray, expected: np.ndarray, classes) -> dict:
    labels = probabilities.argmax(axis=1)
    observed = sorted(set(expected.tolist()) | set(labels.tolist()))
    f1 = []
    for label in observed:
        true_positive = np.sum((labels == label) & (expected == label))
        denominator = np.sum(labels == label) + np.sum(expected == label)
        f1.append(2 * true_positive / denominator if denominator else 0.0)
    result = {"accuracy": float(np.mean(labels == expected)), "macro_f1": float(np.mean(f1)),
              "per_class_f1": {str(int(label)): float(value) for label, value in zip(observed, f1)}}
    if "SevereFault" in classes:
        severe = classes.index("SevereFault")
        count = int(np.sum(expected == severe))
        result["severe_miss_rate"] = (float(np.sum((expected == severe) & (labels != severe)) / count)
                                      if count else None)
    return result


def regression_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict:
    error = predicted - expected
    result = {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}
    # CMAPSS-style NASA score; normalized by sample count so units stay comparable.
    # The score is only meaningful when the target unit is a cycle count (task C).
    # For tasks whose target is minutes (task B) extreme errors would overflow the
    # exponential, so the per-row penalty is capped at 1e6 for report readability;
    # the platform only requires rmse+mae for task B, and selection never uses it.
    penalty = np.where(error < 0, np.expm1(-error / 13.0), np.expm1(error / 10.0))
    result["nasa_score"] = float(np.mean(np.minimum(penalty, 1e6)))
    result["bias"] = float(np.mean(error))
    return result


def metrics(task: str, model, x, y, classes=()) -> dict:
    predicted = model.predict(x)
    if task == "classification":
        return classification_metrics(predicted, y, classes)
    return regression_metrics(predicted, y)


def primary(task: str, values: dict) -> float:
    return values["macro_f1"] if task == "classification" else values["rmse"]


def improved(task: str, candidate: float, parent: float, minimum: float) -> bool:
    """True when candidate beats parent beyond the required minimum margin."""
    if task == "classification":
        return candidate >= parent + minimum
    return candidate <= parent - minimum


def from_predictions(task: str, predicted, expected, classes=()) -> dict:
    """Metrics for already-computed predictions (postprocess/TTA included)."""
    if task == "classification":
        return classification_metrics(predicted, expected, classes)
    return regression_metrics(predicted, expected)
