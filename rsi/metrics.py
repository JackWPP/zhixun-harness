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


def regression_metrics(predicted: np.ndarray, expected: np.ndarray, *, profile=None) -> dict:
    error = predicted - expected
    result = {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}
    # C uses the official untruncated CMAPSS-style penalty sum in cycle units.
    # Other profiles retain the legacy capped mean for report compatibility.
    # In task B, minute-scale errors can overflow the exponential; the platform
    # requires rmse+mae for B and selection never uses this auxiliary NASA value.
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    with np.errstate(over="ignore"):
        penalty = np.expm1(exponent)
    if profile == "C":
        # C's platform score is an untruncated sum. A nonrepresentable score
        # stays explicit and JSON-safe so selection can reject the candidate.
        with np.errstate(over="ignore"):
            nasa_score = float(penalty.sum())
        result["nasa_score"] = nasa_score if np.isfinite(nasa_score) else None
        result["nasa_overflow"] = not np.isfinite(nasa_score)
    else:
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


def from_predictions(task: str, predicted, expected, classes=(), *, profile=None) -> dict:
    """Metrics for already-computed predictions (postprocess/TTA included)."""
    if task == "classification":
        return classification_metrics(predicted, expected, classes)
    return regression_metrics(predicted, expected, profile=profile)
