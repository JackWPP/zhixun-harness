"""Fixed-class metrics shared by the offline lab and the harness.

The official hidden stream contains a frozen class list; a class that never
appears in a slice still contributes a zero F1.  ``rsi.metrics`` uses the
observed union instead, which silently inflates scores on three-class proxies.
"""
from __future__ import annotations

import numpy as np


def fixed_confusion(probabilities, expected, classes: int):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.int64).reshape(-1)
    predicted = np.argmax(probabilities, axis=1)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for label, prediction in zip(expected, predicted):
        if 0 <= label < classes and 0 <= prediction < classes:
            matrix[int(label), int(prediction)] += 1
    return matrix


def macro_f1_fixed(probabilities, expected, classes: int):
    """Macro-F1 over the fixed class list; missing classes score zero."""
    matrix = fixed_confusion(probabilities, expected, classes)
    f1 = []
    for label in range(classes):
        tp = float(matrix[label, label])
        fp = float(matrix[:, label].sum() - tp)
        fn = float(matrix[label, :].sum() - tp)
        support = tp + fn
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if support and denominator else 0.0)
    return float(np.mean(f1))


def support_report(probabilities, expected, classes: int):
    matrix = fixed_confusion(probabilities, expected, classes)
    return {"support": [int(matrix[label, :].sum()) for label in range(classes)],
            "predicted": [int(matrix[:, label].sum()) for label in range(classes)],
            "macro_f1": macro_f1_fixed(probabilities, expected, classes)}
