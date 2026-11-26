import json

import numpy as np
import pytest

from rsi.metrics import from_predictions


def test_c_nasa_keeps_tail_and_sums_samples():
    predicted = np.array([0.0, 200.0])
    expected = np.array([260.0, 0.0])
    values = from_predictions("regression", predicted, expected, profile="C")
    assert values["nasa_score"] == pytest.approx(2 * np.expm1(20.0))
    assert values["nasa_overflow"] is False
    other = from_predictions("regression", predicted, expected, profile="B")
    assert other["nasa_score"] == pytest.approx(1e6)


def test_c_nasa_overflow_is_explicit_and_json_safe():
    values = from_predictions("regression", np.array([1e5]), np.array([0.0]), profile="C")
    assert values["nasa_score"] is None
    assert values["nasa_overflow"] is True
    json.dumps(values, allow_nan=False)


def test_c_selection_rejects_nasa_regression_even_with_lower_rmse():
    from rsi.platform.submission import _choose

    parent = {"kind": "retain", "primary": 50.0,
              "values": {"rmse": 50.0, "mae": 35.0, "nasa_score": 1000.0}}
    unsafe = {"kind": "unsafe", "primary": 45.0,
              "values": {"rmse": 45.0, "mae": 34.0, "nasa_score": 2000.0}}
    safe = {"kind": "safe", "primary": 49.0,
            "values": {"rmse": 49.0, "mae": 34.0, "nasa_score": 900.0}}
    assert _choose([parent, unsafe, safe], "lower", 0.3, task_key="C") == 2
    assert _choose([parent, unsafe], "lower", 0.3, task_key="C") == 0
    assert _choose([parent, unsafe], "lower", 0.3, task_key="B") == 1


@pytest.mark.parametrize("metrics", [
    {"rmse": 49.0, "mae": 36.0, "nasa_score": 900.0},
    {"rmse": 49.0, "mae": 34.0, "nasa_score": None},
    {"rmse": 49.0, "mae": 34.0, "nasa_score": float("inf")},
    {"rmse": 49.9, "mae": 34.0, "nasa_score": 900.0},
])
def test_c_guard_requires_complete_finite_improvement(metrics):
    from rsi.platform.submission import _c_candidate_acceptable

    parent = {"values": {"rmse": 50.0, "mae": 35.0, "nasa_score": 1000.0}}
    assert not _c_candidate_acceptable({"values": metrics}, parent, 0.3)
