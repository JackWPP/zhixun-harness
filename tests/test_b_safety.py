import json
from pathlib import Path

import numpy as np
import pytest

from rsi.platform.b_safety import bearing_guard


def ids(first=10, second=90):
    return ([f"xjtu_Bearing1_4_s{i:06d}" for i in range(first)]
            + [f"xjtu_Bearing3_4_s{i:06d}" for i in range(second)])


def test_small_bearing_regression_cannot_hide_in_dominant_bearing_improvement():
    expected = np.zeros(100)
    parent = np.r_[np.full(10, 2.), np.full(90, 20.)]
    candidate = np.r_[np.full(10, 3.), np.full(90, 10.)]
    report = bearing_guard(parent, candidate, expected, ids())
    assert report["global"]["passed"]
    assert not report["passed"] and report["status"] == "bearing_regression"
    assert report["groups"]["Bearing1_4"]["failed_metrics"] == ["rmse", "mae"]
    assert report["coverage"]["complete"]


def test_both_rmse_and_mae_are_required_globally_and_per_bearing():
    expected = np.zeros(10)
    parent = np.r_[100., np.zeros(9)]
    candidate = np.full(10, 20.)
    report = bearing_guard(parent, candidate, expected, ids(10, 0))
    assert report["global"]["candidate"]["rmse"] < report["global"]["parent"]["rmse"]
    assert report["status"] == "global_regression"
    assert report["global"]["failed_metrics"] == ["mae"]


def test_equal_or_improving_bearings_are_allowed_with_case_insensitive_ids():
    expected = np.zeros(100)
    for candidate in (expected+2, expected+1):
        report = bearing_guard(expected+2, candidate, expected, [x.upper() for x in ids()])
        assert report["passed"] and report["coverage"]["complete"]


def test_all_unknown_ids_fail_closed_even_with_global_improvement():
    y = np.zeros(10)
    report = bearing_guard(y+2, y+1, y, [f"opaque-{i}" for i in range(10)])
    assert not report["passed"] and report["status"] == "unsupported_identifiers"
    assert report["global"]["passed"]
    assert report["coverage"]["supported_groups"] == []


def test_sparse_and_unknown_groups_are_explicit_when_some_bearings_are_supported():
    identifiers = ids(5, 10)+["opaque"]
    y = np.zeros(len(identifiers))
    report = bearing_guard(y+2, y+1, y, identifiers)
    assert report["passed"] and not report["coverage"]["complete"]
    assert report["insufficient"] == ["Bearing1_4"] and report["unknown_rows"] == 1
    assert report["coverage"]["supported_rows"] == 10


def test_duplicate_snapshots_cannot_inflate_support():
    y = np.zeros(10)
    report = bearing_guard(y+2, y+1, y, ["xjtu_Bearing1_4_s000001"]*10)
    assert not report["passed"] and report["status"] == "insufficient_support"
    assert report["groups"]["Bearing1_4"]["unique_snapshots"] == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e308])
def test_nonfinite_or_overflowing_errors_are_rejected_and_json_safe(bad):
    y = np.zeros(10)
    report = bearing_guard(y+2, y+bad, y, ids(10, 0))
    assert not report["passed"]
    json.dumps(report, allow_nan=False)
    if bad == 1e308:
        assert report["global"]["candidate"]["rmse"] is None


def test_real_official_validation_ids_are_fully_recognized():
    path = Path(__file__).resolve().parents[1] / "datasets/cpu-b-rul-20261110-v1/public/validation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    y = np.array([row["answer"] for row in rows], dtype=float)
    report = bearing_guard(y+2, y+1, y, [row["id"] for row in rows])
    assert report["passed"] and report["coverage"]["complete"]
    assert report["groups"]["Bearing1_4"]["rows"] == 122
    assert "Bearing3_4" in report["groups"]


def test_bad_shapes_or_missing_evidence_do_not_authorize_promotion():
    y = np.zeros(10)
    assert not bearing_guard(y, y[:2], y, ids(10, 0))["passed"]
    assert not bearing_guard(y, y, y, None)["passed"]
    with pytest.raises(ValueError):
        bearing_guard(y, y, y, ids(10, 0), min_rows=0)
