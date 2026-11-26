"""Validation-only B bearing guard; identifiers never become model features."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

_BEARING = re.compile(r"^xjtu_Bearing(\d+)_(\d+)_s(\d+)$", re.IGNORECASE)
_METRICS = ("rmse", "mae")


def _metrics(predicted: np.ndarray, expected: np.ndarray) -> dict:
    with np.errstate(over="ignore", invalid="ignore"):
        error = predicted-expected
        values = {"rmse": float(np.sqrt(np.mean(error*error))),
                  "mae": float(np.mean(np.abs(error)))}
    return {key: value if np.isfinite(value) else None for key, value in values.items()}


def _comparison(parent: np.ndarray, candidate: np.ndarray, expected: np.ndarray) -> dict:
    old, new = _metrics(parent, expected), _metrics(candidate, expected)
    failed = [key for key in _METRICS
              if old[key] is None or new[key] is None
              or new[key] > old[key]+1e-9+abs(old[key])*1e-12]
    return {"parent": old, "candidate": new, "passed": not failed,
            "failed_metrics": failed, "status": "regression" if failed else "noninferior"}


def bearing_guard(parent_pred, candidate_pred, expected, row_ids: Iterable[str],
                  min_rows: int = 10) -> dict:
    """Require global and sufficiently supported bearing RMSE/MAE to hold.

    Official IDs encode the bearing and snapshot as xjtu_Bearing1_4_s000001.
    They are used only to audit validation predictions, never at inference.
    Support requires at least ``min_rows`` distinct snapshots, so duplicate
    rows cannot manufacture group support. Unrecognized and sparse groups are
    disclosed. If no bearing is supported, promotion is rejected because no
    validated alternative grouping is available for this task.
    """
    if type(min_rows) is not int or min_rows < 1:
        raise ValueError("min_rows must be a positive integer")
    result = {"passed": False, "status": "invalid_evidence", "minimum_rows": min_rows,
              "global": None, "groups": {}, "insufficient": [], "unknown_rows": 0,
              "coverage": {},
              "limitation": "Public bearing point estimates do not establish hidden-stage or unseen-bearing safety."}
    try:
        parent_pred = np.asarray(parent_pred, dtype=float)
        candidate_pred = np.asarray(candidate_pred, dtype=float)
        expected = np.asarray(expected, dtype=float)
        row_ids = list(row_ids)
    except (TypeError, ValueError):
        return result
    if (expected.ndim != 1 or not len(expected) or parent_pred.shape != expected.shape
            or candidate_pred.shape != expected.shape or len(row_ids) != len(expected)
            or not all(np.all(np.isfinite(x)) for x in (parent_pred, candidate_pred, expected))):
        return result
    result["global"] = _comparison(parent_pred, candidate_pred, expected)
    indices_by_group: dict[str, list[int]] = {}
    snapshots: dict[str, set[int]] = {}
    for index, identifier in enumerate(row_ids):
        match = _BEARING.fullmatch(str(identifier))
        if match is None:
            result["unknown_rows"] += 1
            continue
        condition, bearing, snapshot = map(int, match.groups())
        group = f"Bearing{condition}_{bearing}"
        indices_by_group.setdefault(group, []).append(index)
        snapshots.setdefault(group, set()).add(snapshot)
    supported = []
    for group, indices in sorted(indices_by_group.items()):
        entry = {"rows": len(indices), "unique_snapshots": len(snapshots[group]),
                 "passed": False, "status": "insufficient_support"}
        result["groups"][group] = entry
        if len(snapshots[group]) < min_rows:
            result["insufficient"].append(group)
            continue
        entry.update(_comparison(parent_pred[indices], candidate_pred[indices], expected[indices]))
        supported.append(group)
    result["coverage"] = {
        "total_rows": len(expected), "recognized_rows": len(expected)-result["unknown_rows"],
        "supported_rows": sum(result["groups"][group]["rows"] for group in supported),
        "supported_groups": supported,
        "complete": bool(supported) and result["unknown_rows"] == 0 and not result["insufficient"],
    }
    if not result["global"]["passed"]:
        result["status"] = "global_regression"
    elif not supported:
        result["status"] = "unsupported_identifiers" if not indices_by_group else "insufficient_support"
    elif not all(result["groups"][group]["passed"] for group in supported):
        result["status"] = "bearing_regression"
    else:
        result.update(passed=True, status="noninferior")
    return result
