"""Conservative C continuation and validation-only subgroup checks.

FD identifiers and observed RUL labels only audit validation predictions. They
never enter the model or select a prediction route. Missing FD evidence can use
fixed RUL groups, explicitly reported as a proxy rather than hidden stages.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from rsi.metrics import from_predictions
from rsi.model import ExtendedModel

ALPHAS = (0.25, 0.5)
METRICS = ("rmse", "mae", "nasa_score")
_FD = re.compile(r"^(fd00[1-4])(?:_|-)", re.IGNORECASE)
_CYCLE = re.compile(r"_c\d+$", re.IGNORECASE)


def _check_compatible(parent: ExtendedModel, trained: ExtendedModel) -> None:
    parent.validate()
    trained.validate()
    if parent.meta.get("task") != "regression":
        raise ValueError("C interpolation requires regression checkpoints")
    for key in ("task", "paths", "classes", "hidden", "parameter_count",
                "target_mean", "target_scale"):
        if parent.meta.get(key) != trained.meta.get(key):
            raise ValueError("incompatible checkpoint metadata: " + key)
    for key in ("mean", "scale"):
        if not np.array_equal(getattr(parent, key), getattr(trained, key)):
            raise ValueError("incompatible checkpoint input normalization: " + key)
    for key in parent.params:
        if parent.params[key].shape != trained.params[key].shape:
            raise ValueError("incompatible checkpoint parameter: " + key)
    if trained.meta["epochs"] < parent.meta["epochs"]:
        raise ValueError("trained checkpoint predates parent epoch history")


def interpolate_checkpoint(parent: ExtendedModel, trained: ExtendedModel,
                           alpha: float) -> ExtendedModel:
    """One inference network, with an explicit Adam restart for future training.

    Adam moments estimate gradients along the full candidate's trajectory. They
    are not estimates for an interpolated trajectory, and averaging second
    moments does not reconstruct it. Restarting both moments and the bias-
    correction counter defines a consistent optimizer at the selected weights.
    The actual epoch history remains intact; no training is claimed here.
    """
    if isinstance(alpha, bool) or alpha not in ALPHAS:
        raise ValueError("only the predeclared C interpolation alphas are supported")
    _check_compatible(parent, trained)
    result = trained.clone()
    for key in result.params:
        result.params[key] = parent.params[key] + alpha * (trained.params[key] - parent.params[key])
    result.reset_optimizer()
    result.meta["c_conservative"] = {
        "alpha": float(alpha),
        "operation": "parameter_interpolation",
        "optimizer_policy": "restart_adam_preserve_epoch_history",
        "parent_epochs": int(parent.meta["epochs"]),
        "trained_epochs": int(trained.meta["epochs"]),
        "parent_steps": int(parent.meta["steps"]),
        "trained_steps": int(trained.meta["steps"]),
    }
    result.validate()
    return result


def conservative_candidates(parent: ExtendedModel, trained: ExtendedModel):
    """Return the two fixed alternatives; reuse one already-trained candidate."""
    return [(alpha, interpolate_checkpoint(parent, trained, alpha)) for alpha in ALPHAS]


def group_guard(parent_pred, candidate_pred, expected, row_ids: Iterable[str], *,
                min_rows: int = 8, min_engines: int = 2,
                enforce_rul: bool = False) -> dict:
    """Require supported FD groups not to regress in any of the three metrics.

    The result is JSON-safe and independent of the global RMSE improvement gate.
    Supported FD groups are protected by default. Fixed RUL buckets are always
    audited and become the fallback when no FD group has sufficient support.
    ``enforce_rul=True`` also requires those RUL groups when FD evidence exists.
    Partial FD support is disclosed instead of claiming full FD protection.
    """
    if type(min_rows) is not int or type(min_engines) is not int or min_rows < 1 or min_engines < 1:
        raise ValueError("group support thresholds must be positive integers")
    parent_pred = np.asarray(parent_pred, dtype=float)
    candidate_pred = np.asarray(candidate_pred, dtype=float)
    expected = np.asarray(expected, dtype=float)
    row_ids = list(row_ids)
    result = {"passed": False, "status": "invalid_evidence", "groups": {}, "rul_groups": {},
              "insufficient": [], "unknown_rows": 0, "minimum_rows": min_rows,
              "minimum_engines": min_engines, "enforce_rul": bool(enforce_rul),
              "coverage": {}, "rul_guard_passed": None,
              "limitation": "FD and fixed RUL groups do not identify the platform's hidden drift stages."}
    if (expected.ndim != 1 or not len(expected) or parent_pred.shape != expected.shape
            or candidate_pred.shape != expected.shape or len(row_ids) != len(expected)
            or not all(np.all(np.isfinite(value)) for value in (parent_pred, candidate_pred, expected))):
        return result
    groups: dict[str, list[int]] = {}
    units: dict[str, set[str]] = {}
    for index, raw_id in enumerate(row_ids):
        identifier = str(raw_id)
        match = _FD.match(identifier)
        if match is None or _CYCLE.search(identifier) is None:
            result["unknown_rows"] += 1
            continue
        group = match.group(1).lower()
        groups.setdefault(group, []).append(index)
        units.setdefault(group, set()).add(_CYCLE.sub("", identifier.lower()))
    def compare(indices):
        def measured(predicted):
            with np.errstate(over="ignore", invalid="ignore"):
                values = from_predictions("regression", predicted[indices], expected[indices], profile="C")
            return {key: None if isinstance(value, float) and not np.isfinite(value) else value
                    for key, value in values.items()}
        parent = measured(parent_pred)
        candidate = measured(candidate_pred)
        failed = []
        for metric in METRICS:
            old, new = parent[metric], candidate[metric]
            if old is None or new is None or not np.isfinite(old) or not np.isfinite(new):
                failed.append(metric)
            elif new > old + 1e-9 + abs(old) * 1e-12:
                failed.append(metric)
        return {"parent": parent, "candidate": candidate, "passed": not failed,
                "failed_metrics": failed, "status": "regression" if failed else "noninferior"}

    supported_fd = []
    for group, indices in sorted(groups.items()):
        entry = {"rows": len(indices), "engines": len(units[group]), "passed": False}
        result["groups"][group] = entry
        if len(indices) < min_rows or len(units[group]) < min_engines:
            result["insufficient"].append(group)
            entry["status"] = "insufficient_support"
            continue
        entry.update(compare(indices))
        supported_fd.append(group)
    supported_rul = []
    for group, mask in (("RUL_0_99", (expected >= 0) & (expected < 100)),
                        ("RUL_100_249", (expected >= 100) & (expected < 250)),
                        ("RUL_250_plus", expected >= 250)):
        indices = np.flatnonzero(mask)
        entry = {"rows": len(indices), "passed": False, "status": "insufficient_support"}
        result["rul_groups"][group] = entry
        if len(indices) >= min_rows:
            entry.update(compare(indices))
            supported_rul.append(group)
    if supported_rul:
        result["rul_guard_passed"] = all(result["rul_groups"][group]["passed"] for group in supported_rul)
    result["coverage"] = {
        "total_rows": len(expected), "fd_recognized_rows": len(expected)-result["unknown_rows"],
        "fd_supported_rows": sum(result["groups"][group]["rows"] for group in supported_fd),
        "fd_supported_groups": supported_fd,
        "fd_complete": bool(supported_fd) and not result["unknown_rows"] and not result["insufficient"],
        "rul_supported_groups": supported_rul,
        "rul_supported_rows": sum(result["rul_groups"][group]["rows"] for group in supported_rul),
    }
    if supported_fd:
        passed = all(result["groups"][group]["passed"] for group in supported_fd)
        if enforce_rul:
            passed = passed and result["rul_guard_passed"] is True
        result.update(passed=passed, protection="fd_and_rul" if enforce_rul else "fd",
                      status="noninferior" if passed else "group_regression")
    elif supported_rul:
        passed = result["rul_guard_passed"]
        result.update(passed=passed, protection="rul_fallback",
                      status="rul_fallback_noninferior" if passed else "rul_fallback_regression")
    else:
        result.update(status="insufficient_support", protection="none")
    return result
