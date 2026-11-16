"""Drift characterisation: covariate, label/prior, and performance drift.

The harness never trusts a single number: a drift report names the evidence, a
severity, a likely mechanism, and the strategies that are admissible given it.
All statistics are computed on public data only (unlabelled probes are allowed
for covariate statistics; labels are used when the split provides them).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rsi.metrics import primary, metrics


def _js_divergence(p, q, bins=20):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if len(p) == 0 or len(q) == 0:
        return 0.0
    low, high = float(min(p.min(), q.min())), float(max(p.max(), q.max()))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return 0.0
    edges = np.linspace(low, high, bins + 1)
    hp, _ = np.histogram(p, bins=edges, density=False)
    hq, _ = np.histogram(q, bins=edges, density=False)
    hp = hp / max(1, hp.sum())
    hq = hq / max(1, hq.sum())
    m = 0.5 * (hp + hq)
    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / m[mask])))
    return 0.5 * (kl(hp) + kl(hq))


def _ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (no SciPy dependency)."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return 0.0
    points = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, points, side="right") / len(a)
    cdf_b = np.searchsorted(b, points, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


@dataclass
class DriftReport:
    feature_shift: float
    max_feature_shift: float
    shifted_fraction: float
    ks_max: float
    ks_p95: float
    correlation_shift: float
    label_shift: float | None
    performance_before: dict | None
    performance_target: dict | None
    performance_drop: float | None
    severity: str
    causes: list[str]
    evidence: dict

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


def _regression_shift(reference_y, target_y):
    if reference_y is None or target_y is None or len(reference_y) == 0 or len(target_y) == 0:
        return None
    scale = max(float(np.std(reference_y)), 1e-9)
    return {"mean_shift_sd": float((np.mean(target_y) - np.mean(reference_y)) / scale),
            "std_ratio": float(np.std(target_y) / max(np.std(reference_y), 1e-9)),
            "js": _js_divergence(reference_y, target_y)}


def _classification_shift(reference_y, target_y, n_classes):
    if reference_y is None or target_y is None or len(reference_y) == 0 or len(target_y) == 0:
        return None
    ref = np.bincount(reference_y.astype(int), minlength=n_classes).astype(np.float64)
    tgt = np.bincount(target_y.astype(int), minlength=n_classes).astype(np.float64)
    ref /= max(ref.sum(), 1)
    tgt /= max(tgt.sum(), 1)
    m = 0.5 * (ref + tgt)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(ref > 0, ref * np.log2(ref / np.where(m > 0, m, 1)), 0.0)
        kl2 = np.where(tgt > 0, tgt * np.log2(tgt / np.where(m > 0, m, 1)), 0.0)
    return {"js": float(0.5 * (kl.sum() + kl2.sum())),
            "missing_classes": [int(i) for i in range(n_classes) if ref[i] > 0 and tgt[i] == 0],
            "new_classes": [int(i) for i in range(n_classes) if tgt[i] > 0 and ref[i] == 0],
            "mean_shift": float(np.mean(tgt - ref))}


def analyse_covariates(reference_x, target_x) -> dict:
    """Summarize covariate drift without inspecting target labels or model loss.

    This is the only drift view suitable before a streaming batch's labels are
    revealed. Robust median/MAD location summaries reduce single-sensor-spike
    sensitivity; KS quantiles and correlation change add complementary signals.
    Thresholds remain heuristic until calibrated on a wider no-drift stream.
    """
    rx = np.asarray(reference_x, dtype=np.float64)
    tx = np.asarray(target_x, dtype=np.float64)
    if rx.ndim != 2 or tx.ndim != 2 or not len(rx) or not len(tx):
        raise ValueError("covariate drift requires non-empty 2D reference and target arrays")
    if rx.shape[1] != tx.shape[1]:
        raise ValueError("reference and target feature widths differ")
    if not np.isfinite(rx).all() or not np.isfinite(tx).all():
        raise ValueError("covariate drift inputs must be finite")

    ref_median = np.median(rx, axis=0)
    target_median = np.median(tx, axis=0)
    mad = 1.4826 * np.median(np.abs(rx - ref_median), axis=0)
    std = np.std(rx, axis=0)
    scale = np.where(mad > 1e-12, mad, np.where(std > 1e-12, std, 1.0))
    location_shift = np.abs(target_median - ref_median) / scale
    shifted_fraction = float(np.mean(location_shift > 1.0))
    ks = np.array([_ks_statistic(rx[:, i], tx[:, i]) for i in range(rx.shape[1])])

    if rx.shape[1] == 1:
        correlation_shift = 0.0
    else:
        corr_r = np.nan_to_num(np.corrcoef(rx, rowvar=False))
        corr_t = np.nan_to_num(np.corrcoef(tx, rowvar=False))
        correlation_shift = float(np.linalg.norm(corr_r - corr_t) /
                                  math.sqrt(max(1, rx.shape[1])))
    feature_shift = float(np.median(location_shift))
    ks_p95 = float(np.percentile(ks, 95))
    score = (0.45 * min(1.0, feature_shift) +
             0.35 * min(1.0, ks_p95 / 0.3) +
             0.20 * min(1.0, correlation_shift / 0.5))
    severity = "severe" if score >= 0.6 else "moderate" if score >= 0.32 else "mild"
    causes = []
    if feature_shift > 0.45 and shifted_fraction > 0.1:
        causes.append("covariate_location_shift")
    if ks_p95 > 0.3:
        causes.append("feature_distribution_shift")
    if correlation_shift > 0.5:
        causes.append("feature_relationship_shift")
    if not causes:
        causes.append("no_material_covariate_shift_detected")
    return {
        "monitor_type": "prequential_covariate_only",
        "feature_shift": feature_shift,
        "max_feature_shift": float(np.max(location_shift)),
        "shifted_fraction": shifted_fraction,
        "ks_max": float(ks.max()),
        "ks_p95": ks_p95,
        "correlation_shift": correlation_shift,
        "label_shift": None,
        "performance_before": None,
        "performance_target": None,
        "performance_drop": None,
        "severity": severity,
        "causes": causes,
        "evidence": {
            "top_shifted_features": [int(i) for i in np.argsort(location_shift)[-8:][::-1]],
            "rows_reference": int(len(rx)),
            "rows_target": int(len(tx)),
            "severity_score": score,
            "attribution_limit": "unlabeled covariate evidence cannot establish concept drift or performance loss",
        },
    }


def analyse(model, reference, target, classes=()):
    """Compare parent model behaviour and input statistics on two public splits."""
    rx, tx = reference.x, target.x
    sd = rx.std(axis=0)
    sd[sd < 1e-12] = 1.0
    delta = np.abs(tx.mean(axis=0) - rx.mean(axis=0)) / sd
    shifted_fraction = float(np.mean(delta > 1.0))
    corr_r = np.corrcoef(rx, rowvar=False)
    corr_t = np.corrcoef(tx, rowvar=False)
    corr_r = np.nan_to_num(corr_r)
    corr_t = np.nan_to_num(corr_t)
    correlation_shift = float(np.linalg.norm(corr_r - corr_t) / math.sqrt(max(1, rx.shape[1])))
    ks = np.array([_ks_statistic(rx[:, i], tx[:, i]) for i in range(rx.shape[1])])
    task = model.meta["task"]
    if task == "classification":
        label_shift = _classification_shift(reference.y, target.y, max(len(classes), int(max(reference.y.max(), target.y.max()) + 1)))
    else:
        label_shift = _regression_shift(reference.y, target.y)
    before = metrics(task, model, rx, reference.y, classes)
    after = metrics(task, model, tx, target.y, classes)
    drop = primary(task, after) - primary(task, before)
    feature_shift = float(delta.mean())
    base = max(1e-6, abs(primary(task, before)))
    performance_drop = float(drop / base)
    causes = []
    if feature_shift > 0.45 and shifted_fraction > 0.1:
        causes.append("covariate_shift")
    if label_shift:
        js = label_shift.get("js", 0.0)
        if js > 0.1:
            causes.append("label_or_prior_shift")
    if abs(performance_drop) > 0.15 and feature_shift <= 0.45:
        causes.append("concept_drift")
    if feature_shift > 0.45 and (label_shift is None or label_shift.get("js", 0.0) <= 0.1):
        causes.append("sensor_or_environment_gain_drift")
    if not causes:
        causes.append("stable")
    severity_score = (feature_shift / 1.0) * 0.45 + min(1.0, float(ks.mean()) / 0.3) * 0.35 + min(1.0, abs(performance_drop) / 0.3) * 0.2
    severity = "severe" if severity_score >= 0.6 else "moderate" if severity_score >= 0.32 else "mild"
    evidence = {
        "top_shifted_features": [int(i) for i in np.argsort(delta)[-8:][::-1]],
        "ks_p95": float(np.percentile(ks, 95)),
        "ks_mean": float(np.mean(ks)),
        "mean_shift_sd": float(np.mean(tx.mean(axis=0) - rx.mean(axis=0))),
        "rows_reference": int(len(rx)),
        "rows_target": int(len(tx)),
        "severity_score": float(severity_score),
        "perf_relative_change": performance_drop,
    }
    return DriftReport(feature_shift, float(delta.max()), shifted_fraction, float(ks.max()),
                       float(np.percentile(ks, 95)), correlation_shift, label_shift,
                       before, after, performance_drop, severity, causes, evidence)
