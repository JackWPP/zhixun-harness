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
