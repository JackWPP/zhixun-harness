"""Drift characterisation: covariate, label/prior, and performance drift.

The harness never trusts a single number: a drift report names the evidence, a
severity, a likely mechanism, and the strategies that are admissible given it.
All statistics are computed on public data only (unlabelled probes are allowed
for covariate statistics; labels are used when the split provides them).

The report is deliberately a superset of the original contract (v0.5.1):
``score``, ``drift_type``, ``attribution``, ``recommended_actions``,
``knowledge_hints`` and ``next_checks`` were appended as defaulted fields so
that positional construction, ``to_dict()`` consumers and every existing
formula/severity threshold stay compatible.  ``to_dict()`` additionally exposes
a ``type`` alias for the offline knowledge-base lookup used by the kernel brief,
and ``evidence`` carries a compact copy of the machine-readable hints so the
agent sees them even when a consumer filters the report to a whitelist.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rsi.metrics import primary, metrics

# --- stable severity contract (do not re-tune thresholds in a drift patch) ----
_SEVERITY_SEVERE = 0.6
_SEVERITY_MODERATE = 0.32
_WEIGHT_FEATURE_SHIFT = 0.45
_WEIGHT_KS = 0.35
_WEIGHT_PERFORMANCE = 0.20

_TOP_FEATURES = 8          # legacy evidence.top_shifted_features width
_ATTRIBUTION_LIMIT = 10    # enriched attribution rows kept in the report
_EVIDENCE_ATTRIBUTION = 5  # compact attribution rows mirrored into evidence
_ACTION_LIMIT = 4          # recommended actions kept in the report


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
    # --- v0.5.1 contract: keep field order and positional construction ------
    feature_shift: float
    max_feature_shift: float
    shifted_fraction: float
    ks_max: float
    ks_p95: float
    correlation_shift: float
    label_shift: dict | None
    performance_before: dict | None
    performance_target: dict | None
    performance_drop: float | None
    severity: str
    causes: list[str]
    evidence: dict
    # --- enrichment (defaulted: old call sites remain valid) ----------------
    score: float = 0.0
    drift_type: str = "stable"
    attribution: list[dict] = field(default_factory=list)
    recommended_actions: list[dict] = field(default_factory=list)
    knowledge_hints: list[str] = field(default_factory=list)
    next_checks: list[str] = field(default_factory=list)

    def to_dict(self):
        payload = {key: getattr(self, key) for key in self.__dataclass_fields__}
        # alias consumed by knowledge lookups / dashboards: ``type`` == drift_type
        payload["type"] = self.drift_type
        return payload


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


# ---------------------------------------------------------------------------
# enrichment helpers
# ---------------------------------------------------------------------------
def _feature_attribution(rx, tx, delta, ks):
    """Rank features by a composite evidence score (shift, distribution, spread).

    Returns plain-python dict rows sorted by ``-score`` so the value is JSON
    serialisable and can be copied into records/briefs without conversion.
    """
    rows = []
    for i in range(rx.shape[1]):
        col_r = rx[:, i]
        col_t = tx[:, i]
        sd_r = float(np.std(col_r))
        sd_t = float(np.std(col_t))
        std_ratio = sd_t / max(sd_r, 1e-9)
        js = _js_divergence(col_r, col_t)
        mean_delta = float(np.mean(col_t) - np.mean(col_r))
        spread_penalty = min(1.0, abs(math.log(max(std_ratio, 1e-9))) / 0.5)
        score = (0.5 * min(1.0, float(delta[i]) / 3.0)
                 + 0.3 * min(1.0, float(ks[i]) / 0.3)
                 + 0.2 * spread_penalty)
        rows.append({
            "feature": int(i),
            "score": round(float(score), 4),
            "mean_shift_sd": round(float(delta[i]), 4),
            "ks": round(float(ks[i]), 4),
            "js": round(float(js), 4),
            "std_ratio": round(float(std_ratio), 4),
            "mean_before": round(float(np.mean(col_r)), 6),
            "mean_target": round(float(np.mean(col_t)), 6),
            "direction": ("up" if mean_delta > 0 else "down" if mean_delta < 0 else "flat"),
        })
    rows.sort(key=lambda row: (-row["score"], row["feature"]))
    return rows


def _drift_type(causes):
    """Map the legacy cause list to a single stable machine label."""
    if not causes or causes == ["stable"]:
        return "stable"
    if "concept_drift" in causes:
        return "concept_drift"
    if "label_or_prior_shift" in causes:
        return "label_prior_shift"
    if "sensor_or_environment_gain_drift" in causes:
        return "sensor_gain_drift"
    if "covariate_shift" in causes:
        return "covariate_shift"
    return "mixed_drift"


def _recommended_actions(task, drift_type, feature_shift, ks_p95, performance_drop,
                         label_shift, severity):
    """Propose only experiments/calibrations that are acceptable to the guard.

    Every entry uses the exact ``{strategy, params, postprocess}`` shape that
    ``propose_experiment`` expects, plus human-readable rationale/risk.  The list
    is advisory: the kernel must still record a hypothesis and let the gate
    decide, and validation/report labels are never used for fitting.
    """
    actions = []
    reg = task == "regression"
    label_js = float((label_shift or {}).get("js", 0.0) or 0.0)
    severe = severity == "severe"
    drop = abs(float(performance_drop or 0.0))

    def add(strategy, params, postprocess, rationale, risk, expected, confidence="medium"):
        if len(actions) >= _ACTION_LIMIT:
            return
        actions.append({
            "strategy": strategy,
            "params": dict(params),
            "postprocess": postprocess or {"type": "none"},
            "rationale": rationale,
            "risk": risk,
            "expected_effect": expected,
            "confidence": confidence,
        })

    if drift_type in ("sensor_gain_drift", "covariate_shift"):
        mode = "batch_std" if float(feature_shift) >= 0.8 else "batch_mean"
        weight = 0.5 if not severe else 0.75
        add("retain",
            {},
            {"type": "recenter", "mode": mode, "weight": weight},
            "输入协变量/传感器增益漂移、输出校准仍可信：先做推理期覆盖校准，不改 checkpoint。",
            "low", "用 selection 窗口验证：指标恢复则 retain+recenter 直接交付。",
            "high")
        if reg:
            add("retain", {},
                {"type": "shrink", "alpha": 0.25 if drop < 0.3 else 0.4},
                "回归输出存在系统性偏置时，向 train-target 均值收缩可先止跌，且只用官方 train 标签拟合。",
                "low", "MAE/RMSE 先回落，再决定是否续训。")
        elif not reg:
            add("retain",
                {},
                {"type": "recenter", "mode": mode, "weight": min(0.9, weight + 0.2)},
                "若 batch_mean 权重不足，用更激进的覆盖校准作为单变量对照（一次只改一个因素）。",
                "medium", "确认校准强度与恢复幅度的单调关系，避免过校。")
    if drift_type == "label_prior_shift":
        add("continue_train",
            {"learning_rate": 3e-4, "epochs": 15, "data": "train",
             "class_balance": not reg, "augment": False},
            {"type": "none"},
            "标签/先验分布变化：只重加权官方 train 标签做轻度续训，禁止拟合目标标签。",
            "medium", "宏 F1/次级类召回回升且无严重漏检；若目标窗口缺类则预期不可恢复。")
        if reg:
            add("retain", {}, {"type": "shrink", "alpha": 0.3},
                "回归标签/先验漂移先向 train-target 均值收缩止跌（只用官方 train 标签）。",
                "low", "RMSE/MAE 先回落；若 bias 明显变小再考虑续训。")
    if drift_type == "concept_drift":
        add("continue_train_staged",
            {"stage1_learning_rate": 5e-4, "stage1_epochs": 200,
             "stage2_learning_rate": 1e-4, "stage2_epochs": 50,
             "class_balance": not reg, "augment": False},
            {"type": "none"},
            "输入分布基本稳定但 P(y|x) 变化：用官方 train 做两阶段低成本续训，先验状态不动。",
            "medium", "连续两轮 selection 指标提升且 margin 之外仍为正；否则回滚。")
        add("ridge_last_layer", {"alpha": 10.0, "data": "train"}, {"type": "none"},
            "闭式重拟输出层，用来区分瓶颈在 head 还是表示（低成本诊断实验）。",
            "low", "若 head 重拟即恢复，说明表示仍可用，无需更大改动。")
    if drift_type == "mixed_drift":
        add("retain", {}, {"type": "recenter", "mode": "batch_mean", "weight": 0.5},
            "混合漂移先做零风险校准止跌，再逐项归因（一次一个变量）。",
            "low", "先确认可恢复部分，再做续训对照。")
        add("continue_train",
            {"learning_rate": 2e-4, "epochs": 10, "data": "train",
             "class_balance": not reg, "augment": False},
            {"type": "none"},
            "校准不足时的小步续训对照，参数保守以控制方差。",
            "medium", "对照 recenter-only 判断续训是否带来额外增益。")
    if not actions or drift_type == "stable":
        add("retain", {}, {"type": "none"},
            "报告为稳定/轻度：默认保留父模型，优先复评样本量是否充足。",
            "low", "selection 与 report 窗口一致则无需动作。", "high")

    if severe and len(actions) < _ACTION_LIMIT:
        add("retain", {}, {"type": "none"},
            "严重漂移先跑负对照（retain+none）复评一次，排除评测窗口噪声/小样本造成的假严重。",
            "none", "负对照应接近父模型基线；若差异大则先修评测而非改模型。", "high")
    # tiny hint surface for the human reader; machine hints live in knowledge_hints
    if float(ks_p95) > 0.35:
        for action in actions:
            action.setdefault("note", "ks_p95 偏高：先核对传感器单位/量程与训练统计量冻结值。")
            break
    _ = label_js  # kept for API symmetry / future rules
    return actions[: _ACTION_LIMIT]


def _knowledge_hints(task, drift_type, causes, severity, feature_shift, performance_drop):
    """Short tag queries the agent can feed to ``search_knowledge(tags=...)``."""
    hints = []
    reg = task == "regression"
    if drift_type == "stable":
        hints += ["drift", "taxonomy", "detection"]
    if drift_type in ("sensor_gain_drift", "covariate_shift"):
        hints += ["sensor", "gain", "baseline", "recenter", "calibration", "covariate"]
    if drift_type == "label_prior_shift":
        hints += ["label", "prior", "class-balance", "performance"]
    if drift_type == "concept_drift":
        hints += ["concept", "drift", "residual", "performance", "attribution"]
    if drift_type == "mixed_drift":
        hints += ["drift", "attribution", "feature"]
    if reg:
        hints += ["rul", "regression", "shrink"]
        if abs(float(performance_drop or 0.0)) > 0.15:
            hints += ["damage", "calibration"]
    if severity in ("moderate", "severe"):
        hints += ["experiment", "falsification"]
    if "sensor_or_environment_gain_drift" in causes:
        hints += ["sensor", "gain"]
    seen = []
    for hint in hints:
        if hint not in seen:
            seen.append(hint)
    return seen[:8]


def _next_checks(task, drift_type, evidence, label_shift, rows_target):
    """Ordered, cheap sanity checks before spending an experiment budget."""
    checks = []
    missing = list((label_shift or {}).get("missing_classes") or [])
    if missing:
        checks.append(f"目标窗口缺少类别 {missing}：先查采样/切分覆盖，分类指标下降不代表表示退化。")
    if rows_target < 200:
        checks.append(f"目标窗口仅 {int(rows_target)} 行：先复核统计量与评测噪声，避免小样本假漂移。")
    if "sensor_or_environment_gain_drift" in evidence.get("causes", []) or drift_type == "sensor_gain_drift":
        checks.append("逐特征核对 mean_target/mean_before 与 std_ratio（见 attribution）：增益漂移通常整批同向缩放。")
    if drift_type == "concept_drift":
        checks.append("输入分布基本稳定：用带标签目标窗口看残差是否随 y 变化，区分概念漂移与标注噪声。")
    if task == "regression" and evidence.get("perf_relative_change", 0.0) and float(evidence.get("perf_relative_change", 0.0)) > 0.15:
        checks.append("RUL 类指标对长尾敏感：先看 bias 与分区间误差，再决定 shrink/damage_shrink/续训。")
    checks.append("所有候选只在 selection 窗口比较；report/validation 标签不参与拟合（合规边界）。")
    return checks[:5]


def analyse(model, reference, target, classes=()):
    """Compare parent model behaviour and input statistics on two public splits."""
    rx, tx = reference.x, target.x
    sd = rx.std(axis=0)
    sd[sd < 1e-12] = 1.0
    delta = np.abs(tx.mean(axis=0) - rx.mean(axis=0)) / sd
    shifted_fraction = float(np.mean(delta > 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
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
    severity_score = (feature_shift / 1.0) * _WEIGHT_FEATURE_SHIFT + min(1.0, float(ks.mean()) / 0.3) * _WEIGHT_KS + min(1.0, abs(performance_drop) / 0.3) * _WEIGHT_PERFORMANCE
    severity = "severe" if severity_score >= _SEVERITY_SEVERE else "moderate" if severity_score >= _SEVERITY_MODERATE else "mild"
    attribution = _feature_attribution(rx, tx, delta, ks)
    drift_type = _drift_type(causes)
    severity_components = {
        "feature_shift": round(feature_shift * _WEIGHT_FEATURE_SHIFT, 4),
        "ks": round(min(1.0, float(ks.mean()) / 0.3) * _WEIGHT_KS, 4),
        "performance": round(min(1.0, abs(performance_drop) / 0.3) * _WEIGHT_PERFORMANCE, 4),
    }
    evidence = {
        "top_shifted_features": [int(i) for i in np.argsort(delta)[-8:][::-1]],
        "ks_p95": float(np.percentile(ks, 95)),
        "ks_mean": float(np.mean(ks)),
        "mean_shift_sd": float(np.mean(tx.mean(axis=0) - rx.mean(axis=0))),
        "rows_reference": int(len(rx)),
        "rows_target": int(len(tx)),
        "severity_score": float(severity_score),
        "perf_relative_change": performance_drop,
        # compact copies so whitelisted consumers still see the enrichment
        "drift_type": drift_type,
        "causes": list(causes),
        "severity_components": severity_components,
        "attribution_top": attribution[:_EVIDENCE_ATTRIBUTION],
        "knowledge_hints": _knowledge_hints(task, drift_type, causes, severity,
                                            feature_shift, performance_drop),
    }
    recommended_actions = _recommended_actions(task, drift_type, feature_shift,
                                               float(np.percentile(ks, 95)),
                                               performance_drop, label_shift, severity)
    next_checks = _next_checks(task, drift_type, evidence, label_shift, len(tx))
    return DriftReport(feature_shift, float(delta.max()), shifted_fraction, float(ks.max()),
                       float(np.percentile(ks, 95)), correlation_shift, label_shift,
                       before, after, performance_drop, severity, causes, evidence,
                       score=float(severity_score), drift_type=drift_type,
                       attribution=attribution, recommended_actions=recommended_actions,
                       knowledge_hints=evidence["knowledge_hints"],
                       next_checks=next_checks)
