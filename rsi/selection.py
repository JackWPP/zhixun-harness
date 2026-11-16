"""Model-selection gate.

The gate is deliberately deterministic: the LLM agents propose and critique,
but promotion is decided by a fixed rule over the frozen selection split, so a
persuasive agent cannot talk the system into a regression.

Rules (per task):
    promote   primary metric improves by at least ``margin``
    retain    change is within the noise band -> keep the safer parent
    rollback  metric degrades by more than ``2 * margin`` -> candidate rejected
              explicitly, parent kept, failed proposal recorded for later rounds
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateDecision:
    decision: str
    reason: str
    delta: float

    def to_dict(self) -> dict:
        return {"decision": self.decision, "reason": self.reason, "delta": self.delta}


def _classification_stability(parent_metrics: dict | None,
                               candidate_metrics: dict | None) -> tuple[float, int] | None:
    """Return support-qualified macro-F1 delta, excluding tiny classes."""
    if not parent_metrics or not candidate_metrics:
        return None
    parent_scores = parent_metrics.get("per_class_f1") or {}
    candidate_scores = candidate_metrics.get("per_class_f1") or {}
    parent_support = parent_metrics.get("per_class_support") or {}
    candidate_support = candidate_metrics.get("per_class_support") or {}
    rows = min(int(parent_metrics.get("rows", 0)), int(candidate_metrics.get("rows", 0)))
    minimum_support = max(10, int(rows * 0.05 + 0.999999))
    labels = [label for label in sorted(parent_scores.keys() & candidate_scores.keys())
              if int(parent_support.get(label, 0)) >= minimum_support
              and int(candidate_support.get(label, 0)) >= minimum_support]
    if not labels:
        return None
    delta = sum(float(candidate_scores[label]) - float(parent_scores[label])
                for label in labels) / len(labels)
    return delta, minimum_support


def gate(task: str, parent_primary: float, candidate_primary: float, margin: float,
         parent_metrics: dict | None = None,
         candidate_metrics: dict | None = None) -> GateDecision:
    """Compare a candidate deployment against the current parent."""
    if task == "classification":
        delta = float(candidate_primary - parent_primary)
        if delta >= margin:
            stability = _classification_stability(parent_metrics, candidate_metrics)
            if stability is not None and stability[0] < 0:
                stable_delta, minimum_support = stability
                return GateDecision(
                    "retain",
                    f"macro_f1 +{delta:.6f} >= margin {margin}, but support-qualified "
                    f"class macro_f1 regressed {stable_delta:+.6f} "
                    f"(minimum support {minimum_support})",
                    delta,
                )
            return GateDecision("promote", f"macro_f1 +{delta:.6f} >= margin {margin}", delta)
        if delta <= -2.0 * margin:
            return GateDecision("rollback", f"macro_f1 {delta:+.6f} below -2x margin", delta)
        return GateDecision("retain", f"macro_f1 {delta:+.6f} within noise band", delta)
    delta = float(candidate_primary - parent_primary)
    if delta <= -margin:
        return GateDecision("promote", f"rmse {delta:+.3f} <= -margin {margin}", delta)
    if delta >= 2.0 * margin:
        return GateDecision("rollback", f"rmse {delta:+.3f} worse than +2x margin", delta)
    return GateDecision("retain", f"rmse {delta:+.3f} within noise band", delta)


def best_candidate(results, direction: str = "lower", *, task: str | None = None,
                   parent_metrics: dict | None = None, margin: float | None = None) -> dict | None:
    """Pick the candidate with the best selection primary metric.

    ``direction`` comes from the task config: "higher" for macro-F1, "lower" for
    RMSE. The direction is explicit so a classification task can never be
    silently selected in the wrong sense.
    """
    usable = [result for result in results
              if result.get("ok") and result.get("metrics", {}).get("selection")]
    if not usable:
        return None
    if task and parent_metrics is not None and margin is not None:
        parent_primary = float(parent_metrics["primary"])
        promotable = [result for result in usable
                      if gate(task, parent_primary,
                             float(result["metrics"]["selection"]["primary"]), margin,
                             parent_metrics,
                             result["metrics"]["selection"]).decision == "promote"]
        if promotable:
            usable = promotable
    key = lambda result: result["metrics"]["selection"]["primary"]
    return max(usable, key=key) if direction == "higher" else min(usable, key=key)


def relative_change(task: str, parent: float, candidate: float) -> float:
    if task == "classification":
        return float(candidate - parent)
    return float((candidate - parent) / parent) if parent else 0.0
