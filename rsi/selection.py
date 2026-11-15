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


def gate(task: str, parent_primary: float, candidate_primary: float, margin: float) -> GateDecision:
    """Compare a candidate deployment against the current parent."""
    if task == "classification":
        delta = float(candidate_primary - parent_primary)
        if delta >= margin:
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


def best_candidate(results, direction: str = "lower") -> dict | None:
    """Pick the candidate with the best selection primary metric.

    ``direction`` comes from the task config: "higher" for macro-F1, "lower" for
    RMSE. The direction is explicit so a classification task can never be
    silently selected in the wrong sense.
    """
    usable = [result for result in results
              if result.get("ok") and result.get("metrics", {}).get("selection")]
    if not usable:
        return None
    key = lambda result: result["metrics"]["selection"]["primary"]
    return max(usable, key=key) if direction == "higher" else min(usable, key=key)


def relative_change(task: str, parent: float, candidate: float) -> float:
    if task == "classification":
        return float(candidate - parent)
    return float((candidate - parent) / parent) if parent else 0.0
