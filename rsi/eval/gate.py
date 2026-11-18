"""Evidence gate: promote only when the paired CI clears the margin.

This is the missing "trustworthy selection" layer: instead of picking the arm
with the best point estimate on the public validation slice (which the A task
ranked *against* the hidden blocks), a candidate must clear a multiple-testing
aware, uncertainty-aware gate.  The gate is deliberately conservative and has a
``retain`` (no change) and ``abstain`` (not enough evidence) outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from rsi.eval.stats import Interval, paired_block_bootstrap


@dataclass(frozen=True)
class GateDecision:
    decision: str          # promote / rollback / retain / abstain
    reason: str
    delta: float
    interval: Interval
    margin_eff: float
    direction: str
    arm: str = ""
    family: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "delta": round(self.delta, 6),
            "interval": self.interval.to_dict(),
            "margin_eff": round(self.margin_eff, 6),
            "direction": self.direction,
            "arm": self.arm,
            "family": self.family,
            "extra": self.extra,
        }


def evaluate_evidence(parent: Sequence[float], candidate: Sequence[float],
                      blocks: Sequence, *, margin: float, direction: str = "higher",
                      arm: str = "", family: str = "", reps: int = 1000,
                      alpha: float = 0.05, margin_scale: float = 1.5,
                      min_blocks: int = 3, seed: int = 0) -> GateDecision:
    """Gate one candidate against its parent.

    ``margin_eff = max(margin, margin_scale * standard_error)`` so that the
    required improvement grows when the per-block evidence is noisy.
    """
    if len(parent) != len(candidate):
        raise ValueError("parent and candidate must align")
    if len(set(blocks)) < min_blocks:
        return GateDecision("abstain", "not enough blocks", 0.0,
                            Interval(0.0, 0.0, 0.0, alpha, len(set(blocks))),
                            float(margin), direction, arm, family)
    ci = paired_block_bootstrap(parent, candidate, blocks, direction=direction,
                                reps=reps, alpha=alpha, seed=seed)
    # Standard error from the bootstrap distribution width (normal approx).
    se = ci.width / (2 * 1.96) if ci.width else 0.0
    margin_eff = max(float(margin), margin_scale * se)
    delta = ci.mean
    if delta >= margin_eff and ci.lo > 0:
        return GateDecision("promote", "paired CI lower bound clears zero and effect >= margin_eff",
                            delta, ci, margin_eff, direction, arm, family)
    if delta <= -margin_eff and ci.hi < 0:
        return GateDecision("rollback", "candidate is reliably worse than parent",
                            delta, ci, margin_eff, direction, arm, family)
    if ci.width > 2 * abs(delta) + 1e-12:
        return GateDecision("abstain", "interval too wide relative to effect",
                            delta, ci, margin_eff, direction, arm, family)
    return GateDecision("retain", "effect does not clear margin_eff",
                        delta, ci, margin_eff, direction, arm, family)


def select_arm(decisions: Sequence[GateDecision]) -> GateDecision:
    """Pick the best promotable arm; otherwise retain the parent."""
    promotable = [d for d in decisions if d.decision == "promote"]
    if promotable:
        return max(promotable, key=lambda d: d.delta)
    retains = [d for d in decisions if d.decision == "retain"]
    if retains:
        return max(retains, key=lambda d: d.delta)
    abstain = [d for d in decisions if d.decision == "abstain"]
    if abstain:
        return abstain[0]
    return decisions[0]
