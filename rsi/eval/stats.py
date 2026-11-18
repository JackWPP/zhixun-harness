"""Bootstrap statistics for drift-aware model selection.

The competition scores a hidden stream; the only honest way to compare two
candidate arms is to treat their per-block differences as paired observations
and to require the uncertainty interval (not just the point estimate) to clear
the deployment margin.  Helpers here are dependency-free and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

Stat = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class Interval:
    mean: float
    lo: float
    hi: float
    alpha: float
    blocks: int

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 6),
            "lo": round(self.lo, 6),
            "hi": round(self.hi, 6),
            "alpha": self.alpha,
            "blocks": self.blocks,
        }


def _bootstrap_indices(blocks: Sequence, reps: int, rng: np.random.Generator) -> list[np.ndarray]:
    n = len(blocks)
    return [rng.integers(0, n, size=n) for _ in range(reps)]


def block_bootstrap_ci(values: Sequence[float], blocks: Sequence, *, reps: int = 1000,
                       alpha: float = 0.05, stat: Stat | None = None,
                       seed: int = 0) -> Interval:
    """Bootstrap a statistic by resampling whole blocks (cluster bootstrap)."""
    values = np.asarray(list(values), dtype=np.float64)
    blocks = list(blocks)
    if len(values) != len(blocks):
        raise ValueError("values and blocks must have the same length")
    if len(values) == 0:
        raise ValueError("empty sample")
    fn = stat or (lambda x: float(np.mean(x)))
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for i, idx in enumerate(_bootstrap_indices(blocks, reps, rng)):
        means[i] = fn(values[idx])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(fn(values)), float(lo), float(hi), alpha, len(set(blocks)))


def paired_block_bootstrap(parent: Sequence[float], candidate: Sequence[float],
                           blocks: Sequence, *, direction: str = "higher",
                           reps: int = 1000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """CI of the paired improvement candidate - parent (positive = better)."""
    if direction == "higher":
        diffs = np.asarray(candidate, dtype=np.float64) - np.asarray(parent, dtype=np.float64)
    elif direction == "lower":
        diffs = np.asarray(parent, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    else:
        raise ValueError(f"unknown direction: {direction}")
    return block_bootstrap_ci(diffs, blocks, reps=reps, alpha=alpha, seed=seed)


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment (family-wise error control)."""
    values = list(pvalues)
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [1.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def empirical_coverage(true_delta: float, intervals: Sequence[Interval]) -> float:
    """Fraction of intervals that contain the reference delta."""
    if not intervals:
        return 0.0
    hits = sum(1 for ci in intervals if ci.lo <= true_delta <= ci.hi)
    return hits / len(intervals)
