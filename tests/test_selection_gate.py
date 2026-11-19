"""Selection-gate tests: promotion requires paired evidence, not point estimates."""
from __future__ import annotations

import numpy as np

from rsi.platform.submission import _block_scores, _gate_choose


def _plans(parent_pred, candidate_preds, kind="recenter"):
    return [
        {"kind": "retain", "predicted": np.asarray(parent_pred)},
        *[{"kind": f"{kind}-{index}", "predicted": np.asarray(pred)}
          for index, pred in enumerate(candidate_preds)],
    ]


def _labels(n=512, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n)


def test_gate_promotes_clear_improvement():
    labels = _labels()
    rng = np.random.default_rng(1)
    parent = labels.copy()
    wrong = rng.choice(len(labels), size=int(0.3 * len(labels)), replace=False)
    parent[wrong] = 1 - parent[wrong]
    candidate = labels.copy()  # perfect
    plans = _plans(parent, [candidate])
    position, decisions = _gate_choose(
        plans, "classification", "higher", 0.0, labels, ["a", "b"])
    assert position == 1
    assert decisions[1].decision == "promote"
    assert decisions[1].delta > 0


def test_gate_retains_noisy_non_improvement():
    labels = _labels(seed=3)
    parent = labels.copy()
    rng = np.random.default_rng(4)
    wrong = rng.choice(len(labels), size=int(0.15 * len(labels)), replace=False)
    parent[wrong] = 1 - parent[wrong]
    # candidate is a little better on the point estimate but flips a different
    # random subset -> paired evidence should not clear a positive margin.
    candidate = labels.copy()
    wrong = rng.choice(len(labels), size=int(0.13 * len(labels)), replace=False)
    candidate[wrong] = 1 - candidate[wrong]
    plans = _plans(parent, [candidate])
    position, decisions = _gate_choose(
        plans, "classification", "higher", 0.02, labels, ["a", "b"])
    assert position == 0
    assert decisions[1].decision in {"retain", "abstain"}


def test_gate_abstains_with_too_few_blocks():
    labels = _labels(25, seed=5)
    plans = _plans(labels, [labels])
    position, decisions = _gate_choose(
        plans, "classification", "higher", 0.0, labels, ["a", "b"])
    assert position == 0
    assert decisions[1] is not None
    assert decisions[1].decision == "abstain"


def test_gate_regression_prefers_lower_rmse():
    rng = np.random.default_rng(6)
    labels = rng.normal(500.0, 80.0, size=512)
    parent = labels + rng.normal(0, 30, size=512)
    candidate = labels + rng.normal(0, 5, size=512)
    plans = _plans(parent, [candidate])
    position, decisions = _gate_choose(
        plans, "regression", "lower", 1.0, labels, [])
    assert position == 1
    assert decisions[1].decision == "promote"
    # improvement is reported as ``parent - candidate`` (positive = better)
    assert decisions[1].delta > 0


def test_block_scores_cover_all_rows():
    labels = _labels(200, seed=7)
    scores = _block_scores("classification", labels, labels, ["a", "b"], block_size=50)
    assert len(scores) == 4
    assert all(score > 0 for score in scores)
