import numpy as np

from rsi.eval.gate import evaluate_evidence, select_arm
from rsi.eval.stats import block_bootstrap_ci, holm_adjust, paired_block_bootstrap


def test_block_bootstrap_brackets_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.2, size=40)
    blocks = [f"b{i // 8}" for i in range(40)]
    ci = block_bootstrap_ci(values, blocks, reps=400, seed=1)
    assert ci.lo < ci.mean < ci.hi
    assert ci.blocks == 5


def test_paired_gate_promotes_clear_improvement():
    parent = [0.40, 0.42, 0.38, 0.41, 0.39, 0.40]
    candidate = [0.55, 0.57, 0.52, 0.56, 0.54, 0.55]
    blocks = ["b1", "b2", "b3", "b4", "b5", "b6"]
    decision = evaluate_evidence(parent, candidate, blocks, margin=0.01, reps=300, seed=2)
    assert decision.decision == "promote"
    assert decision.delta > 0


def test_paired_gate_rolls_back_reliable_regression():
    parent = [0.80, 0.79, 0.81, 0.80, 0.82, 0.80]
    candidate = [0.60, 0.61, 0.59, 0.60, 0.58, 0.61]
    blocks = ["b1", "b2", "b3", "b4", "b5", "b6"]
    decision = evaluate_evidence(parent, candidate, blocks, margin=0.01, reps=300, seed=3)
    assert decision.decision == "rollback"


def test_paired_gate_retains_when_effect_inside_noise():
    parent = [0.70, 0.69, 0.71, 0.70, 0.70, 0.69]
    candidate = [0.701, 0.695, 0.706, 0.700, 0.702, 0.699]
    blocks = ["b1", "b2", "b3", "b4", "b5", "b6"]
    decision = evaluate_evidence(parent, candidate, blocks, margin=0.01, reps=300, seed=4)
    assert decision.decision in {"retain", "abstain"}


def test_gate_abstains_with_too_few_blocks():
    decision = evaluate_evidence([0.5], [0.6], ["b1"], margin=0.01)
    assert decision.decision == "abstain"


def test_select_arm_prefers_promotable():
    parent = [0.40, 0.42, 0.38, 0.41, 0.39, 0.40]
    good = [0.55, 0.57, 0.52, 0.56, 0.54, 0.55]
    bad = [0.30, 0.31, 0.29, 0.30, 0.28, 0.31]
    blocks = ["b1", "b2", "b3", "b4", "b5", "b6"]
    first = evaluate_evidence(parent, good, blocks, margin=0.01, reps=300, seed=5, arm="good")
    second = evaluate_evidence(parent, bad, blocks, margin=0.01, reps=300, seed=5, arm="bad")
    chosen = select_arm([first, second])
    assert chosen.arm == "good"


def test_holm_adjust_monotone():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[2] <= adjusted[1]
    assert all(0 <= p <= 1 for p in adjusted)
