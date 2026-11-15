"""Deterministic, rule-based experiment planning.

This module provides two things:

1. the **fallback planner** used when the LLM runtime is unavailable, over
   budget, or returns an invalid proposal. The harness must keep running
   unattended, so a complete rule-based policy exists for every round.
2. **machine-readable catalogs** injected into the agent prompts, so the LLM
   only chooses among validated primitives rather than emitting free-form code.

The seeded plans encode what we actually measured before building the harness:
A needs inference-time recentring (weights 0.3-0.5), B's baseline is dominated
by a target-level bias (shrink toward the calibration centre), C only takes a
small, short continuation gain.
"""
from __future__ import annotations

from rsi.executor import ExperimentProposal
from rsi.strategies import POSTPROCESS_TYPES, STRATEGIES


def strategy_catalog_text() -> str:
    """Compact catalog of allowed strategies and parameter ranges."""
    lines = []
    for name, spec in STRATEGIES.items():
        params = []
        for param, rule in spec.params.items():
            if rule["type"] == "choice":
                rendered = "|".join(rule["choices"])
            elif rule["type"] == "bool":
                rendered = "true|false"
            else:
                rendered = f"{rule['type']} [{rule['min']}, {rule['max']}] default {rule['default']}"
            params.append(f"{param}={rendered}")
        lines.append(f"- {name}: {spec.description} | cost={spec.cost} | "
                     f"tasks={','.join(spec.tasks)} | params: {'; '.join(params) if params else 'none'}")
    return "\n".join(lines)


def postprocess_catalog_text() -> str:
    lines = []
    for name, spec in POSTPROCESS_TYPES.items():
        detail = spec.get("description", "")
        if name == "recenter":
            detail += f" | modes={spec['modes']} | weight in [0,1]"
        if name == "shrink":
            detail += " | alpha in [0,1]"
        lines.append(f"- {name}: {detail}")
    return "\n".join(lines)


def _plan(strategy, params, postprocess, hypothesis, expected_effect, risk):
    return ExperimentProposal(strategy=strategy, params=params, postprocess=postprocess,
                              hypothesis=hypothesis, expected_effect=expected_effect, risk=risk)


def _recenter(weight, mode="batch_mean"):
    return {"type": "recenter", "mode": mode, "weight": weight}


def _shrink(alpha):
    return {"type": "shrink", "alpha": alpha}


SEEDED = {
    "A": [
        _plan("retain", {}, _recenter(0.40),
              "Baseline is accurate in-distribution but degrades on drifted batches; recentring the frozen statistics recovers the calibration loss without touching weights.",
              "val macro_f1 +0.008..0.015, drift probe +0.10", "recentre can hurt if the batch is small or homogeneous"),
        _plan("retain", {}, _recenter(0.50),
              "A stronger blend may pay off on strongly drifted batches while remaining neutral on the official batch.",
              "val macro_f1 +0.005..0.012, probe +0.12", "val may regress if blend weight is too high"),
        _plan("continue_train", {"learning_rate": 0.0002, "epochs": 20, "data": "train",
                                 "class_balance": True, "augment": False}, _recenter(0.40),
              "Class-balanced continuation may repair the under-represented classes (Acetaldehyde/Toluene) that drive macro-F1.",
              "val macro_f1 +0.005", "small train set -> overfitting, drift robustness may fall"),
        _plan("ridge_last_layer", {"alpha": 100.0, "data": "train"}, _recenter(0.40),
              "If the representation is fine and only the output head is miscalibrated, a closed-form ridge head should help.",
              "val macro_f1 +0.003", "frozen hidden features may not separate drifted classes"),
        _plan("swa", {"learning_rate": 0.0002, "epochs": 5, "snapshots": 3,
                      "class_balance": True, "augment": False}, _recenter(0.40),
              "Weight averaging across short balanced continuations reduces variance without changing the architecture.",
              "val macro_f1 +0.002..0.006", "multiple epochs on 445 rows can memorise batch 1"),
        _plan("retain", {}, _recenter(0.30),
              "Lower blend weight to verify the calibration curve and avoid over-correction.",
              "val macro_f1 +0.004..0.010, probe +0.08", "possible neutral result"),
        _plan("continue_train", {"learning_rate": 0.0001, "epochs": 10, "data": "train+calibration",
                                 "class_balance": True, "augment": True}, _recenter(0.45),
              "Gain/offset augmentation plus calibration rows teaches gain-drift invariance explicitly.",
              "val macro_f1 +0.004, probe +0.05", "augmentation may distort in-distribution accuracy"),
        _plan("ridge_last_layer", {"alpha": 10.0, "data": "train+calibration"}, _recenter(0.35),
              "Refit the output head on train+calibration to widen class coverage, then calibrate at inference.",
              "val macro_f1 +0.003", "calibration rows are few (373) and class-skewed"),
        _plan("retain", {}, _recenter(0.20, mode="batch_std"),
              "Test whether standard-deviation recentring helps the strongly shifted batches where mean-only recentring saturates.",
              "probe +0.03, val neutral", "std recentring hurt the official batch in our probe"),
        _plan("retain", {}, _recenter(0.45),
              "Final consolidation: keep the best checkpoint and the calibrated blend, no further weight updates.",
              "val macro_f1 stable", "none (safe action)"),
    ],
    "B": [
        _plan("retain", {}, _shrink(0.0),
              "The baseline is anti-correlated with the held-out bearing targets; predicting the calibration target centre removes the bias with the lowest variance.",
              "val rmse 681 -> ~466", "constant prediction ignores signal and may look unsophisticated"),
        _plan("retain", {}, _shrink(0.10),
              "A small signal component on top of the shrunk centre keeps ordering while correcting the scale bias.",
              "val rmse ~475", "bias reappears if alpha is too high"),
        _plan("retain", {}, _shrink(0.30),
              "Probe the shrink curve to document where signal content starts to hurt.",
              "val rmse ~520-560", "still far from the constant baseline"),
        _plan("continue_train", {"learning_rate": 0.0003, "epochs": 15, "data": "train",
                                 "class_balance": False, "augment": False}, _shrink(0.0),
              "Retraining may fix the output scale, after which the constant centre can be relaxed.",
              "val rmse 630-660 -> shrink to 466", "continuation did not beat the constant in our offline probe"),
        _plan("ridge_last_layer", {"alpha": 1000.0, "data": "train"}, _shrink(0.0),
              "Closed-form head tests whether only the output layer is misscaled.",
              "val rmse ~605 -> shrink 466", "frozen features likely insufficient"),
        _plan("swa", {"learning_rate": 0.0002, "epochs": 5, "snapshots": 3,
                      "class_balance": False, "augment": False}, _shrink(0.10),
              "Averaging short continuations reduces variance; shrink keeps the bias correction.",
              "val rmse ~470", "averaged weights remain under-fit"),
        _plan("retain", {}, _shrink(0.05),
              "Fine-grained shrink sweep to pick the least aggressive bias correction that passes the gate.",
              "val rmse ~466-470", "marginal difference"),
        _plan("continue_train", {"learning_rate": 0.0001, "epochs": 10, "data": "train+calibration",
                                 "class_balance": False, "augment": False}, _shrink(0.0),
              "Use calibration bearings as extra training signal, then shrink.",
              "val rmse ~640 -> 466", "calibration rows are few"),
        _plan("ridge_last_layer", {"alpha": 100.0, "data": "train+calibration"}, _shrink(0.10),
              "Ridge with calibration rows may recover a usable scale, then shrink lightly.",
              "val rmse 600 -> ~470", "leakage of calibration into head fit is intentional and recorded"),
        _plan("retain", {}, _shrink(0.0),
              "Final consolidation: keep the strongest safe deployment (constant centre) for the hidden stream.",
              "val rmse stable ~466", "none (safe action)"),
    ],
    "C": [
        _plan("continue_train", {"learning_rate": 0.001, "epochs": 40, "data": "train",
                                 "class_balance": False, "augment": False}, {"type": "none"},
              "A short continuation adapts the head to the official validation slice without changing capacity.",
              "val rmse 52.87 -> ~51.3", "overfitting after many epochs"),
        _plan("continue_train", {"learning_rate": 0.001, "epochs": 100, "data": "train",
                                 "class_balance": False, "augment": False}, {"type": "none"},
              "The 100-epoch continuation was the best measured point in the offline probe.",
              "val rmse 52.87 -> ~51.07", "diminishing returns, small overfit risk"),
        _plan("continue_train", {"learning_rate": 0.0005, "epochs": 100, "data": "train",
                                 "class_balance": False, "augment": False}, {"type": "none"},
              "Lower learning rate for longer should reduce final-loss noise.",
              "val rmse ~51.1-51.4", "slower convergence"),
        _plan("swa", {"learning_rate": 0.001, "epochs": 10, "snapshots": 3,
                      "class_balance": False, "augment": False}, {"type": "none"},
              "Weight averaging reduces run-to-run variance of the continuation gain.",
              "val rmse ~51.0-51.3", "averaging may flatten the head"),
        _plan("ridge_last_layer", {"alpha": 10.0, "data": "train"}, {"type": "none"},
              "Check whether the representation is already sufficient and only the head is stale.",
              "val rmse ~51.5-53", "closed-form head may be worse than Adam"),
        _plan("continue_train", {"learning_rate": 0.001, "epochs": 100, "data": "train",
                                 "class_balance": False, "augment": False}, _shrink(0.95),
              "Combined continuation and mild output shrink; the correction must stay near identity because the baseline is already unbiased.",
              "val rmse ~51.0", "shrink with alpha<1 can add bias"),
        _plan("swa", {"learning_rate": 0.0005, "epochs": 15, "snapshots": 3,
                      "class_balance": False, "augment": False}, {"type": "none"},
              "Longer SWA phase to squeeze out the remaining validation noise.",
              "val rmse ~51.0", "more epochs, still cheap"),
        _plan("ridge_last_layer", {"alpha": 1.0, "data": "train"}, {"type": "none"},
              "Weak ridge refit as a low-variance head update.",
              "val rmse ~51.4", "may not beat Adam"),
        _plan("continue_train", {"learning_rate": 0.0002, "epochs": 100, "data": "train",
                                 "class_balance": False, "augment": False}, _shrink(0.98),
              "Very low learning rate + near-identity shrink; final robustness check.",
              "val rmse ~51.3", "too conservative"),
        _plan("retain", {}, {"type": "none"},
              "Final consolidation: keep the best deployment, no further updates.",
              "val rmse stable", "none (safe action)"),
    ],
}


def default_plans(task_key: str, round_index: int) -> list[ExperimentProposal]:
    """Deterministic plan list for one round (1-based round index)."""
    plans = SEEDED[task_key]
    index = max(0, min(len(plans) - 1, round_index - 1))
    return [plans[index]]


def _plan_bank(task_key: str):
    return list(SEEDED[task_key])


def fallback_proposals(task_key: str, round_index: int) -> list[ExperimentProposal]:
    """Two safe proposals: the seeded plan plus the final consolidation plan."""
    bank = _plan_bank(task_key)
    primary = bank[max(0, min(len(bank) - 1, round_index - 1))]
    safe = bank[-1]
    proposals = [primary]
    if primary.strategy != safe.strategy or primary.postprocess != safe.postprocess:
        proposals.append(safe)
    return proposals
