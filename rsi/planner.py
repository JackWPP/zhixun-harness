"""Deterministic, rule-based experiment planning.

This module provides two things:

1. the **fallback planner** used when the LLM runtime is unavailable, over
   budget, or returns an invalid proposal. The harness must keep running
   unattended, so a complete rule-based policy exists for every round.
2. **machine-readable catalogs** injected into the agent prompts, so the LLM
   only chooses among validated primitives rather than emitting free-form code.

The seeded plans encode what we actually measured before building the harness:
A needs inference-time recentring (weights 0.3-0.5), B's baseline is dominated
by a target-level bias (shrink toward the checkpoint train-target mean), C only
takes a small, short continuation gain. All seeded plans fit the official
``train`` split only: validation-derived splits are never used as training data.
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


def _damage(a, w):
    return {"type": "damage_shrink", "a": a, "w": w, "fit": "train"}


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
        _plan("extended_pretrain", {"external_epochs": 4, "external_learning_rate": 0.002,
                                    "epochs": 10, "learning_rate": 0.0005, "data": "extended",
                                    "class_balance": True}, {"type": "recenter", "mode": "batch_mean", "weight": 0.4},
              "The official batch-1-only representation is weak on later drifted batches; "
              "opt-in external pretraining on UCI batches 3-7 then fine-tuning on the official "
              "train split should learn drift-robust sensor features without changing the structure.",
              "val macro_f1 +0.02..0.04, drift probe +0.07 (requires RSI_ALLOW_EXTENDED_DATA=1)",
              "external data is not part of the official package; opt-in and disclosed"),
        _plan("retain", {}, _recenter(0.30),
              "Lower blend weight to verify the calibration curve and avoid over-correction.",
              "val macro_f1 +0.004..0.010, probe +0.08", "possible neutral result"),
        _plan("continue_train", {"learning_rate": 0.0001, "epochs": 10, "data": "train",
                                 "class_balance": True, "augment": True}, _recenter(0.45),
              "Gain/offset augmentation on the official train split teaches gain-drift invariance without fitting validation.",
              "val macro_f1 +0.004, probe +0.05", "augmentation may distort in-distribution accuracy"),
        _plan("ridge_last_layer", {"alpha": 10.0, "data": "train"}, _recenter(0.35),
              "Refit the output head on official train only, then calibrate at inference; tests head-vs-representation bottleneck.",
              "val macro_f1 +0.003", "frozen hidden features may not separate drifted classes"),
        _plan("retain", {}, _recenter(0.20, mode="batch_std"),
              "Test whether standard-deviation recentring helps the strongly shifted batches where mean-only recentring saturates.",
              "probe +0.03, val neutral", "std recentring hurt the official batch in our probe"),
        _plan("retain", {}, _recenter(0.45),
              "Final consolidation: keep the best checkpoint and the calibrated blend, no further weight updates.",
              "val macro_f1 stable", "none (safe action)"),
    ],
    "B": [
        _plan("retain", {}, _shrink(0.0),
              "The baseline is anti-correlated with the held-out bearing targets; predicting the train-target centre removes the bias with the lowest variance (no validation labels used).",
              "val rmse 681 -> ~536 (train-mean constant; compliant)", "constant prediction ignores signal and may look unsophisticated"),
        _plan("retain", {}, _damage(0.5, 0.6),
              "The raw model carries a causal degradation signal: a per-segment damage score mapped through a train-fitted curve, blended with the train-centre shrink, corrects the bias conditionally instead of globally.",
              "val rmse ~410-425 without touching the checkpoint", "needs bearing-ordered stream; falls back to plain shrink if the stream is unstable"),
        _plan("retain", {}, _damage(0.5, 0.5),
              "Lower curve weight tests whether the train-fitted curve or the shrunk level carries most of the gain.",
              "val rmse ~430-450", "under-uses the damage curve"),
        _plan("retain", {}, _damage(0.5, 0.7),
              "Higher curve weight explores the end of the calibration range.",
              "val rmse ~415-440", "over-reliance on the curve may hurt bearings unlike the train set"),
        _plan("retain", {}, _shrink(0.10),
              "A small signal component on top of the shrunk train centre keeps ordering while correcting the scale bias.",
              "val rmse ~540", "bias reappears if alpha is too high"),
        _plan("retain", {}, _shrink(0.30),
              "Probe the shrink curve to document where signal content starts to hurt.",
              "val rmse ~550-620", "still far from the constant baseline"),
        _plan("continue_train", {"learning_rate": 0.0003, "epochs": 15, "data": "train",
                                 "class_balance": False, "augment": False}, _shrink(0.0),
              "Retraining may fix the output scale, after which the train-mean centre can be relaxed.",
              "val rmse 630-660 -> shrink to ~536", "continuation did not beat the constant in our offline probe"),
        _plan("ridge_last_layer", {"alpha": 1000.0, "data": "train"}, _shrink(0.0),
              "Closed-form head on official train tests whether only the output layer is misscaled.",
              "val rmse ~605 -> shrink ~536", "frozen features likely insufficient"),
        _plan("swa", {"learning_rate": 0.0002, "epochs": 5, "snapshots": 3,
                      "class_balance": False, "augment": False}, _shrink(0.10),
              "Averaging short continuations reduces variance; shrink keeps the bias correction.",
              "val rmse ~540", "averaged weights remain under-fit"),
        _plan("retain", {}, _shrink(0.05),
              "Fine-grained shrink sweep to pick the least aggressive bias correction that passes the gate.",
              "val rmse ~536-545", "marginal difference"),
        _plan("continue_train", {"learning_rate": 0.0005, "epochs": 10, "data": "train",
                                 "class_balance": False, "augment": False}, _shrink(0.10),
              "Short official-train continuation changes the output scale; shrink then corrects the residual bias.",
              "val rmse ~600 -> 536-560", "continuation may not change the scale on this data"),
        _plan("ridge_last_layer", {"alpha": 10.0, "data": "train"}, _shrink(0.10),
              "Ridge head fitted on official train only; diagnostics for head-vs-representation bottleneck.",
              "val rmse ~600 -> ~540", "frozen features may be uninformative"),
        _plan("retain", {}, _damage(0.5, 0.6),
              "Final consolidation: keep the official parent checkpoint unchanged and deploy the train-fitted damage-aware calibration (postprocess only, no weight updates).",
              "val rmse ~410-425; no checkpoint change", "stream-order assumption disclosed in the round record"),
    ],
    "C": [
        _plan("continue_train_staged",
              {"stage1_learning_rate": 0.001, "stage1_epochs": 1000,
               "stage2_learning_rate": 0.0002, "stage2_epochs": 400,
               "class_balance": False, "augment": False}, {"type": "none"},
              "Long two-stage continuation: the first stage escapes the flat baseline region, the low-lr polish removes the last optimisation noise.",
              "val rmse 52.87 -> ~40.5", "long CPU time; only official train is fitted"),
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


def _stat_key(strategy: str, postprocess: dict | None) -> tuple[str, str]:
    kind = (postprocess or {}).get("type", "none")
    return strategy, "recenter" if kind == "recenter_blend" else kind


def rank_proposals(plans: list[ExperimentProposal], stats: list[dict] | None,
                   epsilon: float = 0.25) -> list[ExperimentProposal]:
    """Order plans by historical win rate, with a deterministic epsilon exploration.

    ``stats`` comes from ``StateDB.strategy_stats(task)``.  Plans whose strategy /
    postprocess pair has a history are ordered by win rate; a plan with no history
    is left in front only when it was the first (seeded) plan, which preserves
    deterministic epsilon exploration without ever dropping the seeded action.
    """
    if not stats:
        return list(plans)
    by_key: dict[tuple[str, str], float] = {}
    for row in stats:
        try:
            by_key[(row.get("strategy"), row.get("postprocess_type", "none"))] = float(
                row.get("win_rate") or 0.0)
        except (TypeError, ValueError):
            continue
    if not by_key:
        return list(plans)

    def score(index_plan):
        index, plan = index_plan
        wins = by_key.get(_stat_key(plan.strategy, plan.postprocess), 0.0)
        return (-wins, index)

    ordered = [plan for _, plan in sorted(enumerate(plans), key=score)]
    # Keep a stable, deterministic exploration slot: the caller passes plans of
    # length >= 1; we never drop the round's seeded plan.
    if epsilon > 0 and len(ordered) > 1 and not by_key.get(
            _stat_key(plans[0].strategy, plans[0].postprocess)):
        ordered = [plans[0]] + [p for p in ordered if p is not plans[0]]
    return ordered


def _generic_bank(task_key: str) -> list[ExperimentProposal]:
    """Fallback policy for an onboarded task with no measured history yet."""
    from rsi.tasks import TASKS
    config = TASKS.get(task_key)
    regression = config is None or config.task == "regression"
    bank = [
        _plan("retain", {}, _recenter(0.4),
              "Inference-time recentring should absorb a mean/scale shift caused by "
              "sensor/environment drift without touching the checkpoint.",
              "Improve macro-F1/RMSE on the public selection split if drift is covariate.",
              "May hurt if the split is in-distribution or labels shifted."),
        _plan("continue_train",
              {"learning_rate": 5e-4, "epochs": 20, "data": "train"},
              {"type": "none"},
              "A short, small-learning-rate continuation on the official train split "
              "may adapt the head to the current drift regime.",
              "Small primary improvement; bounded by the tiny model capacity.",
              "Overfits train if run too long; validation data must never be fitted."),
    ]
    if regression:
        bank.append(_plan("retain", {}, {"type": "shrink", "alpha": 0.2},
                          "Shrink predictions toward the frozen train-target mean to "
                          "reduce variance when the baseline is under-confident.",
                          "Large RMSE drop if the parent's target level is biased.",
                          "Underfits genuinely informative predictions."))
    bank.append(_plan("retain", {}, {"type": "none"},
                      "Consolidate the current parent unchanged.",
                      "No regression; keeps the best model.",
                      "Misses any available gain."))
    return bank


def _plan_bank(task_key: str):
    return list(SEEDED.get(task_key) or _generic_bank(task_key))


def default_plans(task_key: str, round_index: int,
                  stats: list[dict] | None = None) -> list[ExperimentProposal]:
    """Deterministic plan list for one round (1-based round index)."""
    plans = _plan_bank(task_key)
    index = max(0, min(len(plans) - 1, round_index - 1))
    return rank_proposals([plans[index]], stats)


def fallback_proposals(task_key: str, round_index: int,
                       stats: list[dict] | None = None) -> list[ExperimentProposal]:
    """Two safe proposals: the seeded plan plus the final consolidation plan."""
    bank = _plan_bank(task_key)
    primary = bank[max(0, min(len(bank) - 1, round_index - 1))]
    safe = bank[-1]
    proposals = [primary]
    if primary.strategy != safe.strategy or primary.postprocess != safe.postprocess:
        proposals.append(safe)
    return rank_proposals(proposals, stats)
