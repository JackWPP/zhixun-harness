"""Auditable strategy library for the controlled self-improvement loop.

Every strategy is a pure function of (parent checkpoint, public data, params).
The agent may only choose from this registry and within the declared parameter
ranges; the executor enforces the ranges again. This is the "controlled" half of
the recursive self-improvement loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rsi.model import ExtendedModel


@dataclass
class StrategySpec:
    name: str
    description: str
    tasks: tuple[str, ...]
    params: dict
    cost: str
    requires_labels: bool = True

    def to_dict(self):
        return {"name": self.name, "description": self.description, "tasks": list(self.tasks),
                "params": self.params, "cost": self.cost, "requires_labels": self.requires_labels}


@dataclass
class ExperimentContext:
    task_key: str
    task: str
    parent: ExtendedModel
    train: object
    calibration: object
    selection: object
    report: object
    drift: object | None = None
    drift_probe: object | None = None
    extended: object | None = None
    seed: int = 42
    classes: tuple = ()


@dataclass
class Candidate:
    model: ExtendedModel
    postprocess: dict | None = None
    notes: dict = field(default_factory=dict)


def _training_matrix(context: ExperimentContext, data: str, class_balance: bool, augment: bool):
    if data != "train":
        raise ValueError(
            f"training data policy violation: {data!r}; only the official 'train' split "
            "may be fitted (validation/calibration/labels must stay held out)")
    parts = [context.train]
    x = np.vstack([part.x for part in parts])
    y = np.concatenate([part.y for part in parts])
    if class_balance and context.task == "classification":
        classes = [int(c) for c in np.unique(y)]
        weights = [max(1, int(np.sum(y == c))) for c in classes]
        target = max(weights)
        copies = []
        for label, weight in zip(classes, weights):
            repeat = max(1, int(round(target / weight)))
            copies.append(np.where(y == label)[0].tolist() * repeat)
        index = np.concatenate([np.asarray(c, dtype=int) for c in copies])
        rng = np.random.default_rng(context.seed + 17)
        rng.shuffle(index)
        x, y = x[index], y[index]
    if augment:
        x = _augment_features(context, x)
    return x, y


def _augment_features(context: ExperimentContext, x):
    """Mild domain augmentation for sensor batches (covariate-shift robustness).

    Shifts/scales the whole synthetic batch with random factors sampled once per
    batch; records from different UCI batches share sensor gain/offset errors, so
    per-batch augmentation is the honest model of the failure mode.
    """
    rng = np.random.default_rng(context.seed + 29)
    target = context.selection if context.selection is not None else context.calibration
    if context.drift is not None and getattr(context.drift, "evidence", {}).get("mean_shift_sd"):
        magnitude = float(abs(context.drift.evidence["mean_shift_sd"]))
    else:
        magnitude = float(np.mean(np.std(context.train.x, axis=0)))
    rows = x.shape[0]
    copies = []
    for _ in range(2):
        gain = 1.0 + 0.12 * rng.standard_normal(x.shape[1])
        offset = rng.standard_normal(x.shape[1]) * 0.25 * np.maximum(1e-6, magnitude / max(1e-6, np.std(context.train.x, axis=0)))
        copies.append(x * gain + offset)
    index = rng.permutation(rows)
    return np.vstack([x] + copies + [x[index]])


def _class_weight_vector(y, n_classes, beta):
    """Inverse-frequency class weights raised to ``beta`` (0 = uniform).

    The weights are normalised to mean 1 so the effective gradient scale is
    comparable to the unweighted objective. Only official-train labels are
    counted; this is a loss reweighting, not a data or architecture change.
    """
    beta = float(beta)
    if beta <= 0:
        return None
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=int(n_classes)).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (len(y) / (int(n_classes) * counts)) ** beta
    return weights / float(np.mean(weights))


def strategy_retain(context, params):
    return Candidate(context.parent.clone(), postprocess={"type": "none"},
                     notes={"why": "parent retained unchanged"})


def strategy_continue_train(context, params):
    model = context.parent.clone()
    model.set_learning_rate(float(params["learning_rate"]))
    x, y = _training_matrix(context, params["data"], bool(params.get("class_balance")), bool(params.get("augment")))
    beta = float(params.get("class_weight_beta", 0.0) or 0.0)
    weights = None
    if context.task == "classification" and beta > 0:
        weights = _class_weight_vector(y, len(context.classes or []), beta)
    model.fit(x, y, int(params["epochs"]), class_weight=weights)
    return Candidate(model, postprocess={"type": "none"},
                     notes={"rows": int(len(x)), "epochs": int(params["epochs"]),
                            "learning_rate": float(params["learning_rate"]),
                            "class_weight_beta": beta})


def strategy_extended_pretrain(context, params):
    if context.extended is None:
        raise ValueError("extended_pretrain requested but no extended dataset was mounted")
    model = context.parent.clone()
    external_epochs = int(params.get("external_epochs", 4))
    if external_epochs:
        model.set_learning_rate(float(params.get("external_learning_rate", 0.002)))
        model.fit(context.extended.x, context.extended.y, external_epochs)
    model.set_learning_rate(float(params.get("learning_rate", 5e-4)))
    x, y = _training_matrix(context, "train", bool(params.get("class_balance")), False)
    model.fit(x, y, int(params.get("epochs", 10)))
    return Candidate(model, postprocess={"type": "none"},
                     notes={"external_rows": int(len(context.extended)),
                            "external_epochs": external_epochs,
                            "finetune_rows": int(len(x)),
                            "finetune_epochs": int(params.get("epochs", 10)),
                            "source": "external_open_dataset_opt_in"})


def strategy_continue_train_staged(context, params):
    """Two-stage continuation: long high-lr stage then a low-lr polish stage.

    Both stages only see the official train split. Gradient clipping and the
    frozen normalisation come from the official trainer, so the candidate is a
    genuine continuation of the parent checkpoint (same architecture/shapes).
    """
    model = context.parent.clone()
    x, y = _training_matrix(context, "train", bool(params.get("class_balance")),
                            bool(params.get("augment")))
    stages = []
    for suffix in ("1", "2"):
        epochs = int(params[f"stage{suffix}_epochs"])
        if epochs <= 0:
            continue
        model.set_learning_rate(float(params[f"stage{suffix}_learning_rate"]))
        model.fit(x, y, epochs)
        stages.append({"learning_rate": float(params[f"stage{suffix}_learning_rate"]),
                       "epochs": epochs})
    model.validate()
    return Candidate(model, postprocess={"type": "none"},
                     notes={"stages": stages, "rows": int(len(x))})


def strategy_ridge_last_layer(context, params):
    model = context.parent.clone()
    x, y = _training_matrix(context, params["data"], False, False)
    hidden = model.hidden(x)
    design = np.hstack([hidden, np.ones((len(hidden), 1))])
    penalty = float(params["alpha"])
    if model.meta.get("task") == "classification":
        width = int(model.params["w2"].shape[1])
        target = np.zeros((len(y), width), dtype=np.float64)
        target[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    else:
        target = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    weights = np.linalg.solve(design.T @ design + penalty * np.eye(design.shape[1]),
                              design.T @ target)
    model.params["w2"] = weights[:-1].reshape(model.params["w2"].shape)
    model.params["b2"] = np.asarray([weights[-1]], dtype=np.float64).reshape(model.params["b2"].shape)
    model.reset_optimizer()
    model.meta["profile"] = model.meta.get("profile")
    model.validate()
    return Candidate(model, postprocess={"type": "none"},
                     notes={"alpha": penalty, "rows": int(len(x))})


def strategy_swa(context, params):
    model = context.parent.clone()
    model.set_learning_rate(float(params["learning_rate"]))
    x, y = _training_matrix(context, "train", bool(params.get("class_balance")), bool(params.get("augment")))
    snapshots = []
    epochs = int(params["epochs"])
    count = int(params["snapshots"])
    for _ in range(count):
        model.fit(x, y, epochs)
        snapshots.append(model.clone())
    averaged = context.parent.clone()
    averaged.params = {name: np.mean([snap.params[name] for snap in snapshots], axis=0)
                       for name in averaged.params}
    averaged.reset_optimizer()
    averaged.validate()
    return Candidate(averaged, postprocess={"type": "none"},
                     notes={"snapshots": count, "epochs_each": epochs,
                            "learning_rate": float(params["learning_rate"])})


STRATEGIES = {
    "retain": StrategySpec(
        name="retain",
        description="Keep the parent checkpoint unchanged. The safe action after a failed or non-significant experiment.",
        tasks=("A", "B", "C"), params={}, cost="none"),
    "continue_train": StrategySpec(
        name="continue_train",
        description="Continue Adam training from the parent checkpoint on the official train split only (validation must not be fitted).",
        tasks=("A", "B", "C"),
        params={
            "learning_rate": {"type": "float", "min": 1e-5, "max": 0.05, "default": 0.001},
            "epochs": {"type": "int", "min": 1, "max": 300, "default": 20},
            "data": {"type": "choice", "choices": ["train"], "default": "train"},
            "class_balance": {"type": "bool", "default": False},
            "class_weight_beta": {"type": "float", "min": 0.0, "max": 2.0, "default": 0.0},
            "augment": {"type": "bool", "default": False},
        },
        cost="low"),
    "continue_train_staged": StrategySpec(
        name="continue_train_staged",
        description=("Continue training from the parent in two stages: a long first stage at a "
                     "higher learning rate followed by a short low-learning-rate polish. Only the "
                     "official train split is fitted; no architecture or normalisation change."),
        tasks=("A", "B", "C"),
        params={
            "stage1_learning_rate": {"type": "float", "min": 1e-5, "max": 0.05, "default": 0.001},
            "stage1_epochs": {"type": "int", "min": 1, "max": 2000, "default": 400},
            "stage2_learning_rate": {"type": "float", "min": 1e-5, "max": 0.05, "default": 0.0002},
            "stage2_epochs": {"type": "int", "min": 0, "max": 2000, "default": 100},
            "class_balance": {"type": "bool", "default": False},
            "augment": {"type": "bool", "default": False},
        },
        cost="medium"),
    "ridge_last_layer": StrategySpec(
        name="ridge_last_layer",
        description="Closed-form ridge re-fit of the output layer with frozen hidden features (diagnoses whether the head or the representation is the bottleneck).",
        tasks=("A", "B", "C"),
        params={
            "alpha": {"type": "float", "min": 1e-6, "max": 1e6, "default": 10.0},
            "data": {"type": "choice", "choices": ["train"], "default": "train"},
        },
        cost="very-low"),
    "extended_pretrain": StrategySpec(
        name="extended_pretrain",
        description=("Opt-in (RSI_ALLOW_EXTENDED_DATA=1): adapt the parent on the bundled "
                     "external open dataset, then fine-tune on the official train split. "
                     "No structure change; the candidate stays continuation-compatible."),
        tasks=("A",),
        cost="high",
        params={
            "external_epochs": {"type": "int", "min": 1, "max": 20, "default": 4},
            "external_learning_rate": {"type": "float", "min": 0.0005, "max": 0.01, "default": 0.002},
            "epochs": {"type": "int", "min": 1, "max": 60, "default": 10},
            "learning_rate": {"type": "float", "min": 1e-5, "max": 0.005, "default": 5e-4},
            "data": {"type": "choice", "choices": ["extended"], "default": "extended"},
            "class_balance": {"type": "bool", "default": True},
        },
    ),
    "swa": StrategySpec(
        name="swa",
        description="Weight averaging across several short continuation snapshots (variance reduction, no architecture change).",
        tasks=("A", "B", "C"),
        params={
            "learning_rate": {"type": "float", "min": 1e-5, "max": 0.02, "default": 0.0005},
            "epochs": {"type": "int", "min": 1, "max": 100, "default": 10},
            "snapshots": {"type": "int", "min": 2, "max": 6, "default": 3},
            "class_balance": {"type": "bool", "default": False},
            "augment": {"type": "bool", "default": False},
        },
        cost="low"),
}

POSTPROCESS_TYPES = {
    "none": {"description": "No inference-time adaptation."},
    "recenter": {
        "description": "Coverage calibration: re-standardise the incoming feature batch to the checkpoint's frozen training statistics (mean/std or mean-only).",
        "modes": ["batch_std", "batch_mean"],
    },
    "shrink": {
        "description": "Regression output calibration: y = alpha * prediction + (1 - alpha) * center, center = checkpoint train-target mean (no validation labels).",
        "alpha": {"type": "float", "min": 0.0, "max": 1.0},
    },
    "damage_shrink": {
        "description": "RUL degradation calibration: causal damage score -> train-fitted curve, blended with a train-centre shrink. Postprocess only, checkpoint unchanged, fitted on official train labels.",
        "a": {"type": "float", "min": 0.0, "max": 1.0},
        "w": {"type": "float", "min": 0.0, "max": 1.0},
        "fit": {"type": "string", "choices": ["train"]},
    },
}


def run_strategy(context, name, params):
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy: {name}")
    function = {
        "retain": strategy_retain,
        "continue_train": strategy_continue_train,
        "continue_train_staged": strategy_continue_train_staged,
        "ridge_last_layer": strategy_ridge_last_layer,
        "swa": strategy_swa,
        "extended_pretrain": strategy_extended_pretrain,
    }[name]
    return function(context, params)
