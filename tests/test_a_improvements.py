"""Regression tests for the A-improvement work (P0 history adaptation + P1 class weighting)."""
from __future__ import annotations

import json
import math

import numpy as np

from rsi.data import load_jsonl_dataset
from rsi.model import ExtendedModel
from rsi.planner import SEEDED
from rsi.stream.protocol import write_jsonl
from rsi.stream.sources import Batch
from rsi.stream.runner import _adapt_on_history
from rsi.strategies import _class_weight_vector
from rsi.tasks import resolve_tasks


def _task():
    return resolve_tasks(["A"])[0]


def test_class_weight_vector_upweights_rare_classes():
    y = np.asarray([0] * 90 + [1] * 30 + [2] * 10 + [3] * 5 + [4] * 60 + [5] * 3)
    weights = _class_weight_vector(y, 6, 1.0)
    assert weights.shape == (6,)
    assert np.all(weights > 0)
    assert abs(float(np.mean(weights)) - 1.0) < 1e-9
    # class 5 appears least often -> largest weight; class 0 most often -> smallest
    assert weights[5] > weights[1] > weights[0]
    assert _class_weight_vector(y, 6, 0.0) is None


def test_weighted_fit_keeps_checkpoint_schema_and_changes_head():
    config = _task()
    parent = ExtendedModel.from_checkpoint(config.checkpoint_path)
    dataset = load_jsonl_dataset(config.train_path, config.task, config.classes)
    subset = dataset.x[:128], dataset.y[:128]
    weights = _class_weight_vector(subset[1], len(config.classes), 0.5)
    model = parent.clone()
    model.set_learning_rate(5e-4)
    loss = model.fit(subset[0], subset[1], 1, class_weight=weights)
    assert math.isfinite(float(loss[-1]))
    assert model.validate() is None
    delta = float(np.max(np.abs(model.params["w2"] - parent.params["w2"])))
    assert delta > 0
    # checkpoint schema is unchanged apart from the weight values
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.npz"
        model.save(path)
        again = ExtendedModel.from_checkpoint(path)
        assert again.meta["task"] == parent.meta["task"]
        assert again.params["w1"].shape == parent.params["w1"].shape


def test_weighted_continuation_recipe_is_available():
    """The recipe stays available in the strategy registry even though the gate rejects it.

    Canonical A run (2026-11-16) tried `class_weight_beta=0.5, lr=5e-4, ep30 +
    recenter w=0.3`: full-validation macro-F1 0.7279 (deployed 0.7199), but the
    selection split scored it 0.7226 against the parent's 0.7346, so the
    controlled gate correctly kept the parent. The recipe is therefore exposed
    as an opt-in capability, not as a default plan.
    """
    from rsi.strategies import STRATEGIES

    spec = STRATEGIES["continue_train"]
    assert "class_weight_beta" in spec.params
    assert spec.params["class_weight_beta"]["min"] == 0.0
    assert spec.params["class_weight_beta"]["max"] == 2.0
    assert SEEDED["A"], "A plan bank must remain non-empty"


def test_history_adaptation_writes_compatible_checkpoint(tmp_path):
    from rsi.stream import Batch, materialise_package

    config = _task()
    dataset = load_jsonl_dataset(config.train_path, config.task, config.classes)
    rows = []
    for i in range(160):
        rows.append({
            "id": dataset.ids[i],
            "inputs": {"features": dataset.x[i].tolist()},
            "answer": config.classes[int(dataset.y[i])],
        })
    batch = Batch(index=1, train=rows[:128], dev=rows[128:160],
                  prediction=[{"id": r["id"], "inputs": r["inputs"]} for r in rows],
                  answers={r["id"]: r["answer"] for r in rows}, meta={"case": "unit"})
    batch_dir = tmp_path / "batch_0001"
    materialise_package(batch_dir, config, batch, config.checkpoint_path)
    adapted = _adapt_on_history(batch_dir, config, config.checkpoint_path, epochs=1, learning_rate=5e-4)
    assert adapted.exists()
    loaded = ExtendedModel.from_checkpoint(adapted)
    assert loaded.params["w1"].shape == (128, 32)
    assert (batch_dir / "package" / "checkpoint.npz").exists()


def test_cli_stream_accepts_adapt_history_flag():
    from rsi.cli import build_parser

    args = build_parser().parse_args(["stream", "--adapt-history", "--adapt-epochs", "5"])
    assert args.adapt_history is True
    assert args.adapt_epochs == 5


def test_continue_train_augment_and_weights_align_targets():
    """PyTRiO-driven run exposed ValueError in augment path (SIM-009)."""
    from pathlib import Path

    from rsi.data import load_jsonl_dataset
    from rsi.model import ExtendedModel
    from rsi.strategies import ExperimentContext, run_strategy
    from rsi.tasks import resolve_tasks

    config = resolve_tasks(["A"])[0]
    base = ExtendedModel.from_checkpoint(config.checkpoint_path)
    train = load_jsonl_dataset(config.train_path, "classification", config.classes,
                               expected_paths=base.meta["paths"])
    ctx = ExperimentContext(task_key="A", task=config.task, parent=base.clone(),
                            train=train, calibration=train, selection=train,
                            report=train, seed=42, classes=tuple(config.classes))
    candidate = run_strategy(ctx, "continue_train", {
        "learning_rate": 5e-4, "epochs": 2, "data": "train",
        "class_balance": True, "class_weight_beta": 1.0, "augment": True,
    })
    assert candidate.model.params["w1"].shape == base.params["w1"].shape
