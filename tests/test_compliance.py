"""Competition-compliance invariants.

The competition rules state: "验证集和隐藏集不能参与拟合这些统计量" and the
announcement requires real parent-checkpoint lineage.  These tests are the
deterministic gate that keeps the autonomous agent from fitting validation
splits (which an earlier kernel run did before this fix).
"""
from __future__ import annotations

import numpy as np
import pytest

from rsi.agent.guard import Guard, GuardViolation
from rsi.executor import ExperimentProposal, InvalidExperiment, validate_proposal
from rsi.inference import resolve_postprocess
from rsi.model import ExtendedModel
from rsi.planner import SEEDED, default_plans, fallback_proposals
from rsi.tasks import TASKS


@pytest.mark.parametrize("task_key", ["A", "B", "C"])
def test_guard_rejects_validation_derived_training_data(task_key):
    task = TASKS[task_key]
    guard = Guard(task)
    for data in ("train+calibration", "train+all-public", "validation", "hidden"):
        proposal = ExperimentProposal(
            strategy="continue_train",
            params={"learning_rate": 1e-3, "epochs": 5, "data": data},
        )
        with pytest.raises(GuardViolation):
            guard.validate_catalog(proposal)
    # official train is allowed
    ok = guard.validate_catalog(ExperimentProposal(
        strategy="continue_train",
        params={"learning_rate": 1e-3, "epochs": 5, "data": "train"},
    ))
    assert ok.params["data"] == "train"


def test_strategy_catalog_rejects_validation_data():
    task = TASKS["A"]
    proposal = ExperimentProposal(strategy="continue_train",
                                  params={"learning_rate": 1e-3, "epochs": 5,
                                          "data": "train+calibration"})
    with pytest.raises(InvalidExperiment):
        validate_proposal(proposal, task)


def test_all_seeded_plans_fit_train_only():
    for task_key in ("A", "B", "C"):
        plans = list(SEEDED[task_key])
        plans += default_plans(task_key, 0)
        plans += fallback_proposals(task_key, 0)
        for plan in plans:
            data = (plan.params or {}).get("data")
            # "extended" is opt-in external pretraining; it is gated by policy
            # tests and never used for the scored train-only default.
            assert data in (None, "train", "extended"), (task_key, plan.strategy, data)


def test_extended_pretraining_is_opt_in_only(monkeypatch):
    from rsi.agent.guard import Guard, GuardViolation
    from rsi.tasks import load_task_spec
    task = load_task_spec("tasks/A/task.json")
    proposal = ExperimentProposal(
        strategy="extended_pretrain",
        params={"external_epochs": 2, "external_learning_rate": 0.002,
                "epochs": 5, "learning_rate": 5e-4, "data": "extended"},
    )
    monkeypatch.delenv("RSI_ALLOW_EXTENDED_DATA", raising=False)
    guard = Guard(task)
    with pytest.raises(GuardViolation):
        guard.validate_catalog(proposal)
    monkeypatch.setenv("RSI_ALLOW_EXTENDED_DATA", "1")
    guard.validate_catalog(proposal)  # explicitly enabled -> allowed


def test_shrink_center_is_checkpoint_train_mean():
    model = ExtendedModel.from_checkpoint(TASKS["B"].checkpoint_path)
    expected = float(model.meta["target_mean"])
    assert abs(expected - 954.151) < 0.01  # official B train target mean
    # even if a calibration object with very different labels is supplied, the
    # calibration centre must not move to validation statistics.
    class Calibration:
        y = np.full(100, 694.3)
    spec = resolve_postprocess({"type": "shrink", "alpha": 0.0}, model, Calibration())
    assert spec["center_source"] == "train_target_mean"
    assert spec["calibration_rows"] == 0
    assert abs(spec["center"] - expected) < 1e-5


def test_b_train_mean_constant_is_the_compliant_reference():
    """The documented compliant B number: train-mean constant on official val."""
    from rsi.data import load_jsonl_dataset
    model = ExtendedModel.from_checkpoint(TASKS["B"].checkpoint_path)
    val = load_jsonl_dataset(TASKS["B"].validation_path, TASKS["B"].task,
                             expected_paths=model.meta["paths"])
    center = float(model.meta["target_mean"])
    rmse = float(np.sqrt(np.mean((val.y - center) ** 2)))
    assert 530.0 < rmse < 545.0, rmse  # ~536, beats official baseline 681 by ~21%
