"""Regression tests for the full-scale batch-stream rehearsal fixes (SIM-001..004)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rsi.agent.config import TokenBudget
from rsi.data import flatten
from rsi.model import ExtendedModel
from rsi.stream.sources import full_batches
from rsi.tasks import resolve_tasks

REPO = Path(__file__).resolve().parents[1]


def test_token_budget_zero_is_unlimited():
    budget = TokenBudget(cap=0)
    assert budget.exhausted() is False
    assert budget.remaining() > 10**12
    budget.add({"input": 2_000_000, "output": 10})
    assert budget.exhausted() is False

    finite = TokenBudget(cap=10)
    finite.add({"input": 11})
    assert finite.exhausted() is True


def test_full_batches_do_not_leak_current_batch_labels():
    config = resolve_tasks(["A"])[0]
    batches = full_batches(config, n_batches=10, dev_fraction=0.25, seed=42)
    assert len(batches) == 10
    for batch in batches:
        visible = {str(row["id"]) for row in batch.train} | {str(row["id"]) for row in batch.dev}
        current = {str(row["id"]) for row in batch.prediction}
        assert current.isdisjoint(visible), f"batch {batch.index} leaks its own labels"
        assert batch.answers.keys() == current


def test_v2_adapter_has_pair_history_candidate_for_task_a():
    from rsi.platform.plans import PAIR_HISTORY_DELTA
    from rsi.platform.submission import _candidate_plans

    config = resolve_tasks(["A"])[0]
    model = ExtendedModel.from_checkpoint(config.checkpoint_path)
    rows = [json.loads(line) for line in
            (config.validation_path).read_text(encoding="utf-8").splitlines()[:120]]
    train = rows[:80]
    validation = rows[80:]
    plans = _candidate_plans("A", model, train, validation)
    kinds = {plan["kind"] for plan in plans}
    # Compliance: the pair delta is a fixed historical prior, never fitted from
    # labels inside the training run.
    assert "pair_history" in kinds
    assert "pair_threshold" not in kinds
    arm = next(plan for plan in plans if plan["kind"] == "pair_history")
    assert arm["postprocess"]["type"] == "pair_threshold"
    assert float(arm["postprocess"]["delta"]) == float(PAIR_HISTORY_DELTA)
    assert {"retain", "recenter"} <= kinds


def test_code_sha256_tolerates_vanishing_files(tmp_path):
    from unittest.mock import patch

    from rsi.model import code_sha256

    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.py").write_text("print(1)\n", encoding="utf-8")
    (root / "b.md").write_text("# doc\n", encoding="utf-8")
    real = Path.read_bytes

    def fake(self):
        if self.name == "b.md":
            raise FileNotFoundError(self)
        return real(self)

    with patch.object(Path, "read_bytes", fake):
        digest = code_sha256(root)
    assert digest.startswith("sha256:")


def test_full_batches_repeat_keeps_ids_unique():
    config = resolve_tasks(["A"])[0]
    batches = full_batches(config, n_batches=10, dev_fraction=0.25, seed=42, repeat=2)
    assert len(batches) == 20
    for batch in batches:
        ids = [str(r["id"]) for r in batch.train]
        assert len(ids) == len(set(ids)), "history contains duplicate sample ids after repeat"


def test_official_seq_stage_data_keeps_validation_out_of_training(tmp_path):
    """SDK 1.0.4: the public validation split is measurement-only.

    Regression guard for the official-sequence host: `/data/train` must grow
    with the drops revealed by earlier rounds, never with the validation rows
    that are used for selection.
    """
    import sys

    sys.path.insert(0, str(REPO / "scripts" / "sim"))
    import full_env

    spec = full_env.official_seq_spec(rounds=3, seed=0)
    expected = [len(spec["train_rows"])]
    for drop in spec["drops"][:2]:
        expected.append(expected[-1] + len(drop))
    for round_index in (1, 2, 3):
        full_env.stage_data(spec, round_index, 3, tmp_path / ("r%d" % round_index))
        root = tmp_path / ("r%d" % round_index) / "public"
        train = [json.loads(line) for line in
                 (root / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        validation = [json.loads(line) for line in
                      (root / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
        train_ids = {str(row["id"]) for row in train}
        validation_ids = {str(row["id"]) for row in validation}
        assert len(validation) == len(spec["validation_rows"])
        assert not (train_ids & validation_ids), "validation rows leaked into training"
        assert len(train) == expected[round_index - 1]
