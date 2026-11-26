"""Batch-stream protocol tests (honest batch interface)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsi.stream import (
    decision_for,
    make_source,
    materialise_package,
    official_batches,
    predict_rows,
    run_stream,
)
from rsi.stream.protocol import evaluate_rows
from rsi.tasks import resolve_tasks

ROOT = Path(__file__).resolve().parent.parent


def test_official_batches_are_deterministic_and_disjoint():
    config = resolve_tasks(["A"])[0]
    batches = official_batches(config, 4)
    assert len(batches) == 4
    ids = [row["id"] for batch in batches for row in batch.prediction]
    assert len(ids) == len(set(ids)), "batch predictions must not overlap"
    # history grows after each revealed batch; current rows never appear in train
    assert len(batches[1].train) > len(batches[0].train)
    first_ids = {row["id"] for row in batches[0].prediction}
    assert not (first_ids & {row["id"] for row in batches[0].train})
    revealed_pool = {row["id"] for row in batches[1].train} | {row["id"] for row in batches[1].dev}
    assert first_ids <= revealed_pool  # batch 0 labels revealed into train/dev after scoring
    again = official_batches(config, 4)
    assert [r["id"] for r in again[2].prediction] == [r["id"] for r in batches[2].prediction]


def test_private_sim_uses_uci_labels(tmp_path):
    root = ROOT / "extra" / "uci-gas" / "Dataset"
    if not (root / "batch3.dat").is_file():
        pytest.skip("UCI gas data not vendored")
    config = resolve_tasks(["A"])[0]
    batches = make_source("private-sim", config, 2, external_root=str(root),
                          arrival=(3, 4), hidden=(8, 8))
    assert len(batches) == 3
    labels = {value for batch in batches for value in batch.answers.values()}
    assert labels <= set(config.classes)
    for batch in batches:
        assert batch.meta["source"] == "private-sim"
        assert all("answer" not in row for row in batch.prediction)
        assert all(len(row["inputs"]["features"]) == 128 for row in batch.prediction)


def test_make_source_rejects_unknown_kind():
    config = resolve_tasks(["A"])[0]
    with pytest.raises(ValueError):
        make_source("moon", config, 2)


def test_decision_for_both_directions():
    higher = decision_for("classification", "higher", 0.70, 0.75, 0.004)
    assert higher["decision"] == "promote" and higher["delta"] == pytest.approx(0.05)
    lower = decision_for("regression", "lower", 500.0, 450.0, 5.0)
    assert lower["decision"] == "promote"
    retain = decision_for("regression", "lower", 500.0, 498.0, 5.0)
    assert retain["decision"] == "retain"
    rollback = decision_for("classification", "higher", 0.80, 0.70, 0.004)
    assert rollback["decision"] == "rollback"


def test_materialise_and_predict_roundtrip(tmp_path):
    config = resolve_tasks(["A"])[0]
    batch = official_batches(config, 4)[0]
    spec = materialise_package(tmp_path, config, batch, Path(config.checkpoint_path))
    package = Path(spec.package_dir)
    assert (package / "public" / "train.jsonl").is_file()
    assert (package / "public" / "prediction.jsonl").is_file()
    val_path = package / "public" / "validation.jsonl"
    metrics = evaluate_rows("classification", val_path, config.checkpoint_path,
                            {"type": "none"}, config.classes)
    assert 0.0 <= metrics["primary"] <= 1.0
    output = tmp_path / "predictions.jsonl"
    count = predict_rows(package / "public" / "prediction.jsonl", config.checkpoint_path,
                         {"type": "none"}, spec, output)
    rows = [json.loads(line) for line in output.read_text().splitlines() if line]
    assert count == len(rows) == len(batch.prediction)
    assert all("outputs" in row and "id" in row for row in rows)


@pytest.mark.integration
def test_official_stream_frozen_two_batches(tmp_path):
    report = run_stream(
        keys=["A"], source="official", n_batches=2, state_dir=tmp_path,
        controller="pipeline", llm_mode="off", experiments_cap=1,
        batch_wall=600.0, retention_every=0, parallel_tasks=1, arm="frozen",
        hidden_tail=0,  # Exercise two arrival records, not the default hidden tail.
    )
    task = report["tasks"]["A"]
    assert not task["errors"], task["errors"]
    assert len(task["batches"]) == 2
    assert task["hidden"] == []
    for batch in task["batches"]:
        assert batch["phase"] == "arrival"
        batch_dir = tmp_path / "tasks" / "A" / f"batch_{batch['batch']:04d}"
        record = json.loads((batch_dir / "decision.json").read_text())
        evaluation = json.loads((batch_dir / "evaluation.json").read_text())
        assert record["decision"] in {"promote", "retain", "rollback"}
        assert record["arm"] == "frozen"
        # Scored outputs belong to the separate evaluation artifact; the report
        # merges them after the decision has already been written.
        assert "prediction_count" not in record
        assert "prediction_metrics" not in record
        assert evaluation["prediction_count"] == batch["prediction_count"] > 0
    assert Path(task["report_path"]).is_file()
