"""Batch-stream protocol tests (track S official + track X external)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsi.stream import (
    decision_for,
    external_a_batches,
    make_source,
    materialise_package,
    official_batches,
    run_stream,
)
from rsi.stream.protocol import evaluate_rows, write_predictions
from rsi.tasks import resolve_tasks

ROOT = Path(__file__).resolve().parent.parent


def test_official_batches_are_deterministic_and_disjoint():
    config = resolve_tasks(["A"])[0]
    batches = official_batches(config, 4)
    assert len(batches) == 4
    ids = [row["id"] for batch in batches for row in batch.validation]
    assert len(ids) == len(set(ids)), "batch validations must not overlap"
    assert sum(len(b.validation) for b in batches) == len(ids)
    # cumulative labels: batch 1 may train on batch 0 rows, never on batch 1 eval rows
    assert len(batches[1].train) > len(batches[0].train)
    first_ids = {row["id"] for row in batches[0].validation}
    assert not (first_ids & {row["id"] for row in batches[0].train})
    # deterministic re-run
    again = official_batches(config, 4)
    assert [r["id"] for r in again[2].validation] == [r["id"] for r in batches[2].validation]


def test_external_a_batches_use_uci_labels(tmp_path):
    root = ROOT / "extra" / "uci-gas" / "Dataset"
    if not (root / "batch3.dat").is_file():
        pytest.skip("UCI gas data not vendored")
    config = resolve_tasks(["A"])[0]
    batches = external_a_batches(config, root, n_batches=2, start=3, end=4, holdout=0.3)
    assert len(batches) == 2
    labels = {row["answer"] for batch in batches for row in batch.validation}
    assert labels <= set(config.classes)
    for batch in batches:
        assert batch.meta["source"] == "external-a"
        assert batch.train and batch.validation
        assert all(len(row["inputs"]["features"]) == 128 for row in batch.validation)


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
    package = Path(spec.package)
    assert (package / "public" / "train.jsonl").is_file()
    assert (package / "public" / "validation.jsonl").is_file()
    assert (package / "checkpoint.npz").is_file()
    assert spec.package_dir == package.resolve()
    val_path = package / "public" / "validation.jsonl"
    metrics = evaluate_rows("classification", val_path, config.checkpoint_path, {"type": "none"}, config.classes)
    assert 0.0 <= metrics["primary"] <= 1.0
    output = tmp_path / "predictions.jsonl"
    count = write_predictions(val_path, config.checkpoint_path, {"type": "none"}, config, output)
    rows = [json.loads(line) for line in output.read_text().splitlines() if line]
    assert count == len(rows) and count == len(batch.validation)
    assert all("outputs" in row and "id" in row for row in rows)


@pytest.mark.integration
def test_official_stream_two_batches_end_to_end(tmp_path):
    report = run_stream(
        keys=["A"], source="official", n_batches=2, state_dir=tmp_path,
        controller="pipeline", llm_mode="off", experiments_cap=2,
        batch_wall=600.0, retention_every=0, parallel_tasks=1,
    )
    task = report["tasks"]["A"]
    assert len(task["batches"]) == 2, task.get("errors")
    for batch in task["batches"]:
        decision_path = tmp_path / "tasks" / "A" / f"batch_{batch['batch']:04d}" / "decision.json"
        assert decision_path.is_file()
        record = json.loads(decision_path.read_text())
        assert record["decision"] in {"promote", "retain", "rollback"}
        assert record["decision_detail"]["decision"] == record["decision"]
        assert record["round_index"] == batch["batch"]
        assert record["checkpoint_sha256"].startswith("sha256:")
        predictions = Path(record["predictions"]["path"])
        assert predictions.is_file()
        lines = [json.loads(line) for line in predictions.read_text().splitlines() if line]
        assert len(lines) == record["predictions"]["count"]
        assert all("outputs" in row for row in lines)
    assert Path(task["report_path"]).is_file()
