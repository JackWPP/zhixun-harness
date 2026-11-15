"""Tests for the private / synthetic data adapters (P4)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from rsi.adapters import SchemaError, describe_dataset, evaluate_private, synthesize, write_synthetic
from rsi.adapters.schema import load_private_dataset
from rsi.tasks import resolve_tasks


@pytest.fixture(scope="module")
def config_a():
    return resolve_tasks(["A"])[0]


def _rows(n=6, dim=128):
    labels = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
    rng = np.random.default_rng(0)
    return [
        {"id": f"p{index}", "inputs": {"features": rng.normal(size=dim).tolist()},
         "answer": labels[index % len(labels)]}
        for index in range(n)
    ]


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_schema_summary(config_a, tmp_path):
    path = _write(tmp_path / "probe.jsonl", _rows())
    dataset = load_private_dataset(config_a, path)
    summary = describe_dataset(dataset, config_a)
    assert summary["rows"] == 6
    assert summary["features"] == 128
    assert summary["used_for_training"] is False


def test_schema_rejects_bad_rows(config_a, tmp_path):
    bad_answer = tmp_path / "bad_answer.jsonl"
    bad_answer.write_text(json.dumps({"id": "x", "inputs": {"features": [0.0] * 128},
                                      "answer": "NotAClass"}) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_private_dataset(config_a, bad_answer)

    leaked = _rows(1)
    leaked[0]["inputs"]["rul"] = 12.0
    with pytest.raises(SchemaError):
        load_private_dataset(config_a, _write(tmp_path / "leaked.jsonl", leaked))

    with pytest.raises(SchemaError):
        load_private_dataset(config_a, tmp_path / "missing.jsonl")


def test_private_evaluation_and_isolation(config_a, tmp_path):
    path = _write(tmp_path / "probe.jsonl", _rows(12))
    report = evaluate_private(config_a, path, config_a.checkpoint_path)
    assert 0.0 <= report["primary"] <= 1.0
    assert report["metrics"]["macro_f1"] == pytest.approx(report["primary"])
    assert report["training_isolation"]
    assert report["per_group"]


def test_synthetic_deterministic_and_grouped(config_a, tmp_path):
    source = _write(tmp_path / "source.jsonl", _rows(5))
    first = synthesize(config_a, source, batches=3, seed=11)
    second = synthesize(config_a, source, batches=3, seed=11)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert first[0]["inputs"] == second[0]["inputs"]
    assert len(first) == 15
    assert all(row["meta"]["synthetic"] for row in first)
    assert all(row["answer"] == row["answer"] for row in first)

    other = synthesize(config_a, source, batches=3, seed=99)
    assert other[0]["inputs"] != first[0]["inputs"]

    out = write_synthetic(first, tmp_path / "synth.jsonl")
    dataset = load_private_dataset(config_a, out)
    assert sorted(set(dataset.groups)) == ["batch_1", "batch_2", "batch_3"]

    report = evaluate_private(config_a, out, config_a.checkpoint_path)
    assert len(report["per_group"]) == 3


def test_synthetic_ids_do_not_contain_labels_for_leakage(config_a, tmp_path):
    source = _write(tmp_path / "source.jsonl", _rows(2))
    rows = synthesize(config_a, source, batches=1, seed=1)
    assert rows[0]["id"].startswith("synbatch1_")
    assert all(row["id"] != row["answer"] for row in rows)
