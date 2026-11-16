"""Honest batch-protocol tests: no current/hidden labels may leak into fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rsi.stream import Batch, materialise_package, predict_rows, score_predictions
from rsi.stream.sources import official_batches, private_sim_batches
from rsi.tasks import resolve_tasks


def _config(key="A"):
    return resolve_tasks([key])[0]


def _row(i, answer="Ethanol", value=0.0):
    return {"id": f"r{i}", "inputs": {"features": [float(value)] * 128}, "answer": answer}


def test_materialise_package_never_contains_current_labels(tmp_path):
    config = _config()
    rows = [_row(i) for i in range(4)]
    batch = Batch(index=0, train=rows[:3], dev=rows[2:3],
                  prediction=[{"id": r["id"], "inputs": r["inputs"]} for r in rows],
                  answers={r["id"]: r["answer"] for r in rows})
    spec = materialise_package(tmp_path, config, batch, config.checkpoint_path)
    for filename, expected in (("train.jsonl", 3), ("validation.jsonl", 1), ("prediction.jsonl", 4)):
        path = Path(spec.package_dir) / "public" / filename
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == expected, filename
        if filename == "prediction.jsonl":
            assert all("answer" not in line for line in lines), filename
        else:
            assert all("answer" in line for line in lines), filename


def test_answers_are_not_used_by_predictions(tmp_path):
    config = _config()
    base = [_row(i, "Ethanol" if i % 2 else "Acetaldehyde", value=0.01 * i) for i in range(8)]
    prediction = [{"id": r["id"], "inputs": r["inputs"]} for r in base]
    answers_a = {r["id"]: "Ethanol" for r in base}
    answers_b = {r["id"]: ("Acetaldehyde" if r["id"] == "r0" else r["answer"]) for r in base}
    outs = []
    for name, answers in (("a", answers_a), ("b", answers_b)):
        batch = Batch(index=0, train=base[:4], dev=base[4:6], prediction=prediction, answers=answers)
        spec = materialise_package(tmp_path / name, config, batch, config.checkpoint_path)
        out = tmp_path / name / "preds.jsonl"
        assert predict_rows(Path(spec.package_dir) / "public" / "prediction.jsonl",
                            spec.checkpoint_path, None, spec, out) == 8
        outs.append(out.read_text())
    assert outs[0] == outs[1]


def test_official_hidden_tail_is_not_revealed():
    config = _config("A")
    batches = official_batches(config, 5, hidden_tail=2)
    assert [b.meta["hidden"] for b in batches] == [False, False, False, True, True]
    assert batches[3].reveal == [] and batches[4].reveal == []
    assert all(b.reveal for b in batches[:3])


def test_label_schedule_and_fraction():
    config = _config("A")
    batches = private_sim_batches(config, "", arrival=(3, 4), hidden=(8, 8),
                                  label_every=2, label_fraction=0.5)
    arrival = [b for b in batches if not b.meta["hidden"]]
    assert arrival, "expected arrival batches from UCI batch3/4"
    assert arrival[0].reveal == []          # labelled only every 2nd batch
    assert arrival[1].reveal                # second arrival reveals a stratified half
    hidden = [b for b in batches if b.meta["hidden"]]
    assert len(hidden) == 1 and hidden[0].reveal == []


def test_score_predictions_matches_official_shape(tmp_path):
    config = _config()
    rows = [_row(i, "Ethanol" if i % 2 else "Acetaldehyde", value=0.01 * i) for i in range(8)]
    prediction = [{"id": r["id"], "inputs": r["inputs"]} for r in rows]
    batch = Batch(index=0, train=rows[:4], dev=rows[4:6], prediction=prediction,
                  answers={r["id"]: r["answer"] for r in rows})
    spec = materialise_package(tmp_path, config, batch, config.checkpoint_path)
    out = tmp_path / "preds.jsonl"
    predict_rows(Path(spec.package_dir) / "public" / "prediction.jsonl",
                 spec.checkpoint_path, None, spec, out)
    metrics = score_predictions(out, batch.answers, "classification", config.classes)
    assert metrics["rows"] == 8 and metrics["primary"] is not None
    assert 0.0 <= metrics["primary"] <= 1.0


@pytest.mark.integration
def test_frozen_stream_roundtrip(tmp_path):
    from rsi.stream.runner import run_task_stream

    result = run_task_stream({
        "key": "A", "source": "official", "n_batches": 2, "arm": "frozen",
        "state_dir": str(tmp_path), "llm_mode": "off", "experiments_cap": 1,
        "memory": False, "retention_every": 0, "verbose": False,
    })
    assert result["errors"] == []
    assert len(result["batches"]) == 2
    assert all(item["decision"] == "retain" for item in result["batches"])
    assert all(item["prediction_count"] > 0 for item in result["batches"])
    assert result["report_path"] and Path(result["report_path"]).is_file()


def test_auto_arm_routes_by_task_and_source():
    from rsi.stream.runner import AUTO_ARMS, resolve_arm

    assert resolve_arm("auto", "A", "official") == "pair"
    assert resolve_arm("auto", "B", "official") == "adapt"
    assert resolve_arm("auto", "C", "official") == "frozen"
    assert resolve_arm("auto", "A", "private-sim") == "full"
    assert resolve_arm("auto", "B", "private-sim") == "adapt"
    assert resolve_arm("auto", "C", "private-sim") == "frozen"
    assert resolve_arm("auto", "A", "external-a") == AUTO_ARMS["private-sim"]["A"]
    assert resolve_arm("auto", "A", "unknown-source") == "frozen"
    # explicit arms are never overridden by the router
    assert resolve_arm("recenter", "A", "official") == "recenter"
