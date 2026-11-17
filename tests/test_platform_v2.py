"""Contract tests for the contest.harness.workspace.v2 participant adapter."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "docker" / "app"
SDK = REPO / "vendor" / "contest_harness-1.0.2"


def run_python(args, **kwargs):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(SDK), str(APP)])
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env,
                          timeout=180, **kwargs)


def first_row(package: str) -> dict:
    path = REPO / "datasets" / package / "public" / "validation.jsonl"
    with path.open(encoding="utf-8") as handle:
        return json.loads(handle.readline())


def test_describe_speaks_workspace_v2():
    result = run_python([str(APP / "harness.py"), "describe"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload == {
        "protocol": "contest.harness.workspace.v2",
        "model_access": "host_gateway_rpc",
        "sdk": "contest_harness",
        "sdk_version": "1.0.2",
        "tasks": ["classification", "regression"],
    }


def test_flatten_uses_checkpoint_path_specs():
    from rsi.model import ExtendedModel
    from rsi.platform.submission import _flatten_row, _load_specs

    for package, width in (("cpu-a-guide-20261110-v1", 128), ("cpu-b-rul-20261110-v1", 34)):
        model = ExtendedModel.from_checkpoint(REPO / "datasets" / package / "checkpoint.npz")
        row = first_row(package)
        values = _flatten_row(_load_specs(model), row["inputs"])
        assert values.shape == (width,)
        if width == 128:
            assert values.tolist() == [float(value) for value in row["inputs"]["features"]]


def test_predict_classification_artifact(tmp_path):
    from rsi.platform.submission import predict

    source = REPO / "datasets" / "cpu-a-guide-20261110-v1"
    shutil.copyfile(source / "checkpoint.npz", tmp_path / "checkpoint.npz")
    (tmp_path / "submission_meta.json").write_text(json.dumps({
        "task": "classification",
        "classes": ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
        "postprocess": {"type": "recenter", "mode": "batch_mean", "weight": 0.4},
    }))
    row = first_row("cpu-a-guide-20261110-v1")
    outputs = []
    for _ in range(3):
        outputs.append(predict(SimpleNamespace(task="classification", inputs=row["inputs"]), tmp_path))
    for item in outputs:
        assert len(item) == 6
        labels = {entry["label"] for entry in item}
        assert len(labels) == len(item)
        scores = [entry["score"] for entry in item]
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert sum(score == max(scores) for score in scores) == 1


def test_predict_regression_shrink(tmp_path):
    from rsi.platform.submission import predict

    source = REPO / "datasets" / "cpu-c-guide-20261110-v1"
    shutil.copyfile(source / "checkpoint.npz", tmp_path / "checkpoint.npz")
    (tmp_path / "submission_meta.json").write_text(json.dumps({
        "task": "regression",
        "postprocess": {"type": "shrink", "alpha": 0.3, "center": 123.0},
    }))
    row = first_row("cpu-c-guide-20261110-v1")
    output = predict(SimpleNamespace(task="regression", inputs=row["inputs"]), tmp_path)
    assert set(output) == {"value"}
    assert output["value"] >= 0


def test_emulator_round_trip_a():
    result = run_python([str(REPO / "scripts" / "emulate_workspace_v2.py"), "--task", "A", "--rounds", "1"])
    assert result.returncode == 0, result.stderr
    assert "workspace v2 emulation PASS" in result.stdout


@pytest.mark.integration
def test_emulator_regression_task_c():
    result = run_python([str(REPO / "scripts" / "emulate_workspace_v2.py"), "--task", "C", "--rounds", "1"])
    assert result.returncode == 0, result.stderr
    assert "workspace v2 emulation PASS" in result.stdout


def test_contract_field_variants_are_tolerated():
    from rsi.platform.submission import _contract_value

    contract = {"task_type": "regression", "metric_name": "rmse", "metric_direction": "lower"}
    assert _contract_value(contract, ("task", "task_type"), "classification") == "regression"
    assert _contract_value(contract, ("metric", "metric_name"), "macro_f1") == "rmse"
    assert _contract_value(contract, ("direction", "metric_direction"), "higher") == "lower"
    assert _contract_value(None, ("task",), "classification") == "classification"


def test_uniform_scores_always_have_single_top():
    import numpy as np
    from rsi.platform.submission import _uniform_scores

    for raw in ([0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]):
        scores = _uniform_scores(np.asarray(raw, dtype=np.float64))
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
        assert int((scores == scores.max()).sum()) == 1, (raw, scores)


def test_change_summary_is_honest_for_retain():
    from rsi.platform.submission import _change_summary

    assert "保留父模型" in _change_summary({"kind": "retain"})
    assert "续训" in _change_summary({"kind": "continue_train"})
    assert "未改 checkpoint" in _change_summary({"kind": "recenter"})
