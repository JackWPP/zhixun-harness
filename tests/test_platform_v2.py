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


def test_predict_fallback_keeps_stream_valid():
    from rsi.platform.submission import _fallback_prediction

    classification = _fallback_prediction({
        "task": "classification", "classes": ["a", "b", "c"],
        "model": SimpleNamespace(meta={}),
    })
    scores = [item["score"] for item in classification]
    assert {item["label"] for item in classification} == {"a", "b", "c"}
    assert sum(score == max(scores) for score in scores) == 1
    assert all(0.0 <= score <= 1.0 for score in scores)
    regression = _fallback_prediction({
        "task": "regression", "classes": [],
        "model": SimpleNamespace(meta={"target_mean": 123.0}),
    })
    assert regression == {"value": 123.0}


def test_gateway_call_retries_transient_errors():
    from rsi.platform.submission import _gateway_model

    class FakeRound:
        def __init__(self):
            self.calls = 0

        def model(self, messages, max_tokens=None, temperature=None):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("temporary gateway error")
            return SimpleNamespace(call_id="call-%d" % self.calls, text="{}")

    rnd = FakeRound()
    receipt = _gateway_model(rnd, [{"role": "user", "content": "x"}], max_tokens=16)
    assert receipt.call_id == "call-3"
    assert rnd.calls == 3


def _load_jsonl(path, limit):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _fake_round(task_contract):
    base = REPO / "datasets" / "cpu-b-rul-20261110-v1"
    train = _load_jsonl(base / "public" / "train.jsonl", 120)
    validation = _load_jsonl(base / "public" / "validation.jsonl", 60)
    checkpoint = base / "checkpoint.npz"

    class FakeParent:
        checkpoint_sha256 = "fake-parent-sha"

        def path(self, relative):
            assert relative == "checkpoint.npz"
            return checkpoint

    class FakeData:
        def training_rows(self):
            return iter(train)

        def validation_rows(self):
            return iter(validation)

    class FakeRound:
        dataset_key = "B"
        round_index = 1
        max_output_tokens = 256

        def __init__(self):
            self.task = task_contract
            self.parent = FakeParent()
            self.data = FakeData()
            self.calls = []
            self.submitted = None
            self.finished = None
            self.saved_state = None

        def model(self, messages, max_tokens=256, temperature=0.0):
            self.calls.append(messages)
            return SimpleNamespace(
                text='{"choice": 0, "hypothesis": "keep parent", "reason": "test"}',
                call_id="call-test-1",
            )

        def history(self):
            return []

        def state_workspace(self, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            return Path(destination)

        def submit_candidate(self, directory, **kwargs):
            self.submitted = {"directory": Path(directory), **kwargs}
            return {"artifact_id": "artifact-test-1"}

        def save_state(self, directory):
            self.saved_state = Path(directory)

        def finish(self, **kwargs):
            self.finished = kwargs

    return FakeRound()


def test_train_follows_checkpoint_task_when_host_contract_disagrees(tmp_path, monkeypatch, capsys):
    """Dataset B regression checkpoint must not hard-fail when the host
    contract wrongly declares classification (platform review 2026-11-18)."""
    from rsi.platform.submission import train

    monkeypatch.setenv("CONTEST_WORK", str(tmp_path / "work"))
    rnd = _fake_round({"task": "classification", "metric": "macro_f1", "direction": "higher"})
    train(rnd)

    assert len(rnd.calls) == 1, "round must still make the required gateway call"
    assert rnd.submitted is not None
    assert rnd.finished is not None
    assert rnd.finished["model_call_ids"] == ["call-test-1"]
    meta_path = rnd.submitted["directory"] / "submission_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["task"] == "regression"
    assert meta["metric"] == "rmse"
    assert "checkpoint task" in capsys.readouterr().err


def test_predict_coerces_task_mismatch_shapes():
    from rsi.platform.submission import _coerce_task_output

    regression_artifact = {"task": "regression", "classes": []}
    classification_out = _coerce_task_output(
        "classification", regression_artifact, {"value": 30.0})
    assert isinstance(classification_out, list) and len(classification_out) == 2
    assert len({item["label"] for item in classification_out}) == 2
    scores = [item["score"] for item in classification_out]
    assert scores.count(max(scores)) == 1
    assert max(scores) == pytest.approx(0.9)
    assert all(0.0 <= score <= 1.0 for score in scores)

    classification_artifact = {"task": "classification", "classes": ["a", "b", "c"]}
    regression_out = _coerce_task_output(
        "regression", classification_artifact,
        [{"label": "a", "score": 0.1}, {"label": "b", "score": 0.8}, {"label": "c", "score": 0.1}])
    assert regression_out == {"value": 1.0}

    same_artifact = {"task": "regression", "classes": []}
    same_output = {"value": 7.0}
    assert _coerce_task_output("regression", same_artifact, same_output) is same_output


def test_predict_on_regression_artifact_with_classification_request(monkeypatch):
    from rsi.platform.submission import predict

    base = REPO / "datasets" / "cpu-b-rul-20261110-v1"
    row = _load_jsonl(base / "public" / "validation.jsonl", 1)[0]
    monkeypatch.setenv("CONTEST_ARTIFACT_PATH", str(base / "checkpoint.npz"))
    output = predict(SimpleNamespace(task="classification", inputs=row["inputs"]), base)

    assert isinstance(output, list) and len(output) == 2
    assert len({item["label"] for item in output}) == len(output)
    scores = [item["score"] for item in output]
    assert scores.count(max(scores)) == 1
    assert all(0.0 <= score <= 1.0 for score in scores)
