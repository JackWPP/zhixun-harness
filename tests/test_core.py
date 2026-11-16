"""Core harness tests: gate, splits, inference, predictions and records."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rsi.inference import apply_postprocess, format_outputs
from rsi.metrics import from_predictions, primary
from rsi.model import ExtendedModel, checkpoint_sha256
from rsi.records import Recorder
from rsi.selection import gate
from rsi.splits import build_splits
from rsi.tasks import resolve_tasks


# ---------------------------------------------------------------- selection
def test_gate_classification_promote():
    decision = gate("classification", 0.70, 0.72, margin=0.004)
    assert decision.decision == "promote"
    assert decision.delta == pytest.approx(0.02)


def test_classification_gate_rejects_gain_from_tiny_class_when_common_classes_regress():
    parent = {"primary": 0.70, "rows": 100,
              "per_class_f1": {"0": 0.70, "1": 0.80, "2": 0.50},
              "per_class_support": {"0": 50, "1": 48, "2": 2}}
    candidate = {"primary": 0.72, "rows": 100,
                 "per_class_f1": {"0": 0.65, "1": 0.80, "2": 0.71},
                 "per_class_support": {"0": 50, "1": 48, "2": 2}}
    decision = gate("classification", parent["primary"], candidate["primary"], 0.004,
                    parent, candidate)
    assert decision.decision == "retain"
    assert "support-qualified class macro_f1 regressed" in decision.reason


def test_classification_gate_allows_tiny_class_gain_without_common_class_regression():
    parent = {"primary": 0.70, "rows": 100,
              "per_class_f1": {"0": 0.70, "1": 0.80, "2": 0.50},
              "per_class_support": {"0": 50, "1": 48, "2": 2}}
    candidate = {"primary": 0.72, "rows": 100,
                 "per_class_f1": {"0": 0.71, "1": 0.80, "2": 1.00},
                 "per_class_support": {"0": 50, "1": 48, "2": 2}}
    decision = gate("classification", parent["primary"], candidate["primary"], 0.004,
                    parent, candidate)
    assert decision.decision == "promote"


def test_gate_regression_promote_needs_lower_rmse():
    assert gate("regression", 52.0, 51.0, margin=0.3).decision == "promote"
    assert gate("regression", 52.0, 52.1, margin=0.3).decision == "retain"
    assert gate("regression", 52.0, 54.0, margin=0.3).decision == "rollback"


# ------------------------------------------------------------------- splits
@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    config = resolve_tasks(["C"])[0]
    state = tmp_path_factory.mktemp("splits")
    return config, build_splits(config, state, seed=42)


def test_split_groups_do_not_leak_engine_units(splits):
    config, data = splits
    cal = set(data["calibration"].groups)
    selection = set(data["selection"].groups)
    report = set(data["report"].groups)
    assert cal.isdisjoint(selection) and selection.isdisjoint(report) and cal.isdisjoint(report)


def test_split_manifest_is_recorded(splits):
    config, data = splits
    manifest = data["manifest"]
    assert manifest["seed"] == 42
    rows = [manifest["splits"][name]["rows"] for name in ("calibration", "selection", "report")]
    assert all(row > 0 for row in rows)
    assert manifest["train"]["rows"] > 0


# ---------------------------------------------------------------- inference
def test_recenter_postprocess_keeps_shapes():
    config = resolve_tasks(["A"])[0]
    model = ExtendedModel.from_checkpoint(config.checkpoint_path)
    x = np.random.default_rng(0).normal(size=(11, len(model.mean))).astype(np.float64)
    raw = model.predict(x)
    spec = {"type": "recenter", "mode": "batch_mean", "weight": 0.4}
    calibrated = apply_postprocess(spec, model, x)
    assert calibrated.shape == raw.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(apply_postprocess({"type": "none"}, model, x), raw)


def test_shrink_postprocess_is_monotone():
    config = resolve_tasks(["C"])[0]
    model = ExtendedModel.from_checkpoint(config.checkpoint_path)
    x = np.random.default_rng(1).normal(size=(9, len(model.mean))).astype(np.float64)
    raw = model.predict(x)
    shrunk = apply_postprocess({"type": "shrink", "alpha": 0.2, "center": 100.0}, model, x)
    assert np.allclose(shrunk, 0.2 * np.asarray(raw).reshape(-1) + 0.8 * 100.0)


def test_format_outputs_matches_official_contract():
    classification = format_outputs(np.asarray([0.2, 0.8]), "classification", ("a", "b"))
    assert classification["outputs"][0]["label"] == "a"
    assert classification["outputs"][1] == {"label": "b", "score": pytest.approx(0.8)}
    regression = format_outputs(12.5, "regression", ())
    assert regression == {"outputs": {"value": 12.5}}


# ---------------------------------------------------------------- prediction
def test_predict_cli_roundtrip():
    config = resolve_tasks(["A"])[0]
    records = [json.loads(line) for line in
               config.validation_path.read_text(encoding="utf-8").splitlines()[:3]]
    requests = [json.dumps({"task": config.task, "inputs": record["inputs"]})
                for record in records]
    completed = subprocess.run(
        [sys.executable, "-m", "rsi.predict", "--checkpoint", str(config.checkpoint_path)],
        input="\n".join(requests) + "\n", text=True, capture_output=True, cwd=REPO, check=True)
    lines = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 3
    for line in lines:
        assert len(line["outputs"]) == len(config.classes)
        assert all(0.0 <= item["score"] <= 1.0 for item in line["outputs"])


# ------------------------------------------------------------------- records
def test_recorder_round_trip(tmp_path):
    recorder = Recorder(tmp_path, "rsi-test")
    recorder.event("info", "A", "hello")
    recorder.round({"id": "A-r1", "task": "A", "index": 1, "decision": "promote"})
    recorder.write_state({"run": {"run_id": "rsi-test"}, "tasks": [], "events": []})
    assert recorder.read_records()[0]["type"] == "event"
    assert recorder.read_rounds("A")[0]["decision"] == "promote"
    state = json.loads(recorder.state_path.read_text())
    assert state["run"]["run_id"] == "rsi-test"


# -------------------------------------------------------- metrics semantics
def test_metrics_primary_direction():
    predicted = np.eye(3)[[0, 1, 1, 2]]
    expected = np.asarray([0, 1, 2, 2])
    values = from_predictions("classification", predicted, expected, ("x", "y", "z"))
    assert 0.0 <= primary("classification", values) <= 1.0
    regression = from_predictions("regression", np.asarray([1.0, 2.0]),
                                  np.asarray([2.0, 4.0]), ())
    assert regression["rmse"] == pytest.approx(1.5811388300841898)


def test_checkpoint_sha_is_stable():
    config = resolve_tasks(["A"])[0]
    assert checkpoint_sha256(config.checkpoint_path).startswith("sha256:")
    assert checkpoint_sha256(config.checkpoint_path) == checkpoint_sha256(config.checkpoint_path)
