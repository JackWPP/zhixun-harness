"""Exercise the real container predictor with official requests and live pipes."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import selectors
import subprocess
import sys

import numpy as np
import pytest

from rsi import predict
from rsi.model import ExtendedModel
from rsi.tasks import TASKS

ROOT = Path(__file__).resolve().parents[1]


def requests_for(key, count=3):
    config = TASKS[key]
    requests = []
    with config.validation_path.open(encoding="utf-8") as handle:
        for _, line in zip(range(count), handle):
            requests.append({"task": config.task, "inputs": json.loads(line)["inputs"]})
    return requests


def run_cli(key, rows, *args):
    return subprocess.run(
        [sys.executable, "-m", "rsi.predict", "--checkpoint", str(TASKS[key].checkpoint_path), *args],
        input="".join(json.dumps(row) + "\n" for row in rows), text=True,
        capture_output=True, cwd=ROOT, timeout=20,
    )


@pytest.mark.parametrize("key", ["A", "B", "C"])
@pytest.mark.parametrize("mode", ["batch", "stream"])
def test_real_official_requests_match_reference_predictor(key, mode):
    rows = requests_for(key)
    actual = run_cli(key, rows, "--mode", mode)
    reference = subprocess.run(
        [sys.executable, str(TASKS[key].package_dir / "baselines" / "predict.py"),
         "--checkpoint", str(TASKS[key].checkpoint_path)],
        input="".join(json.dumps(row) + "\n" for row in rows), text=True,
        capture_output=True, cwd=ROOT, timeout=20,
    )
    assert actual.returncode == reference.returncode == 0, (actual.stderr, reference.stderr)
    assert not actual.stderr
    outputs = [json.loads(line)["outputs"] for line in actual.stdout.splitlines()]
    expected = [json.loads(line)["outputs"] for line in reference.stdout.splitlines()]
    assert len(outputs) == len(rows)
    for got, want in zip(outputs, expected):
        if key == "A":
            assert [v["label"] for v in got] == [v["label"] for v in want]
            assert [v["score"] for v in got] == pytest.approx([v["score"] for v in want])
        else:
            assert got["value"] == pytest.approx(want["value"])


@pytest.mark.parametrize("key", ["A", "B", "C"])
def test_task_alias_matches_checkpoint_profile(key):
    model = ExtendedModel.from_checkpoint(TASKS[key].checkpoint_path)
    row = requests_for(key, 1)[0]
    expected = predict._features(row, model)
    for alias in (key, key.lower()):
        row["task"] = alias
        np.testing.assert_array_equal(predict._features(row, model), expected)
    del row["task"]
    np.testing.assert_array_equal(predict._features(row, model), expected)


@pytest.mark.parametrize("mode", ["batch", "stream"])
def test_wrong_regression_profile_is_rejected(mode):
    row = requests_for("B", 1)[0]
    row["task"] = "C"
    process = run_cli("B", [row], "--mode", mode)
    assert process.returncode != 0
    assert process.stdout == ""
    assert "line 1: prediction task mismatch" in process.stderr


def test_equal_width_wrong_feature_paths_rejected():
    model = ExtendedModel.from_checkpoint(TASKS["C"].checkpoint_path)
    row = requests_for("C", 1)[0]
    row["inputs"]["other_settings"] = row["inputs"].pop("settings")
    with pytest.raises(ValueError, match="feature schema differs"):
        predict._features(row, model)


@pytest.mark.parametrize("raw, message", [
    ('{"inputs":{"features":[NaN]}}', "nonfinite JSON number"),
    ('{"inputs":{},"inputs":{}}', "duplicate JSON key"),
    ('{"inputs":{"rul":10}}', "target-derived input field"),
    ('{"inputs":{"features":[true]}}', "finite numeric features"),
    ('{"inputs":{},"answer":1}', "requires only task and inputs"),
    ('[]', "requires only task and inputs"),
])
def test_invalid_jsonl_fails_clearly_on_stderr(raw, message, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n" + raw + "\n"))
    assert predict.main(["--checkpoint", str(TASKS["A"].checkpoint_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "line 2:" in captured.err
    assert message in captured.err


def test_stream_answers_each_request_without_eof():
    process = subprocess.Popen(
        [sys.executable, "-m", "rsi.predict", "--checkpoint", str(TASKS["C"].checkpoint_path),
         "--mode", "stream"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, cwd=ROOT,
    )
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        for row in requests_for("C", 2):
            process.stdin.write(json.dumps(row) + "\n")
            process.stdin.flush()
            assert selector.select(timeout=5), "predictor blocked waiting for stdin EOF"
            output = json.loads(process.stdout.readline())
            assert np.isfinite(output["outputs"]["value"])
            assert process.poll() is None
        process.stdin.close()
        assert process.wait(timeout=5) == 0
        assert process.stderr.read() == ""
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for pipe in (process.stdin, process.stdout, process.stderr):
            pipe.close()


def test_stream_calibration_does_not_depend_on_future_rows():
    rows = requests_for("A", 14)
    changed = copy.deepcopy(rows)
    for row in changed[10:]:
        row["inputs"]["features"] = [v * 100 + 1000 for v in row["inputs"]["features"]]
    args = ("--mode", "stream", "--postprocess",
            '{"type":"recenter","mode":"batch_mean","weight":0.5}')
    prefix, full, different_future = (run_cli("A", selected, *args)
                                      for selected in (rows[:10], rows, changed))
    assert all(p.returncode == 0 for p in (prefix, full, different_future))
    assert prefix.stdout.splitlines() == full.stdout.splitlines()[:10]
    assert prefix.stdout.splitlines() == different_future.stdout.splitlines()[:10]


def test_batch_validates_entire_request_before_output():
    rows = requests_for("A", 2)
    rows[1]["inputs"]["features"].pop()
    process = run_cli("A", rows, "--mode", "batch")
    assert process.returncode == 2
    assert process.stdout == ""
    assert "line 2: prediction feature schema differs" in process.stderr
