"""Tests for the submission bundle builder (P5)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_submission.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("make_submission", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_run(tmp_path: Path) -> Path:
    source = ROOT / "datasets" / "cpu-b-rul-20261110-v1" / "checkpoint.npz"
    sha = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    run_dir = tmp_path / "run"
    (run_dir / "B").mkdir(parents=True)
    (run_dir / "B" / "experiments" / "exp-0001").mkdir(parents=True)
    shutil.copy2(source, run_dir / "B" / "experiments" / "exp-0001" / "candidate.npz")
    state = {
        "schema_version": 2,
        "run": {"run_id": "test-run"},
        "tasks": [{
            "key": "B", "task": "regression", "primary_metric": "rmse", "controller": "pipeline",
            "postprocess": {"type": "shrink", "alpha": 0.0, "center": 600.0},
            "rounds": [{"index": 1, "decision": "retain", "delta": 0.0,
                        "hypothesis": "baseline is worse than the constant",
                        "change_summary": "retain", "validation_metrics": {"rmse": 681.0},
                        "parent_checkpoint_sha256": sha, "checkpoint_sha256": sha,
                        "code_sha256": "sha256:aaa"}],
        }],
    }
    summary = {"datasets": [{"key": "B", "metrics": {"primary": 681.0, "rmse": 681.0},
                             "baseline_checkpoint_sha256": sha}]}
    (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_bundle_builds_and_predicts(tmp_path):
    module = _load_module()
    run_dir = _fake_run(tmp_path)
    archive = module.build(run_dir, tmp_path / "dist")
    assert archive.is_file()
    assert archive.stat().st_size <= module.LIMIT_BYTES
    task_dir = tmp_path / "dist" / "submission_test-run" / "B"
    for name in ("checkpoint.npz", "postprocess.json", "predict.py", "manifest.json"):
        assert (task_dir / name).is_file(), name
    manifest = json.loads((task_dir / "manifest.json").read_text())
    assert manifest["task"] == "B"
    assert manifest["postprocess"]["type"] == "shrink"
    assert manifest["rounds"][0]["checkpoint_sha256"].startswith("sha256:")

    request = json.dumps({"task": "", "inputs": {
        "features": {str(index): 0.1 for index in range(1, 19)},
        "bearing_context": [0.0] * 16}})
    proc = subprocess.run([sys.executable, str(task_dir / "predict.py")],
                          input=request + "\n", text=True, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[0])
    assert "outputs" in payload and "value" in payload["outputs"]
