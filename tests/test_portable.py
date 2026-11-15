"""The runtime must accept a task it has never seen (private/hidden data path).

This is the portability proof for the competition: a new dataset package is
onboarded through ``task.json`` only — no change to rsi/ is required.
"""
from __future__ import annotations

import json

import pytest

from rsi.tasks import TASKS, resolve_tasks
from rsi.runner import Runner
from scripts.make_demo_task import make as make_demo_task


@pytest.fixture()
def demo_task(tmp_path):
    result = make_demo_task(tmp_path / "demo")
    spec_path = tmp_path / "demo" / "task.json"
    from rsi.tasks import load_task_spec, register_task_specs
    config = load_task_spec(spec_path)
    TASKS[config.key] = config
    yield config
    TASKS.pop(config.key, None)


def test_task_json_fully_describes_a_new_dataset(demo_task):
    config = demo_task
    assert config.key == "D"
    assert config.task == "classification"
    assert config.classes == ("alpha", "beta", "gamma")
    assert config.train_path.is_file() and config.validation_path.is_file()
    assert config.checkpoint_path.is_file()
    assert config.split_strategy == "stratified"
    assert resolve_tasks(["D"])[0] is config


def test_new_task_builds_splits_with_advertised_schema(demo_task, tmp_path):
    from rsi.splits import build_splits
    bundle = build_splits(demo_task, tmp_path / "state", seed=42)
    manifest = bundle["manifest"]
    assert manifest["protocol"]["split_strategy"] == "stratified"
    rows = [manifest["splits"][name]["rows"] for name in ("calibration", "selection", "report")]
    assert sum(rows) == len(bundle["validation"])
    assert all(value > 0 for value in rows)
    assert bundle["probe"] is None  # probe=none in the spec


def test_new_task_runs_one_honest_round(demo_task, tmp_path):
    run = Runner(["D"], rounds=1, llm_mode="off", state_dir=tmp_path / "run",
                 run_dir=tmp_path / "run" / "one", memory=False)
    summary = run.run()
    dataset = summary["datasets"][0]
    assert dataset["key"] == "D"
    assert len(dataset["rounds"]) == 1
    assert dataset["record_complete"] is True
    # no validation-derived data may be fitted
    rounds_path = tmp_path / "run" / "one" / "rounds.jsonl"
    for line in rounds_path.read_text().splitlines():
        record = json.loads(line)
        for experiment in record["experiments"]:
            data = (experiment.get("proposal", {}).get("params") or {}).get("data")
            assert data in (None, "train")
