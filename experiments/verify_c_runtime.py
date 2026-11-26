"""Recompute all C emulator artifacts against the complete official validation set."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.c_candidates import load_rows, metrics, runtime_info
from rsi.model import ExtendedModel
from rsi.platform.c_safety import group_guard

report = json.loads((REPO / "outputs/c_runtime_10rounds.json").read_text())
source = Path(report["root"])
model = ExtendedModel.from_checkpoint(REPO / "datasets/cpu-c-guide-20261110-v1/checkpoint.npz")
x, y, groups, validation_sha = load_rows(REPO / "datasets/cpu-c-guide-20261110-v1/public/validation.jsonl", model)
previous = metrics(model.predict(x), y, groups)
previous_pred = model.predict(x)
row_ids = [json.loads(line)["id"] for line in (REPO / "datasets/cpu-c-guide-20261110-v1/public/validation.jsonl").read_text().splitlines()]
summary = {"runtime": runtime_info(), "validation_sha256": validation_sha, "fake_gateway": True, "rounds": []}
for row in report["rounds"]:
    index = row["round_index"]
    artifact = source / f"artifact-{index:02d}"
    path = artifact / "checkpoint.npz"
    model = ExtendedModel.from_checkpoint(path)
    predicted = model.predict(x)
    values = metrics(predicted, y, groups)
    guard = group_guard(previous_pred, predicted, y, row_ids, enforce_rul=True)
    assert guard["passed"]
    meta = json.loads((artifact / "submission_meta.json").read_text())
    for metric in ("rmse", "mae", "nasa"):
        assert values[metric] <= previous[metric] + abs(previous[metric]) * 1e-12
    assert len(row["model_call_ids"]) == 1
    assert meta["data_boundary"]["validation_used_for_fit"] is False
    state = source / f"state-{index:02d}"
    journal = (state / "round_records.jsonl").read_text().splitlines()
    assert len(journal) == index, "native state was not restored across rounds"
    configuration = json.loads(journal[-1])["configuration"]
    candidate_audit = [{key: entry.get(key) for key in
                       ("kind", "training_action", "experiment_cache", "alpha")}
                      for entry in configuration["candidates"]]
    continuation = [entry for entry in candidate_audit if entry["experiment_cache"]]
    assert len(continuation) == 1
    assert continuation[0]["experiment_cache"]["cache_hit"] == (index >= 3)
    assert len([p for p in state.rglob("*") if p.is_file()]) <= 64
    summary["rounds"].append({"round": index, "chosen": meta["chosen"],
        "metrics": {key: values[key] for key in ("rmse", "mae", "nasa")},
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_call_count": len(row["model_call_ids"]), "resource": meta["resource"],
        "group_guard": guard, "candidates": candidate_audit,
        "journal_rounds": len(journal), "state_sha256": row["state_sha256"]})
    previous = values
    previous_pred = predicted
last = REPO / "outputs/c_runtime_artifact"
last.mkdir(exist_ok=True)
for filename in ("checkpoint.npz", "submission_meta.json", "data_boundary.json"):
    shutil.copyfile(artifact / filename, last / filename)
(REPO / "outputs/c_runtime_verification.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
