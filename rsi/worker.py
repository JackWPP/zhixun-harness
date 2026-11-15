"""Isolated experiment worker.

Runs exactly one proposed experiment inside its own process (invoked by
``rsi.executor``), then reports selection/report/probe metrics, the resolved
postprocess, the candidate checkpoint digest and its own CPU cost.

The worker never writes outside its experiment directory and never reads the
hidden test stream.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from rsi import metrics as metrics_mod
from rsi.data import load_jsonl_dataset
from rsi.inference import apply_postprocess, resolve_postprocess
from rsi.model import ExtendedModel, checkpoint_sha256, code_sha256
from rsi.strategies import ExperimentContext, run_strategy


def evaluate(task, model, postprocess, dataset, classes):
    if dataset is None or len(dataset) == 0:
        return None
    predicted = apply_postprocess(postprocess, model, dataset.x)
    values = metrics_mod.from_predictions(task, predicted, dataset.y, classes)
    values["primary"] = metrics_mod.primary(task, values)
    values["rows"] = int(len(dataset))
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    job_dir = Path(args.job)
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    result_path = job_dir / "result.json"
    started = time.time()
    cpu_start = time.process_time()
    result = {
        "ok": False,
        "experiment_id": job.get("experiment_id"),
        "parent_checkpoint": job.get("parent_checkpoint"),
        "candidate_checkpoint": None,
        "checkpoint_sha256": None,
        "code_sha256": code_sha256(Path(__file__).resolve().parent.parent),
        "metrics": {},
        "postprocess": job.get("proposal", {}).get("postprocess", {"type": "none"}),
        "cpu_seconds": 0.0,
        "wall_seconds": 0.0,
        "error": None,
        "notes": {},
    }
    try:
        task_key = job["task_key"]
        task = job["task"]
        classes = tuple(job["classes"])
        proposal = job["proposal"]
        parent = ExtendedModel.from_checkpoint(job["parent_checkpoint"])
        if parent.meta["task"] != task or list(parent.meta.get("classes", ())) != list(classes):
            raise ValueError("parent checkpoint does not match the declared task")

        train = load_jsonl_dataset(job["train"], task, classes, parent.meta["paths"])
        calibration = (load_jsonl_dataset(job["calibration"], task, classes, parent.meta["paths"])
                       if job.get("calibration") else None)
        selection = load_jsonl_dataset(job["selection"], task, classes, parent.meta["paths"])
        report = load_jsonl_dataset(job["report"], task, classes, parent.meta["paths"])
        probe = None
        if job.get("drift_probe"):
            probe_path = Path(job["drift_probe"])
            if probe_path.is_file():
                probe = load_jsonl_dataset(probe_path, task, classes, parent.meta["paths"])

        extended = None
        if job.get("extended_train"):
            extended_path = Path(job["extended_train"])
            if extended_path.is_file():
                extended = load_jsonl_dataset(extended_path, task, classes, parent.meta["paths"])
        context = ExperimentContext(
            task_key=task_key, task=task, parent=parent, train=train,
            calibration=calibration, selection=selection, report=report,
            drift=job.get("drift"), drift_probe=probe, extended=extended,
            seed=int(job.get("seed", 42)), classes=classes,
        )
        candidate = run_strategy(context, proposal["strategy"], proposal["params"])
        model = candidate.model
        postprocess = resolve_postprocess(proposal.get("postprocess", {"type": "none"}),
                                          model, calibration)

        candidate_path = Path(job["candidate_checkpoint"])
        model.save(candidate_path)
        result.update({
            "ok": True,
            "candidate_checkpoint": str(candidate_path),
            "checkpoint_sha256": checkpoint_sha256(candidate_path),
            "metrics": {
                "selection": evaluate(task, model, postprocess, selection, classes),
                "report": evaluate(task, model, postprocess, report, classes),
                "probe": evaluate(task, model, postprocess, probe, classes),
            },
            "postprocess": postprocess,
            "notes": {"strategy_notes": candidate.notes,
                      "split_protocol": job.get("split_protocol", {})},
        })
    except Exception as error:  # noqa: BLE001 - the controller needs the message
        result["error"] = f"{type(error).__name__}: {error}"
        result["notes"] = {"traceback": traceback.format_exc()[-4000:]}
    result["cpu_seconds"] = round(time.process_time() - cpu_start, 3)
    result["wall_seconds"] = round(time.time() - started, 3)
    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tmp.replace(result_path)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
