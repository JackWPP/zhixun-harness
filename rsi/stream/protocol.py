"""On-disk protocol pieces shared by the batch stream runner.

The harness itself (``rsi.runner.Runner``) already knows how to train, gate and
record one round on a package directory.  This module only prepares the tiny
per-batch package a stream implies, reads back the resulting state, and writes
the artefacts the platform protocol asks for:

* ``decision.json``    – promote / retain / rollback *before* the next batch
* ``predictions.jsonl``– one line per current-batch row, official output shape
* ``run/``             – the untouched Runner artefacts (state, rounds, summary)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..data import load_jsonl_dataset
from ..inference import apply_postprocess, format_outputs, resolve_postprocess
from ..metrics import from_predictions, primary
from ..model import ExtendedModel, checkpoint_sha256
from ..tasks import TaskConfig
from .sources import Batch


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def materialise_package(work_dir: str | Path, config: TaskConfig, batch: Batch,
                        parent_checkpoint: str | Path) -> TaskConfig:
    """Write ``work_dir/package`` and return a TaskConfig pointing at it.

    Honest protocol: the package contains only *revealed historical* labels
    (``batch.train`` for fitting, ``batch.dev`` for selection) and the current
    batch inputs without labels (``prediction.jsonl``).  Current/hidden answers
    never enter the package directory.
    """

    work_dir = Path(work_dir)
    package = work_dir / "package"
    public = package / "public"
    public.mkdir(parents=True, exist_ok=True)
    write_jsonl(public / "train.jsonl", batch.train)
    write_jsonl(public / "validation.jsonl", batch.dev)
    write_jsonl(public / "prediction.jsonl", batch.prediction)
    target = package / "checkpoint.npz"
    shutil.copy2(Path(parent_checkpoint), target)
    spec = replace(
        config,
        package=str(package.resolve()),
        data_root="",
        train_file="public/train.jsonl",
        validation_file="public/validation.jsonl",
        checkpoint_file="checkpoint.npz",
        split_strategy="stratified" if config.task == "classification" else "group",
        probe="none",
        extended={},
        rounds=1,
    )
    return spec


def predict_rows(rows_path: str | Path, checkpoint: str | Path,
                 postprocess: dict | None, config: TaskConfig,
                 out_path: str | Path) -> int:
    """Predict an unlabeled JSONL file and write official-shape predictions."""

    from ..data import flatten

    rows = read_jsonl(rows_path)
    if not rows:
        write_jsonl(out_path, [])
        return 0
    vectors = []
    identities = []
    for row in rows:
        _, values = flatten(row.get("inputs") or {})
        vectors.append(values)
        identities.append(str(row.get("id", "")))
    x = np.asarray(vectors, dtype=np.float64)
    model = ExtendedModel.from_checkpoint(Path(checkpoint))
    spec = resolve_postprocess(postprocess or {"type": "none"}, model)
    predicted = apply_postprocess(spec, model, x)
    out = []
    for i, identity in enumerate(identities):
        formatted = format_outputs(predicted[i], config.task, config.classes)
        out.append({"id": identity, "outputs": formatted["outputs"]})
    write_jsonl(out_path, out)
    return len(out)


def score_predictions(predictions_path: str | Path, answers: dict[str, str],
                      task: str, classes: list[str] | None) -> dict:
    """Post-hoc scoring of submitted predictions; never fed back to the harness."""

    rows = read_jsonl(predictions_path)
    if not rows:
        return {"rows": 0, "primary": None}
    try:
        if task == "classification":
            label_index = {str(name): i for i, name in enumerate(classes or [])}
            probabilities = np.zeros((len(rows), len(label_index)), dtype=np.float64)
            expected = np.zeros(len(rows), dtype=np.int64)
            for i, row in enumerate(rows):
                scores = {}
                for item in row.get("outputs") or []:
                    name = str(item.get("label"))
                    if name in label_index:
                        scores[label_index[name]] = float(item.get("score", 0.0))
                for j, value in scores.items():
                    probabilities[i, j] = value
                expected[i] = label_index[str(answers.get(str(row.get("id")), ""))]
            values = from_predictions(task, probabilities, expected, classes)
        else:
            predicted = np.asarray([float((row.get("outputs") or {}).get("value", 0.0)) for row in rows])
            expected = np.asarray([float(answers.get(str(row.get("id")), "nan")) for row in rows])
            values = from_predictions(task, predicted, expected, classes)
    except (KeyError, TypeError, ValueError) as exc:
        return {"rows": len(rows), "primary": None, "error": f"{type(exc).__name__}: {exc}"}
    values["primary"] = primary(task, values)
    values["rows"] = len(rows)
    return values


def latest_run_dir(root: str | Path) -> Path | None:
    root = Path(root)
    candidates = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("rsi-")])
    return candidates[-1] if candidates else None


def find_run_state(run_dir: str | Path) -> dict:
    """Read the state and summary of a finished Runner run."""

    run_dir = Path(run_dir)
    state_path = run_dir / "state.json"
    summary_path = run_dir / "summary.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {"run_dir": str(run_dir), "state": state, "summary": summary}


def decision_for(task: str, direction: str, parent_primary: float | None,
                 candidate_primary: float | None, margin: float) -> dict:
    """Turn two public-selection numbers into a deployment decision."""

    if candidate_primary is None or parent_primary is None:
        return {"decision": "retain", "delta": 0.0,
                "reason": "missing parent or candidate selection metric; keep current model"}
    if direction == "higher":
        delta = float(candidate_primary) - float(parent_primary)
        if delta >= margin:
            kind = "promote"
        elif delta <= -2.0 * margin:
            kind = "rollback"
        else:
            kind = "retain"
    else:
        delta = float(parent_primary) - float(candidate_primary)
        if delta >= margin:
            kind = "promote"
        elif delta <= -2.0 * margin:
            kind = "rollback"
        else:
            kind = "retain"
    return {"decision": kind, "delta": delta,
            "reason": f"selection delta {delta:.6g} vs margin {margin:.6g}"}


def evaluate_rows(task: str, validation_path: str | Path, checkpoint: str | Path,
                  postprocess: dict | None, classes: list[str] | None = None) -> dict:
    """Evaluate a checkpoint on a labelled JSONL file (train-only fit already frozen)."""

    dataset = load_jsonl_dataset(validation_path, task, classes)
    model = ExtendedModel.from_checkpoint(Path(checkpoint))
    spec = resolve_postprocess(postprocess or {"type": "none"}, model, train=dataset)
    predicted = apply_postprocess(spec, model, dataset.x)
    values = from_predictions(task, predicted, dataset.y, classes)
    values["primary"] = primary(task, values)
    values["rows"] = len(dataset)
    return values


def write_predictions(validation_path: str | Path, checkpoint: str | Path,
                      postprocess: dict | None, config: TaskConfig,
                      out_path: str | Path) -> int:
    dataset = load_jsonl_dataset(validation_path, config.task, config.classes)
    model = ExtendedModel.from_checkpoint(Path(checkpoint))
    spec = resolve_postprocess(postprocess or {"type": "none"}, model, train=dataset)
    predicted = apply_postprocess(spec, model, dataset.x)
    rows = []
    for i, identity in enumerate(dataset.ids):
        formatted = format_outputs(predicted[i], config.task, config.classes)
        rows.append({"id": identity, "outputs": formatted["outputs"]})
    write_jsonl(out_path, rows)
    return len(rows)


def checkpoint_of(path: str | Path) -> dict:
    path = Path(path)
    return {"checkpoint": str(path.resolve()), "sha256": checkpoint_sha256(path)}
