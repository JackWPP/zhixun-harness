"""Evaluate a private/hidden-style probe with a finished checkpoint.

This is the P4 capability: the harness can point at *any* JSONL file that
follows the official schema (``id`` / ``inputs`` / ``answer``) and report
metrics for it without ever training on it.  Compliance argument: the private
file is only read after the model is fully trained, and the Guard whitelist
(``train`` / ``train+calibration``) makes it impossible to hand the path to the
training executor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rsi.inference import apply_postprocess, resolve_postprocess
from rsi.metrics import from_predictions, primary as primary_metric
from rsi.model import ExtendedModel, checkpoint_sha256

from .schema import describe_dataset, load_private_dataset


def _group_slices(groups: list[str]):
    order: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        order.setdefault(group, []).append(index)
    return {group: np.asarray(index, dtype=int) for group, index in order.items()}


def evaluate_private(config, path, checkpoint, postprocess=None, *,
                     max_groups: int = 50, group_metrics: bool = True) -> dict:
    """Return per-file and per-group metrics, plus a machine-readable report."""
    dataset = load_private_dataset(config, path)
    model = ExtendedModel.from_checkpoint(checkpoint)
    if model.meta["task"] != config.task:
        raise ValueError(f"checkpoint task {model.meta['task']} != {config.task}")
    expected_width = len(model.meta.get("paths") or [])
    if expected_width and int(dataset.x.shape[1]) != int(expected_width):
        raise ValueError(
            f"feature width mismatch: dataset {dataset.x.shape[1]} != checkpoint {expected_width}")
    spec = resolve_postprocess(postprocess or {"type": "none"}, model)
    predicted = apply_postprocess(spec, model, dataset.x)
    metrics = from_predictions(config.task, predicted, dataset.y, config.classes)
    report = {
        "task": config.key,
        "primary_metric": config.primary_metric,
        "primary": primary_metric(config.task, metrics),
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "postprocess": spec,
        "dataset": describe_dataset(dataset, config),
        "per_group": [],
        "training_isolation": "private file is evaluation-only; Guard rejects it as training data",
    }
    if group_metrics:
        slices = _group_slices(dataset.groups)
        rows = []
        for group, index in list(slices.items())[:max_groups]:
            group_metrics_row = from_predictions(
                config.task, predicted[index], dataset.y[index], config.classes)
            rows.append({"group": group, "rows": int(len(index)),
                         "primary": primary_metric(config.task, group_metrics_row),
                         "metrics": group_metrics_row})
        report["per_group"] = rows
    return report


def evaluate_private_file(config, path, checkpoint, postprocess=None, output=None) -> dict:
    report = evaluate_private(config, path, checkpoint, postprocess)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    return report
