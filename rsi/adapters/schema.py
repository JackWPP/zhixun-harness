"""Strict JSONL adapter for private/hidden-style datasets.

The official data reader ({mod}`rsi.data_official`) already enforces the
submission-facing contract: unique string ids, finite numeric inputs, no
target-derived fields, class names inside the configured list, non-negative
RUL answers.  This module only adds friendly error wrapping and a schema
summary so the harness can validate an unknown file before spending compute.
"""
from __future__ import annotations

from pathlib import Path

from rsi.data import Dataset, load_jsonl_dataset


class SchemaError(ValueError):
    """Raised when a private JSONL file does not match the official schema."""


def load_private_dataset(config, path) -> Dataset:
    """Load and validate a private JSONL file for the given task config.

    Never fits anything: the returned dataset is only fed to the frozen model
    and the metric functions.  Training data must never come from here.
    """
    if path is None:
        raise SchemaError("private dataset path is required")
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise SchemaError(f"private dataset not found: {candidate}")
    try:
        return load_jsonl_dataset(candidate, config.task, config.classes)
    except ValueError as exc:
        raise SchemaError(f"{candidate}: {exc}") from exc


def describe_dataset(dataset: Dataset, config) -> dict:
    """Small, printable summary that never exposes labels of a hidden set."""
    groups = sorted(set(dataset.groups))
    return {
        "name": dataset.name,
        "rows": len(dataset),
        "features": int(dataset.x.shape[1]) if dataset.x.size else 0,
        "task": config.task,
        "classes": list(config.classes) if config.task == "classification" else [],
        "groups": groups[:200],
        "group_count": len(groups),
        "sha256": dataset.sha256,
        "used_for_training": False,
    }
