"""Batch-stream protocol: labelled batches in, decisions + predictions out."""

from .protocol import (
    atomic_write_json,
    checkpoint_of,
    decision_for,
    evaluate_rows,
    find_run_state,
    materialise_package,
    read_jsonl,
    write_jsonl,
    write_predictions,
)
from .runner import TaskStreamResult, run_stream, run_task_stream
from .sources import Batch, external_a_batches, make_source, official_batches

__all__ = [
    "Batch",
    "TaskStreamResult",
    "atomic_write_json",
    "checkpoint_of",
    "decision_for",
    "evaluate_rows",
    "external_a_batches",
    "find_run_state",
    "make_source",
    "materialise_package",
    "official_batches",
    "read_jsonl",
    "run_stream",
    "run_task_stream",
    "write_jsonl",
    "write_predictions",
]
