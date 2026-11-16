"""Batch-stream protocol: labelled batches in, decisions + predictions out."""

from .protocol import (
    atomic_write_json,
    checkpoint_of,
    decision_for,
    evaluate_rows,
    find_run_state,
    materialise_package,
    predict_rows,
    read_jsonl,
    score_predictions,
    write_jsonl,
    write_predictions,
)
from .runner import TaskStreamResult, run_stream, run_task_stream
from .sources import Batch, make_source, official_batches, private_sim_batches

__all__ = [
    "Batch",
    "TaskStreamResult",
    "atomic_write_json",
    "checkpoint_of",
    "decision_for",
    "evaluate_rows",
    "private_sim_batches",
    "find_run_state",
    "make_source",
    "materialise_package",
    "predict_rows",
    "score_predictions",
    "official_batches",
    "read_jsonl",
    "run_stream",
    "run_task_stream",
    "write_jsonl",
    "write_predictions",
]
