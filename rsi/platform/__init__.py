"""Workspace v2 participant adapter for the zhixun RSI harness."""
from .submission import TASKS, predict, train  # noqa: F401

__all__ = ["TASKS", "predict", "train"]
