"""Cumulative revealed-label pool persisted in the participant state.

The platform reveals the previous stage's labels together with the next
projection (``/data/train`` grows round by round).  Carrying those rows in the
participant state is what makes bounded re-training possible: the pool survives
the container restart between rounds because it is written through
``rnd.save_state``.

Only rows the platform actually projected are stored; the pool never invents
labels and it is capped so the state bundle stays small.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POOL_NAME = "label_pool.npz"
MAX_ROWS = 40000
MAX_BYTES = 16 * 1024 * 1024


class LabelPool:
    """Feature/label rows accumulated across rounds."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / POOL_NAME
        self.x = np.zeros((0, 0), dtype=np.float32)
        self.y = np.zeros((0,), dtype=np.float64)
        self.ids = []
        self.rounds = []
        self.dropped = 0

    # -- persistence -----------------------------------------------------
    @classmethod
    def load(cls, root: Path) -> "LabelPool":
        pool = cls(root)
        target = pool.path
        if not target.is_file():
            return pool
        try:
            with np.load(target, allow_pickle=False) as arrays:
                x = np.asarray(arrays["x"], dtype=np.float32)
                y = np.asarray(arrays["y"], dtype=np.float64).reshape(-1)
                ids = [str(value) for value in arrays["ids"].tolist()]
                rounds = [int(value) for value in arrays["rounds"].tolist()]
        except (OSError, KeyError, ValueError):
            return pool
        if len(x) and (x.ndim != 2 or len(x) != len(y) != len(ids)):
            return pool
        pool.x, pool.y, pool.ids, pool.rounds = x, y, ids, rounds
        return pool

    def save(self) -> bool:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"x": self.x, "y": self.y,
                   "ids": np.asarray(self.ids, dtype=object) if False else np.asarray(self.ids, dtype="U128"),
                   "rounds": np.asarray(self.rounds, dtype=np.int64)}
        temporary = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **payload)
        if temporary.stat().st_size > MAX_BYTES:
            temporary.unlink(missing_ok=True)
            return False
        temporary.replace(self.path)
        return True

    # -- mutation --------------------------------------------------------
    @property
    def size(self) -> int:
        return int(len(self.y))

    def merge(self, x: np.ndarray, y: np.ndarray, ids, round_index: int) -> int:
        """Merge rows, deduplicating by row id, keeping the newest copy."""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        ids = [str(value) for value in ids]
        if len(x) != len(y) != len(ids):
            raise ValueError("label pool merge shape mismatch")
        if self.size and self.x.shape[1] != x.shape[1]:
            # schema changed between rounds: keep only the newest projection
            self.x = np.zeros((0, x.shape[1]), dtype=np.float32)
            self.y = np.zeros((0,), dtype=np.float64)
            self.ids, self.rounds = [], []
        known = {row_id: position for position, row_id in enumerate(self.ids)}
        added = 0
        for position, row_id in enumerate(ids):
            if row_id in known:
                index = known[row_id]
                self.y[index] = y[position]
                self.rounds[index] = int(round_index)
                continue
            self.x = np.vstack([self.x, x[position:position + 1]])
            self.y = np.append(self.y, y[position])
            self.ids.append(row_id)
            self.rounds.append(int(round_index))
            known[row_id] = len(self.ids) - 1
            added += 1
        if self.size > MAX_ROWS:
            keep = self.size - MAX_ROWS
            self.x = self.x[keep:]
            self.y = self.y[keep:]
            self.ids = self.ids[keep:]
            self.rounds = self.rounds[keep:]
            self.dropped += keep
        return added

    def arrays(self):
        return self.x, self.y

    def summary(self) -> dict:
        return {"rows": self.size, "rounds": int(self.rounds[-1]) if self.rounds else 0,
                "dropped": self.dropped}
