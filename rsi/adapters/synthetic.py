"""Deterministic synthetic drift probes built from the official public splits.

Why: the official validation split is a single batch, so the harness cannot
observe its own robustness against distribution shift without touching hidden
test data.  These generators create *evaluation-only* pseudo-batches with known
drift axes (gain / offset / noise / unit degradation).  They are written to the
run directory and never handed to the training executor (Guard enforces that).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rsi.data_official import flatten, json_value


def _read_rows(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json_value(line))
    return rows


def _map_numbers(value, fn):
    if isinstance(value, dict):
        return {key: _map_numbers(child, fn) for key, child in value.items()}
    if isinstance(value, list):
        return [_map_numbers(child, fn) for child in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(fn(float(value)))
    return value


def _feature_stats(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    matrix = []
    for row in rows:
        _paths, values = flatten(row["inputs"])
        matrix.append(values)
    data = np.asarray(matrix, dtype=np.float64)
    return data.mean(axis=0), np.maximum(data.std(axis=0), 1e-9)


def synthesize(config, source_train, *, batches: int = 4, seed: int = 42,
               drift: float = 2.5, noise: float = 0.25) -> list[dict]:
    """Create ``batches`` pseudo-batches with deterministic gain/offset/noise drift."""
    rows = _read_rows(source_train)
    if not rows:
        raise ValueError("source train split is empty")
    mean, std = _feature_stats(rows)
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, 0.35, size=len(mean))
    gains = rng.normal(0.0, 0.25, size=len(mean))
    output: list[dict] = []
    for batch in range(max(1, int(batches))):
        progress = (batch + 1) / max(1, batches)
        # Every batch shifts a different, growing amount; the final batch is the
        # hardest one, mirroring the drift phases the platform evaluates.
        gain = 1.0 + drift * progress * gains
        offset = drift * progress * std * offsets
        sigma = noise * (0.5 + progress) * std
        for index, row in enumerate(rows):
            counter = {"i": 0}

            def transform(value: float, _counter=counter) -> float:
                position = _counter["i"] % len(mean)
                _counter["i"] += 1
                return value * gain[position] + offset[position] + rng.normal(0.0, sigma[position])

            clone = {
                "id": f"synbatch{batch + 1}_r{index:05d}_{row['id']}",
                "inputs": _map_numbers(row["inputs"], transform),
                "answer": row["answer"],
                "meta": {"synthetic": True, "batch": batch + 1, "source_id": row["id"],
                         "drift": {"batch": batch + 1, "progress": round(progress, 4),
                                   "gain": drift * progress, "noise": round(noise * (0.5 + progress), 4)}},
            }
            output.append(clone)
    return output


def write_synthetic(rows: list[dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
