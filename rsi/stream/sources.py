"""Batch sources for streaming evaluation.

A *Batch* is a labelled slice that a platform would release.  The harness is
allowed to train on the batch's train rows and to evaluate on its validation
rows; the cumulative training set grows as batches arrive, mirroring the
competition rule "every time you get a batch, all earlier data is re-sent".

Two families of sources are supported:

* ``official``   – slice the official package validation stream into N batches.
  Training at batch *i* may use the official train split plus the *labels of
  batches 0..i-1* (already delivered), never batch *i*'s validation rows.
* ``external-a`` – the UCI gas batches 3..10 as a Track X continual-adaptation
  test-bed.  Each external batch is split into train/eval rows; the harness may
  adapt its industrial model on the labelled train rows and is scored on the
  eval rows.  This never feeds the official submission lineage.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..tasks import TaskConfig


@dataclass
class Batch:
    """One labelled slice handed to the harness."""

    index: int
    train: list[dict]
    validation: list[dict]
    meta: dict = field(default_factory=dict)


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_order(rows: Iterable[dict]) -> list[str]:
    from ..data import _derive_groups

    return _derive_groups([str(r.get("id", "")) for r in rows])


def _partition_indices(groups: list[str], n_batches: int) -> list[list[int]]:
    """Split ordered rows into *n_batches* contiguous chunks.

    Whole groups (a bearing, an engine unit) are kept together whenever the
    stream has enough distinct groups; otherwise the split is a plain row block.
    """

    if n_batches <= 0:
        raise ValueError("n_batches must be positive")
    unique: list[str] = []
    seen: set[str] = set()
    for g in groups:
        if g not in seen:
            seen.add(g)
            unique.append(g)
    buckets: list[list[int]] = [[] for _ in range(n_batches)]
    if len(unique) >= n_batches:
        counts = Counter(groups)
        total = max(1, len(groups))
        per_bucket = total / float(n_batches)
        current = 0
        weight = 0.0
        for g in unique:
            if current < n_batches - 1 and weight >= per_bucket * (current + 1) and buckets[current]:
                current += 1
            buckets[current].extend(i for i, g2 in enumerate(groups) if g2 == g)
            weight += counts[g]
    else:
        total = len(groups)
        step = max(1, total // n_batches)
        for i in range(n_batches):
            start = min(total, i * step)
            stop = total if i == n_batches - 1 else min(total, (i + 1) * step)
            buckets[i] = list(range(start, stop))
    return buckets


def _stratified_rows(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("answer")), []).append(row)
    left: list[dict] = []
    right: list[dict] = []
    for label, group in by_label.items():
        rng.shuffle(group)
        cut = int(round(len(group) * fraction))
        left.extend(group[:cut])
        right.extend(group[cut:])
    return left, right


def official_batches(config: TaskConfig, n_batches: int) -> list[Batch]:
    """Slice the official validation stream; cumulative labelled history."""

    train_rows = read_jsonl(config.train_path)
    val_rows = read_jsonl(config.validation_path)
    if len(val_rows) < n_batches:
        raise ValueError(
            f"validation stream for {config.key} has {len(val_rows)} rows, cannot make {n_batches} batches"
        )
    groups = _group_order(val_rows)
    buckets = _partition_indices(groups, n_batches)
    prev: list[dict] = []
    batches: list[Batch] = []
    for i, idxs in enumerate(buckets):
        batch_validation = [val_rows[j] for j in idxs]
        batch_train = list(train_rows) + list(prev)
        batches.append(
            Batch(
                index=i,
                train=batch_train,
                validation=batch_validation,
                meta={
                    "source": "official",
                    "validation_ids": [str(r.get("id")) for r in batch_validation],
                    "train_rows": len(batch_train),
                    "cumulative_prev_rows": len(prev),
                },
            )
        )
        prev.extend(batch_validation)
    return batches


def _uci_rows(dataset, classes: list[str], batch_no: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(len(dataset.ids)):
        label = int(dataset.y[i])
        if label < 0 or label >= len(classes):
            continue
        rows.append(
            {
                "id": f"uci-b{batch_no}-{i:06d}",
                "inputs": {"features": [float(v) for v in dataset.x[i].tolist()]},
                "answer": classes[label],
            }
        )
    return rows


def external_a_batches(
    config: TaskConfig,
    root: str | Path,
    n_batches: int | None = None,
    start: int = 3,
    end: int = 10,
    holdout: float = 0.3,
    seed: int = 42,
) -> list[Batch]:
    """UCI gas sensor drift, batches 3..10, as a Track X test-bed."""

    from ..data import read_uci_gas_dat
    from ..tasks import TASKS

    class_cfg = config.classes or TASKS["A"].classes
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"external data root not found: {root}")
    wanted = list(range(start, end + 1))
    if n_batches:
        wanted = wanted[:n_batches]
    paths = sorted(root.glob("batch*.dat"), key=lambda p: int(p.stem.replace("batch", "")))
    chosen = [p for p in paths if int(p.stem.replace("batch", "")) in wanted]
    if not chosen:
        raise FileNotFoundError(f"no batch*.dat files in {root}")
    train_rows = read_jsonl(config.train_path)
    batches: list[Batch] = []
    prev_train = list(train_rows)
    for i, path in enumerate(chosen):
        batch_no = int(path.stem.replace("batch", ""))
        dataset = read_uci_gas_dat(path)
        all_rows = _uci_rows(dataset, class_cfg, batch_no)
        if len(all_rows) < 8:
            continue
        train_part, eval_part = _stratified_rows(all_rows, 1.0 - holdout, seed + batch_no)
        batch_train = prev_train + train_part
        batches.append(
            Batch(
                index=i,
                train=batch_train,
                validation=eval_part,
                meta={
                    "source": "external-a",
                    "batch": batch_no,
                    "file": str(path),
                    "validation_rows": len(eval_part),
                    "train_rows": len(batch_train),
                },
            )
        )
        prev_train = batch_train
    return batches


def make_source(kind: str, config: TaskConfig, n_batches: int, **kwargs) -> list[Batch]:
    if kind == "official":
        return official_batches(config, n_batches)
    if kind == "external-a":
        root = kwargs.get("external_root")
        if not root:
            raise ValueError("external-a source requires --external-root")
        return external_a_batches(
            config,
            root,
            n_batches=n_batches,
            start=int(kwargs.get("external_start", 3)),
            end=int(kwargs.get("external_end", 10)),
            holdout=float(kwargs.get("holdout", 0.3)),
            seed=int(kwargs.get("seed", 42)),
        )
    raise ValueError(f"unknown stream source: {kind}")
