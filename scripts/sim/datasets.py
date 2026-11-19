#!/usr/bin/env python3
"""Dataset packages for the full RSI simulation (official + external/private).

Builds three kinds of contest.harness.workspace.v2 packages:

* S1  official  -- the three shipped packages under ``datasets/`` (unchanged).
* S2  external  -- UCI gas sensor-array drift archive (batch1..10) repackaged
  as a classification task with the same 6 gas classes; batches 1-2 are the
  initial public projection, batches 3-7 arrive one per round, batches 8-10
  stay hidden until final scoring.
* S3  schema    -- an official-A checkpoint whose feature schema/keys are
  renamed ("features" -> "probe", addresses prefixed), used to exercise the
  private-schema adaptation path.

The generated packages are simulation fixtures only: they live under
``.scratch/sim/datasets/`` and are never bundled into the competition image.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rsi.data import read_uci_gas_dat  # noqa: E402
from rsi.model import ExtendedModel  # noqa: E402

GAS_BATCHES = 10
GAS_TRAIN_BATCHES = (1, 2)
GAS_DROP_BATCHES = (3, 4, 5, 6, 7)
GAS_HIDDEN_BATCHES = (8, 9, 10)
GAS_CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def _gas_rows(batch: int) -> list[dict]:
    dataset = read_uci_gas_dat(REPO / "extra" / "uci-gas" / "Dataset" / f"batch{batch}.dat")
    rows = []
    for index, row in enumerate(dataset.x):
        label = int(dataset.y[index])
        rows.append({
            "id": f"gas_b{batch:02d}_{index:06d}",
            "inputs": {"features": [float(value) for value in row]},
            "answer": GAS_CLASSES[label],
            "meta": {"batch": batch, "source": "uci-gas"},
        })
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def build_gas_package(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    train_rows = []
    for batch in GAS_TRAIN_BATCHES:
        train_rows.extend(_gas_rows(batch))
    x = np.asarray([row["inputs"]["features"] for row in train_rows], dtype=np.float64)
    y = np.asarray([GAS_CLASSES.index(row["answer"]) for row in train_rows], dtype=np.int64)
    paths = [json.dumps([["key", "features"], ["index", index]], separators=(",", ":"))
             for index in range(x.shape[1])]
    model = ExtendedModel("classification", paths, GAS_CLASSES, x, y, hidden=32, seed=42,
                          learning_rate=0.003, batch_size=128)
    model.fit(x, y, 200)
    model.save(root / "checkpoint.npz")
    _write_jsonl(root / "public" / "train.jsonl", train_rows)
    _write_jsonl(root / "public" / "validation.jsonl", _gas_rows(GAS_DROP_BATCHES[0]))
    for batch in GAS_DROP_BATCHES[1:]:
        _write_jsonl(root / "drops" / f"batch{batch}.jsonl", _gas_rows(batch))
    for batch in GAS_HIDDEN_BATCHES:
        _write_jsonl(root / "hidden" / f"batch{batch}.jsonl", _gas_rows(batch))
    meta = {
        "package": "sim-gas-ext-v1",
        "task": "classification",
        "metric": "macro_f1",
        "direction": "higher",
        "classes": GAS_CLASSES,
        "initial_batches": list(GAS_TRAIN_BATCHES),
        "drop_batches": list(GAS_DROP_BATCHES[1:]),
        "hidden_batches": list(GAS_HIDDEN_BATCHES),
        "features": x.shape[1],
        "public_train_rows": len(train_rows),
    }
    (root / "sim_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"root": root, **meta, "checkpoint": root / "checkpoint.npz"}


def _rename_inputs(row: dict, prefix: str) -> dict:
    features = row["inputs"].get("features")
    if features is None:
        return row
    renamed = {f"{prefix}_{index:04d}": value for index, value in enumerate(features)}
    return {"id": row.get("id"), "inputs": {"probe": renamed}, "answer": row.get("answer"),
            "meta": row.get("meta")}


def build_schema_package(root: Path) -> dict:
    source = REPO / "datasets" / "cpu-a-guide-20261110-v1"
    root.mkdir(parents=True, exist_ok=True)
    model = ExtendedModel.from_checkpoint(source / "checkpoint.npz")
    specs = []
    for entry in model.meta["paths"]:
        spec = json.loads(entry) if isinstance(entry, str) else entry
        address = spec[-1][-1]
        specs.append([["key", "probe"], ["key", f"sensor_{int(address):04d}"]])
    meta = dict(model.meta)
    meta["paths"] = [json.dumps(spec, separators=(",", ":")) for spec in specs]
    meta["hidden"] = meta.get("hidden") or {}
    variant = ExtendedModel("classification", meta["paths"], list(meta["classes"]),
                            np.zeros((1, len(specs))), np.zeros(1, dtype=np.int64),
                            hidden=32)
    variant.meta = meta
    variant.params = {key: value.copy() for key, value in model.params.items()}
    variant.first = {key: value.copy() for key, value in model.first.items()}
    variant.second = {key: value.copy() for key, value in model.second.items()}
    variant.mean = model.mean.copy()
    variant.scale = model.scale.copy()
    variant.save(root / "checkpoint.npz")

    def rows(path: Path, limit: int | None = None) -> list[dict]:
        output = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and len(output) >= limit:
                    break
                output.append(_rename_inputs(json.loads(line), "sensor"))
        return output

    train = rows(source / "public" / "train.jsonl")
    validation = rows(source / "public" / "validation.jsonl")
    # Stratified split: A validation rows are grouped by class, so a naive
    # "second half" hidden split would contain only 2 classes and make the
    # fixed-6 macro-F1 ceiling 2/6. Split per class round-robin instead.
    by_class: dict[str, list[dict]] = {}
    for row in validation:
        by_class.setdefault(str(row.get("answer")), []).append(row)
    public_rows: list[dict] = []
    hidden_rows: list[dict] = []
    for label in sorted(by_class):
        group = by_class[label]
        for index, row in enumerate(group):
            (public_rows if index % 2 == 0 else hidden_rows).append(row)
    half = len(public_rows)
    _write_jsonl(root / "public" / "train.jsonl", train)
    _write_jsonl(root / "public" / "validation.jsonl", public_rows)
    _write_jsonl(root / "hidden" / "private_schema.jsonl", hidden_rows)
    meta_out = {
        "package": "sim-private-schema-v1",
        "task": "classification",
        "metric": "macro_f1",
        "direction": "higher",
        "classes": list(meta["classes"]),
        "public_train_rows": len(train),
        "public_validation_rows": half,
        "hidden_rows": len(validation) - half,
        "note": "official-A weights, renamed input schema; hidden split = second half of A validation",
    }
    (root / "sim_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"root": root, **meta_out, "checkpoint": root / "checkpoint.npz"}


def build_a_stream_package(root: Path) -> dict:
    """Official-A feature that keeps the *official* package untouched.

    ``public/train.jsonl`` is UCI batch1, ``public/validation.jsonl`` is batch2
    (the contest's public projection), and batches 3..10 are stored as ordered
    stream drops.  A faithful official-sequence host reveals them one per round:
    round r trains on batch1 + drops[0..r-2] and predicts drops[r-1].
    """
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "public" / "train.jsonl", _gas_rows(1))
    _write_jsonl(root / "public" / "validation.jsonl", _gas_rows(2))
    for batch in range(3, GAS_BATCHES + 1):
        _write_jsonl(root / "stream" / f"b{batch:02d}.jsonl", _gas_rows(batch))
    meta = {
        "package": "sim-a-official-seq-v1",
        "task": "classification",
        "metric": "macro_f1",
        "direction": "higher",
        "classes": GAS_CLASSES,
        "public_train_batch": 1,
        "public_validation_batch": 2,
        "stream_batches": list(range(3, GAS_BATCHES + 1)),
        "note": "faithful cumulative-reveal protocol for the official A task",
    }
    (root / "sim_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"root": root, **meta}


def ensure_packages(base: Path | None = None) -> dict:
    base = base or (REPO / ".scratch" / "sim" / "datasets")
    gas_root = base / "sim-gas-ext-v1"
    schema_root = base / "sim-private-schema-v1"
    stream_root = base / "sim-a-official-seq-v1"
    gas = build_gas_package(gas_root) if not (gas_root / "checkpoint.npz").is_file() else {
        "root": gas_root, "checkpoint": gas_root / "checkpoint.npz",
        **json.loads((gas_root / "sim_meta.json").read_text(encoding="utf-8"))}
    schema = build_schema_package(schema_root) if not (schema_root / "checkpoint.npz").is_file() else {
        "root": schema_root, "checkpoint": schema_root / "checkpoint.npz",
        **json.loads((schema_root / "sim_meta.json").read_text(encoding="utf-8"))}
    stream = build_a_stream_package(stream_root) if not (stream_root / "sim_meta.json").is_file() else {
        "root": stream_root,
        **json.loads((stream_root / "sim_meta.json").read_text(encoding="utf-8"))}
    return {"gas": gas, "schema": schema, "a_stream": stream}


if __name__ == "__main__":
    built = ensure_packages()
    for key, value in built.items():
        print(key, value["root"], "task=", value.get("task"), "metric=", value.get("metric"))
