"""Deterministic, auditable data splits and optional unlabeled drift probes.

The public validation file is the *only* labelled signal available before the
hidden evaluation. It is split once, deterministically, into three disjoint
roles and never re-split between rounds:

calibration
    held-out labels only. Never used for fitting: the competition rule forbids
    fitting validation data. (Regression shrink (if chosen) uses the checkpoint's
    frozen train-target mean, not these labels.)
selection
    the gate: candidate deployments are compared with the current parent on
    this split only.
report
    documentation metrics only; never used to accept or reject candidates.

For task A a class-stratified row split is used (the official validation file is
one UCI batch). For task C units are disjoint across splits. Task B has only
three validation bearings, so rows are split within bearings; the resulting
within-bearing leakage is documented in the run record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rsi.data import Dataset, load_jsonl_dataset, split_by_groups, write_uci_gas_jsonl
from rsi.tasks import TaskConfig, gas_probe_root


def sha256_file(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _subset(dataset: Dataset, name: str, index) -> Dataset:
    index = np.asarray(sorted(index), dtype=int)
    return Dataset(name, [dataset.ids[i] for i in index], dataset.x[index], dataset.y[index],
                   dataset.paths, dataset.sha256, [dataset.groups[i] for i in index])


def partition_rows(dataset: Dataset, fractions, seed: int, stratify: str = "label") -> list[Dataset]:
    """Split rows into len(fractions) parts, stratified by label or group."""
    if len(fractions) != 3:
        raise ValueError("this harness only builds three-way splits")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    if stratify == "group":
        keys = dataset.groups
    else:
        keys = [f"y{int(value)}" for value in dataset.y]
    parts = [[], [], []]
    for key in sorted(set(keys)):
        members = np.asarray([i for i, candidate in enumerate(keys) if candidate == key], dtype=int)
        rng.shuffle(members)
        n = len(members)
        cut1 = int(round(n * fractions[0]))
        cut2 = cut1 + int(round(n * fractions[1]))
        if n >= 3:
            cut1 = min(max(1, cut1), n - 2)
            cut2 = min(max(cut1 + 1, cut2), n - 1)
        elif n == 2:
            cut1, cut2 = 1, 1
        else:
            cut1, cut2 = 0, 1
        parts[0].extend(members[:cut1].tolist())
        parts[1].extend(members[cut1:cut2].tolist())
        parts[2].extend(members[cut2:].tolist())
    if any(not part for part in parts):
        raise ValueError("split produced an empty partition")
    return [_subset(dataset, name, part) for name, part in
            zip(("calibration", "selection", "report"), parts)]


def _write_subset(src_path: Path, ids, out_path: Path) -> str:
    """Copy the exact source JSONL lines for the selected ids (no re-serialisation)."""
    wanted = set(ids)
    kept = []
    with src_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") in wanted:
                kept.append(line.rstrip("\n"))
    if len(kept) != len(wanted):
        raise ValueError(f"split produced {len(kept)} rows for {len(wanted)} ids")
    out_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return sha256_file(out_path)


def _write_probe(config: TaskConfig, out_path: Path) -> Dataset | None:
    """Build the optional external UCI gas drift probe (development evidence only)."""
    root = gas_probe_root()
    wanted = config.probe == "uci_gas" or (config.probe == "auto" and config.key == "A")
    if not wanted or root is None:
        return None
    rows, labels, ids = [], [], []
    batch_files = sorted(Path(root).glob("batch*.dat"))
    for path in batch_files:
        batch = path.stem
        for number, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            parts = line.strip().split()
            label = int(parts[0]) - 1
            values = [0.0] * 128
            for item in parts[1:]:
                index, value = item.split(":")
                values[int(index) - 1] = float(value)
            rows.append(values)
            labels.append(label)
            ids.append(f"probe_{batch}_{number:06d}")
    if not rows:
        return None
    with out_path.open("w", encoding="utf-8") as handle:
        for identity, values, label in zip(ids, rows, labels):
            record = {"id": identity, "inputs": {"features": values},
                      "answer": config.classes[label]}
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return Dataset("drift_probe", ids, x, y, [], "sha256:multi",
                   [identity.rsplit("_", 1)[0] for identity in ids])


def build_splits(config: TaskConfig, state_dir: Path, seed: int = 42) -> dict:
    """Create the split JSONLs once and return paths plus in-memory datasets."""
    split_dir = Path(state_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train = load_jsonl_dataset(config.train_path, config.task, config.classes)
    validation = load_jsonl_dataset(config.validation_path, config.task, config.classes)
    strategy = config.split_strategy
    if strategy == "auto":
        unique_groups = len(set(validation.groups))
        if config.task == "classification":
            strategy = "stratified"
        elif unique_groups >= 10:
            strategy = "units"
        elif unique_groups >= 2:
            strategy = "group"
        else:
            strategy = "stratified"
    if strategy == "units":
        calibration, rest = split_by_groups(validation, 0.30, seed)
        selection, report = split_by_groups(rest, 0.50, seed + 1)
        parts = [calibration, selection, report]
    elif strategy in {"group", "bearing"}:
        parts = partition_rows(validation, (0.30, 0.35, 0.35), seed, stratify="group")
    else:
        parts = partition_rows(validation, (0.30, 0.35, 0.35), seed, stratify="label")

    paths = {"train": config.train_path}
    files = {}
    for name, dataset in zip(("calibration", "selection", "report"), parts):
        out_path = split_dir / f"{name}.jsonl"
        files[name] = _write_subset(config.validation_path, dataset.ids, out_path)
        paths[name] = out_path
        dataset.sha256 = files[name]

    extended = None
    extended_path = None
    extended_spec = getattr(config, "extended", None) or {}
    if extended_spec.get("batches"):
        from rsi.policy import extended_enabled
        if extended_enabled():
            root = gas_probe_root()
            if root is not None:
                extended_path = split_dir / "extended_train.jsonl"
                extended = write_uci_gas_jsonl(extended_spec.get("batches", [1, 3, 4, 5, 6, 7]),
                                               root, extended_path, config.classes)
                paths["extended"] = extended_path
                files["extended"] = sha256_file(extended_path)

    probe = None
    probe_path = None
    if config.probe == "uci_gas" or (config.probe == "auto" and config.key == "A"):
        probe_path = split_dir / "probe.jsonl"
        probe = _write_probe(config, probe_path)
        if probe is not None:
            paths["drift_probe"] = probe_path
            files["probe"] = sha256_file(probe_path)
        else:
            probe_path = None

    manifest = {
        "seed": seed,
        "train": {"path": str(paths["train"]), "rows": len(train), "sha256": sha256_file(paths["train"])},
        "validation": {"path": str(config.validation_path), "rows": len(validation),
                       "sha256": sha256_file(config.validation_path)},
        "splits": {name: {"rows": len(dataset), "sha256": files[name]}
                   for name, dataset in zip(("calibration", "selection", "report"), parts)},
        "probe": ({"rows": len(probe), "sha256": files["probe"]} if probe is not None else None),
        "extended": ({"rows": len(extended), "sha256": files["extended"], "batches": extended_spec.get("batches")}
                     if extended is not None else None),
        "protocol": {
            "split_strategy": strategy,
            "calibration": "held-out labels only; never fitted",
            "selection": "gate only; never used for fitting or reporting",
            "report": "documentation only; never used for accept/reject",
            "b_stratify": "rows split within bearings (3 validation bearings only)",
            "a_stratify": "class-stratified rows of the single validation batch",
            "c_stratify": "whole engine units (no unit crosses splits)",
            "portable": "strategy from task.json; auto = stratified for classification, units when >=10 groups, else within-group rows",
            "probe": "external UCI batch3-10, unlabelled for scoring; development evidence only",
            "extended": ("opt-in external pretraining data (RSI_ALLOW_EXTENDED_DATA=1); "
                         "never part of the official package"),
        },
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"paths": paths, "train": train, "validation": validation,
            "calibration": parts[0], "selection": parts[1], "report": parts[2],
            "probe": probe, "probe_path": probe_path,
            "extended": extended, "extended_path": extended_path, "manifest": manifest,
            "manifest_path": split_dir / "manifest.json", "split_dir": split_dir}
