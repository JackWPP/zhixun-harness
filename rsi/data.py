"""Dataset adapters: official JSONL packages plus optional unlabeled drift probes."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rsi.data_official import flatten, json_value, read_dataset, sha256


@dataclass
class Dataset:
    name: str
    ids: list[str]
    x: np.ndarray
    y: np.ndarray
    paths: list[str]
    sha256: str
    groups: list[str] = field(default_factory=list)

    def __len__(self):
        return len(self.ids)


def _derive_groups(ids):
    """Best-effort grouping key used for honest, leakage-free evaluation.

    A: UCI gas sample ids are batchN-<n>; consecutive batches are the natural
    drift phases, but the official validation file is one batch, so fall back to
    a stable hash for stratified splits.
    B: xjtu_Bearing<unit>_<sub>_sNNNNNN -> bearing unit (held-out bearings).
    C: fd001_train_u0001_c00001 -> engine unit.
    """
    groups = []
    for identity in ids:
        match = re.match(r"xjtu_Bearing(\d+_\d+)_", identity)
        if match:
            groups.append("bearing_" + match.group(1))
            continue
        match = re.match(r"(fd\d+)_[a-z]+_u(\d+)_", identity)
        if match:
            groups.append(f"{match.group(1)}_u{match.group(2)}")
            continue
        match = re.match(r"([A-Za-z]+)-?batch(\d+)", identity) or re.match(r"batch(\d+)", identity)
        if match:
            groups.append("batch_" + match.groups()[-1])
            continue
        groups.append("row")
    return groups


def load_jsonl_dataset(path, task, classes=None, expected_paths=None) -> Dataset:
    path = Path(path)
    data = read_dataset(path, task, classes, expected_paths)
    return Dataset(path.stem, data["ids"], data["x"], data["y"], data["paths"], data["sha256"],
                   _derive_groups(data["ids"]))


def split_by_groups(dataset: Dataset, fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    """Split so that no group appears on both sides (honest selection/report split)."""
    rng = np.random.default_rng(seed)
    groups = sorted(set(dataset.groups))
    rng.shuffle(groups)
    selected, remaining, taken = [], 0, set()
    target = int(round(len(dataset) * fraction))
    for group in groups:
        if taken and selected and taken_size(selected) >= target:
            break
        selected.extend(i for i, g in enumerate(dataset.groups) if g == group)
        taken.add(group)
    selected_set = set(selected)
    part_a = [i for i in range(len(dataset)) if i in selected_set]
    part_b = [i for i in range(len(dataset)) if i not in selected_set]
    # Ensure both sides are non-empty even for tiny datasets.
    if not part_a or not part_b:
        index = rng.permutation(len(dataset))
        cut = max(1, min(len(dataset) - 1, target))
        part_a, part_b = index[:cut].tolist(), index[cut:].tolist()
    return _subset(dataset, "selection", part_a), _subset(dataset, "report", part_b)


def taken_size(ids):
    return len(ids)


def _subset(dataset: Dataset, name: str, index):
    index = np.asarray(sorted(index), dtype=int)
    return Dataset(name, [dataset.ids[i] for i in index], dataset.x[index], dataset.y[index],
                   dataset.paths, dataset.sha256, [dataset.groups[i] for i in index])


def stratified_split(dataset: Dataset, fraction: float, seed: int, classes):
    """Class-aware split for classification tasks without natural groups."""
    rng = np.random.default_rng(seed)
    index = np.arange(len(dataset))
    selection, report = [], []
    for label in range(len(classes)):
        members = index[dataset.y == label]
        if len(members) == 0:
            continue
        rng.shuffle(members)
        cut = int(round(len(members) * fraction))
        if len(members) >= 2:
            cut = min(max(1, cut), len(members) - 1)
        selection.extend(members[:cut].tolist())
        report.extend(members[cut:].tolist())
    if not selection or not report:
        return split_by_groups(dataset, fraction, seed)
    return _subset(dataset, "selection", selection), _subset(dataset, "report", report)


def read_uci_gas_dat(path) -> Dataset:
    """Unlabeled-less loader for the public UCI gas sensor array drift archive.

    Each row starts with a class id (1..6) followed by 128 "sensor:value" fields,
    matching the feature schema of official task A. Used only as an optional,
    non-scored drift probe; never for fitting the submitted model.
    """
    rows, labels = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        labels.append(int(parts[0]) - 1)
        values = [0.0] * 128
        for item in parts[1:]:
            key, value = item.split(":")
            values[int(key) - 1] = float(value)
        rows.append(values)
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    batch = Path(path).stem
    return Dataset(batch, [f"{batch}-{i:06d}" for i in range(len(x))], x, y,
                   [json.dumps([["index", i]], separators=(",", ":")) for i in range(128)],
                   "sha256:" + sha256(Path(path).read_bytes()).split(":", 1)[1], [batch] * len(x))


def load_drift_probe(task_key: str, root: Path | None) -> Dataset | None:
    if not root or task_key != "A":
        return None
    root = Path(root)
    files = sorted(root.glob("batch*.dat"))
    if not files:
        return None
    parts = [read_uci_gas_dat(path) for path in files]
    ids, x, y, groups = [], [], [], []
    for part in parts:
        ids.extend(part.ids)
        x.append(part.x)
        y.append(part.y)
        groups.extend(part.groups)
    paths = parts[0].paths
    return Dataset("drift_probe", ids, np.vstack(x), np.concatenate(y), paths, "sha256:multi", groups)


def write_uci_gas_jsonl(batches, root, out_path, classes) -> Dataset:
    """Materialise selected UCI gas batches as official-schema JSONL (opt-in)."""
    root = Path(root)
    rows, labels, ids, groups = [], [], [], []
    for batch in batches:
        path = root / f"batch{batch}.dat"
        if not path.is_file():
            raise FileNotFoundError(f"extended gas batch missing: {path}")
        part = read_uci_gas_dat(path)
        y = [int(value) for value in part.y]
        for index, (vector, label) in enumerate(zip(part.x, y)):
            rows.append(vector)
            labels.append(label)
            ids.append(f"ext_{path.stem}_{index:06d}")
            groups.append(path.stem)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for identity, vector, label in zip(ids, rows, labels):
            record = {"id": identity, "inputs": {"features": [round(float(v), 6) for v in vector]},
                      "answer": classes[label]}
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return Dataset("extended_train", ids, x, y, [], "sha256:multi", groups)
