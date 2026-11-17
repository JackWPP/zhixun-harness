"""Batch sources for honest streaming evaluation.

Every ``Batch`` follows the platform protocol:

* ``train``       labelled rows whose labels were revealed in *earlier* batches
* ``dev``         a labelled hold-out from earlier revealed batches, used for
                  model selection / postprocess fitting inside the harness
* ``prediction``  the current batch inputs (NO labels)
* ``answers``     current-batch labels, kept outside the harness package and
                  only used by the scoring routine after predictions are made
* ``meta``        phase (arrival|hidden), batch number, reveal policy, ...

Nothing in ``train``/``dev`` comes from the current or a hidden batch.  The
legacy eager sources are replaced by this interface because the old one leaked
current-batch labels into the Runner's selection split.
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
    """One batch under the honest protocol."""

    index: int
    train: list[dict]
    dev: list[dict]
    prediction: list[dict]
    answers: dict[str, str]
    reveal: list[dict] = field(default_factory=list)
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


def _strip(rows: Iterable[dict]) -> list[dict]:
    return [{"id": str(r.get("id", "")), "inputs": r.get("inputs", {})} for r in rows]


def _answers(rows: Iterable[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        out[str(row.get("id", ""))] = str(row.get("answer", ""))
    return out


def _shuffle(rows: list[dict], seed: int) -> list[dict]:
    """Deterministic shuffle that keeps every row (unlike a fraction split)."""

    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


def _strat(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified (by answer when present) split; fraction goes to the first part."""

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("answer", "")), []).append(row)
    first: list[dict] = []
    second: list[dict] = []
    for group in by_label.values():
        rng.shuffle(group)
        cut = int(round(len(group) * fraction))
        if len(group) >= 2:
            cut = max(1, min(len(group) - 1, cut))
        first.extend(group[:cut])
        second.extend(group[cut:])
    return first, second


def _add_history(state: dict, rows: list[dict], dev_fraction: float, seed: int) -> None:
    """Reveal labelled rows into the persistent train/dev pools (25% dev by default)."""

    if not rows:
        return
    # A "repeat" stream reveals the same batches again; rows must not enter the
    # history twice (the official loader rejects duplicate sample ids).
    seen = {str(row.get("id")) for row in state["train"]} | {
        str(row.get("id")) for row in state["dev"]}
    fresh = []
    for row in rows:
        sample_id = str(row.get("id"))
        if sample_id in seen:
            continue
        seen.add(sample_id)
        fresh.append(row)
    if not fresh:
        return
    dev_rows, train_rows = _strat(list(fresh), max(0.0, min(0.5, dev_fraction)), seed)
    state["train"].extend(train_rows)
    state["dev"].extend(dev_rows)


def _perturb(rows: list[dict], kind: str, seed: int) -> list[dict]:
    """Apply a sensor-style feature perturbation to a row list (label unchanged)."""

    if kind in (None, "", "none"):
        return rows
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(rows)
    if n == 0:
        return rows
    width = len((rows[0].get("inputs") or {}).get("features") or [])
    if width == 0:
        return rows
    gain = 1.0 + rng.normal(0.0, 0.05, size=(1, width))
    offset = rng.normal(0.0, 0.02, size=(1, width))
    out = []
    for row in rows:
        features = list((row.get("inputs") or {}).get("features") or [])
        vec = np.asarray(features, dtype=np.float64)
        if kind == "gain":
            vec = vec * gain.reshape(-1)
        elif kind == "offset":
            vec = vec + offset.reshape(-1)
        elif kind == "noise":
            vec = vec + rng.normal(0.0, 0.01, size=vec.shape)
        elif kind == "gain_offset":
            vec = vec * gain.reshape(-1) + offset.reshape(-1)
        else:
            raise ValueError(f"unknown perturb kind: {kind}")
        out.append({"id": row.get("id"), "inputs": {"features": [float(v) for v in vec]},
                    **({"answer": row["answer"]} if "answer" in row else {})})
    return out


def _missing(rows: list[dict], class_name: str | None) -> list[dict]:
    if not class_name:
        return rows
    return [r for r in rows if str(r.get("answer")) != class_name]


def _initial_pools(config: TaskConfig, dev_fraction: float, seed: int) -> dict:
    train_rows = read_jsonl(config.train_path)
    dev_rows, train_rows = _strat(train_rows, max(0.0, min(0.5, dev_fraction)), seed) if dev_fraction > 0 else ([], train_rows)
    return {"train": train_rows, "dev": dev_rows}


def official_batches(config: TaskConfig, n_batches: int, dev_fraction: float = 0.25,
                     seed: int = 42, hidden_tail: int = 0, **_kwargs) -> list[Batch]:
    """Official validation rows sliced into a stream; labels released after each batch."""

    train_rows = read_jsonl(config.train_path)
    val_rows = read_jsonl(config.validation_path)
    if len(val_rows) < n_batches:
        raise ValueError(
            f"validation stream for {config.key} has {len(val_rows)} rows, cannot make {n_batches} batches"
        )
    groups = _group_order(val_rows)
    buckets = _partition_indices(groups, n_batches)
    state = _initial_pools(config, dev_fraction, seed)
    batches: list[Batch] = []
    hidden_from = max(n_batches - max(0, int(hidden_tail)), 0)
    for i, idxs in enumerate(buckets):
        rows = [val_rows[j] for j in idxs]
        hidden = i >= hidden_from and hidden_tail > 0
        batches.append(Batch(
            index=i,
            train=list(state["train"]),
            dev=list(state["dev"]),
            prediction=_strip(rows),
            answers=_answers(rows),
            reveal=[] if hidden else list(rows),
            meta={"source": "official", "batch": i, "revealed": not hidden, "hidden": hidden,
                  "prediction_rows": len(rows), "train_rows": len(state["train"]),
                  "dev_rows": len(state["dev"])},
        ))
        if not hidden:
            _add_history(state, rows, dev_fraction, seed + i + 1)
    return batches


def _default_gas_root() -> Path:
    """Locate the vendored UCI gas .dat files when no explicit root is given."""

    import os

    candidates = []
    env = os.environ.get("RSI_GAS_ROOT", "")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parents[2]
    candidates.extend([here / "extra" / "uci-gas" / "Dataset", here / ".scratch" / "gas" / "Dataset"])
    for candidate in candidates:
        if candidate.is_dir() and list(candidate.glob("batch*.dat")):
            return candidate
    return candidates[0] if candidates else Path(".")


def _uci_rows(dataset, classes: list[str], batch_no: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(len(dataset.ids)):
        label = int(dataset.y[i])
        if label < 0 or label >= len(classes):
            continue
        rows.append({
            "id": f"uci-b{batch_no}-{i:06d}",
            "inputs": {"features": [float(v) for v in dataset.x[i].tolist()]},
            "answer": classes[label],
        })
    return rows


def private_sim_batches(
    config: TaskConfig,
    root: str | Path,
    arrival: tuple[int, int] = (3, 7),
    hidden: tuple[int, int] = (8, 10),
    dev_fraction: float = 0.25,
    label_every: int = 1,
    label_fraction: float = 1.0,
    perturb: str = "none",
    missing_class: str | None = None,
    repeat: int = 1,
    seed: int = 42,
    **_kwargs,
) -> list[Batch]:
    """UCI batches as a private-set simulator.

    Arrival batches are scored under the platform protocol (labels revealed only
    after predictions).  Hidden batches are never revealed to the harness; they
    are scored once at the end and never enter any history pool.
    """

    from ..data import read_uci_gas_dat

    class_cfg = list(config.classes or [])
    root = Path(root) if str(root or "").strip() else _default_gas_root()
    if not root.exists():
        raise FileNotFoundError(f"external data root not found: {root}")
    paths = sorted(root.glob("batch*.dat"), key=lambda p: int(p.stem.replace("batch", "")))
    if not paths:
        raise FileNotFoundError(f"no batch*.dat files under {root}")
    path_map = {int(p.stem.replace("batch", "")): p for p in paths}
    state = _initial_pools(config, dev_fraction, seed)
    batches: list[Batch] = []
    index = 0
    arrival_numbers = [b for b in range(int(arrival[0]), int(arrival[1]) + 1) if b in path_map]
    hidden_numbers = [b for b in range(int(hidden[0]), int(hidden[1]) + 1) if b in path_map]
    for round_no in range(max(1, int(repeat))):
        for pos, batch_no in enumerate(arrival_numbers):
            dataset = read_uci_gas_dat(path_map[batch_no])
            rows = _uci_rows(dataset, class_cfg, batch_no)
            if missing_class:
                rows = _missing(rows, missing_class)
            if not rows:
                continue
            rows = _perturb(rows, perturb, seed + batch_no)
            rows = _shuffle(rows, seed)  # keep every row
            revealed = (label_fraction > 0) and ((pos + 1) % max(1, int(label_every)) == 0)
            reveal_rows: list[dict] = []
            if revealed and label_fraction > 0:
                if float(label_fraction) >= 1.0:
                    reveal_rows = list(rows)
                else:
                    selected = _strat(rows, float(label_fraction), seed + 100 + batch_no)[0]
                    reveal_rows = selected or rows[:1]
            batches.append(Batch(
                index=index,
                train=list(state["train"]),
                dev=list(state["dev"]),
                prediction=_strip(rows),
                answers=_answers(rows),
                reveal=list(reveal_rows),
                meta={"source": "private-sim", "phase": "arrival", "batch": batch_no,
                      "round": round_no, "revealed": bool(revealed), "hidden": False,
                      "prediction_rows": len(rows), "reveal_rows": len(reveal_rows),
                      "train_rows": len(state["train"]), "dev_rows": len(state["dev"]),
                      "label_every": label_every, "label_fraction": label_fraction,
                      "perturb": perturb, "missing_class": missing_class},
            ))
            if reveal_rows:
                _add_history(state, reveal_rows, dev_fraction, seed + 200 + batch_no)
            index += 1
    for batch_no in hidden_numbers:
        dataset = read_uci_gas_dat(path_map[batch_no])
        rows = _uci_rows(dataset, class_cfg, batch_no)
        if missing_class:
            rows = _missing(rows, missing_class)
        if not rows:
            continue
        rows = _perturb(rows, perturb, seed + batch_no)
        batches.append(Batch(
            index=index,
            train=list(state["train"]),
            dev=list(state["dev"]),
            prediction=_strip(rows),
            answers=_answers(rows),
            meta={"source": "private-sim", "phase": "hidden", "batch": batch_no,
                  "revealed": False, "hidden": True, "prediction_rows": len(rows),
                  "train_rows": len(state["train"]), "dev_rows": len(state["dev"]),
                  "perturb": perturb, "missing_class": missing_class},
        ))
        index += 1
    return batches


def full_batches(
    config: TaskConfig,
    n_batches: int = 10,
    dev_fraction: float = 0.25,
    seed: int = 42,
    hidden_tail: int = 2,
    perturb: str = "none",
    label_every: int = 1,
    label_fraction: float = 1.0,
    repeat: int = 1,
    **_kwargs,
) -> list[Batch]:
    """Whole-dataset batch stream used by the full-scale rehearsal.

    * A: every UCI gas batch 1..N (the official package only contains b1/b2).
    * B/C: every official row, split by bearing / engine-unit groups.

    The current batch labels are never placed into ``train``/``dev``; labels are
    revealed only after the batch has been predicted (``reveal``) and hidden
    batches are never revealed.
    """

    if config.key == "A":
        n = max(2, int(n_batches))
        return private_sim_batches(
            config,
            "",
            arrival=(1, n),
            hidden=(n + 1, n + 1),
            dev_fraction=dev_fraction,
            label_every=label_every,
            label_fraction=label_fraction,
            perturb=perturb,
            repeat=repeat,
            seed=seed,
        )

    batches = official_batches(
        config, n_batches, dev_fraction=dev_fraction, seed=seed,
        hidden_tail=hidden_tail,
    )
    if perturb not in (None, "", "none"):
        for batch in batches:
            batch.prediction = _perturb(batch.prediction, perturb, seed + 1000 * batch.index + 1)
            batch.reveal = _perturb(batch.reveal, perturb, seed + 1000 * batch.index + 2)
            batch.meta["perturb"] = perturb
    return batches


def make_source(kind: str, config: TaskConfig, n_batches: int, **kwargs) -> list[Batch]:
    """Build a batch list for one of the supported sources."""

    if kind in ("full", "full-scale"):
        return full_batches(config, n_batches, **kwargs)
    if kind == "official":
        return official_batches(config, n_batches, **kwargs)
    if kind == "private-sim":
        root = kwargs.pop("external_root", None) or ""
        if config.key != "A" or (root and not Path(root).is_dir()):
            return official_batches(config, n_batches, hidden_tail=2, **kwargs)
        return private_sim_batches(config, root, **kwargs)
    if kind == "external-a":
        # Track-X alias: replay UCI batches as an arrival stream with hidden tail.
        start = int(kwargs.pop("external_start", 3))
        end = int(kwargs.pop("external_end", 10))
        hidden_tail = max(0, int(kwargs.pop("hidden_tail", 2)))
        arrival_end = max(start, end - hidden_tail)
        return private_sim_batches(
            config,
            kwargs.pop("external_root", None) or "",
            arrival=(start, arrival_end),
            hidden=(arrival_end + 1, end),
            **kwargs,
        )
    raise ValueError(f"unknown batch source: {kind}")
