"""Task definitions for the three official contest packages.

All paths are resolved from ``RSI_DATA_ROOT`` when set (used inside the
evaluation image) and otherwise from the repository's ``datasets/`` directory.
Hidden labels are never referenced anywhere in this file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TaskConfig:
    key: str
    name: str
    name_en: str
    task: str
    package: str
    classes: tuple
    primary_metric: str
    direction: str
    baseline_primary: float
    margin: float
    rounds: int = 10
    # Portable-runtime fields: a task is fully described by its task.json, so a
    # new dataset/package can be onboarded without touching this module.
    data_root: str = ""
    train_file: str = "public/train.jsonl"
    validation_file: str = "public/validation.jsonl"
    checkpoint_file: str = "checkpoint.npz"
    split_strategy: str = "auto"   # auto | stratified | group | units
    probe: str = "auto"            # auto | uci_gas | none
    extended: dict = field(default_factory=dict)  # opt-in external pretrain source
    description: str = ""

    @property
    def package_dir(self) -> Path:
        if self.package:
            package = Path(self.package)
            if package.is_absolute():
                return package
        else:
            package = Path(".")
        if self.data_root:
            root = Path(self.data_root)
            if not root.is_absolute():
                root = (REPO_ROOT / root)
            return root / package
        root = Path(os.environ.get("RSI_DATA_ROOT", REPO_ROOT / "datasets"))
        return root / package

    @property
    def checkpoint_path(self) -> Path:
        return self.package_dir / self.checkpoint_file

    @property
    def train_path(self) -> Path:
        return self.package_dir / self.train_file

    @property
    def validation_path(self) -> Path:
        return self.package_dir / self.validation_file


TASKS = {
    "A": TaskConfig(
        key="A",
        name="气体传感器漂移分类",
        name_en="Gas Sensor Array Drift Classification",
        task="classification",
        package="cpu-a-guide-20261110-v1",
        classes=("Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"),
        primary_metric="macro_f1",
        direction="higher",
        baseline_primary=0.7140297220788709,
        margin=0.004,
    ),
    "B": TaskConfig(
        key="B",
        name="滚动轴承剩余寿命预测",
        name_en="Bearing Remaining Useful Life",
        task="regression",
        package="cpu-b-rul-20261110-v1",
        classes=(),
        primary_metric="rmse",
        direction="lower",
        baseline_primary=681.1598326043113,
        margin=5.0,
    ),
    "C": TaskConfig(
        key="C",
        name="涡扇发动机剩余寿命预测",
        name_en="Turbofan Engine RUL",
        task="regression",
        package="cpu-c-guide-20261110-v1",
        classes=(),
        primary_metric="rmse",
        direction="lower",
        baseline_primary=52.86857256611151,
        margin=0.3,
    ),
}


def gas_probe_root() -> Path | None:
    """Optional external UCI gas archive used only as a development drift probe."""
    raw = os.environ.get("RSI_GAS_ROOT")
    candidates = [Path(raw) if raw else None, REPO_ROOT / ".scratch" / "gas" / "Dataset"]
    for candidate in candidates:
        if candidate and candidate.is_dir() and list(candidate.glob("batch*.dat")):
            return candidate
    return None


def resolve_tasks(keys):
    result = []
    for key in keys:
        key = key.strip().upper()
        if key not in TASKS:
            raise SystemExit(f"unknown task {key!r}; choose from {sorted(TASKS)}")
        result.append(TASKS[key])
    return result


# ---------------------------------------------------------------------------
# Portable task specs
# ---------------------------------------------------------------------------
def _spec_paths(raw: str | None = None) -> list[Path]:
    """Resolve task-spec sources: explicit paths, RSI_TASK_SPECS, repo tasks/."""
    candidates: list[Path] = []
    if raw:
        parts = raw if isinstance(raw, (list, tuple)) else raw.split(os.pathsep)
        candidates.extend(Path(part) for part in parts if part)
    env = os.environ.get("RSI_TASK_SPECS")
    if env:
        candidates.extend(Path(part) for part in env.split(os.pathsep) if part)
    repo_tasks = REPO_ROOT / "tasks"
    if repo_tasks.is_dir():
        candidates.append(repo_tasks)
    found: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            found.extend(sorted(candidate.glob("*.json")))
            found.extend(sorted(candidate.glob("*/task.json")))
        elif candidate.is_file():
            found.append(candidate)
    seen, unique = set(), []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def load_task_spec(path) -> TaskConfig:
    """Load a single portable ``task.json`` (paths resolve relative to the spec)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "key" not in data:
        raise SystemExit(f"task spec {path} is missing 'key'")
    data_root = data.get("data_root", ".")
    if not Path(data_root).is_absolute():
        data_root = str((path.parent / data_root).resolve())
    classes = tuple(data.get("classes") or ())
    if data.get("task") == "classification" and not classes:
        raise SystemExit(f"classification spec {path} needs non-empty 'classes'")
    return TaskConfig(
        key=str(data["key"]).upper(),
        name=data.get("name", data["key"]),
        name_en=data.get("name_en", data.get("name", data["key"])),
        task=data.get("task", "classification"),
        package=data.get("package", "."),
        classes=classes,
        primary_metric=data.get("primary_metric", "macro_f1"),
        direction=data.get("direction", "higher"),
        baseline_primary=float(data.get("baseline_primary", 0.0)),
        margin=float(data.get("margin", 0.0)),
        rounds=int(data.get("rounds", 10)),
        data_root=data_root,
        train_file=data.get("train_file", "public/train.jsonl"),
        validation_file=data.get("validation_file", "public/validation.jsonl"),
        checkpoint_file=data.get("checkpoint_file", "checkpoint.npz"),
        split_strategy=data.get("split_strategy", "auto"),
        probe=data.get("probe", "auto"),
        extended=dict(data.get("extended") or {}),
        description=data.get("description", ""),
    )


def register_task_specs(spec: str | None = None, *, replace: bool = True) -> list[str]:
    """Register every discoverable task spec into ``TASKS``; returns keys added."""
    added = []
    for path in _spec_paths(spec):
        try:
            config = load_task_spec(path)
        except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - diagnostics
            raise SystemExit(f"invalid task spec {path}: {exc}") from exc
        if config.key in TASKS and not replace:
            continue
        TASKS[config.key] = config
        added.append(config.key)
    return added


def describe_tasks() -> str:
    lines = []
    for key in sorted(TASKS):
        cfg = TASKS[key]
        lines.append(f"{key}: {cfg.name} [{cfg.task}/{cfg.primary_metric} {cfg.direction}] "
                     f"package={cfg.package_dir}")
    return "\n".join(lines)
