"""Task definitions for the three official contest packages.

All paths are resolved from ``RSI_DATA_ROOT`` when set (used inside the
evaluation image) and otherwise from the repository's ``datasets/`` directory.
Hidden labels are never referenced anywhere in this file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
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

    @property
    def package_dir(self) -> Path:
        root = Path(os.environ.get("RSI_DATA_ROOT", REPO_ROOT / "datasets"))
        return root / self.package

    @property
    def checkpoint_path(self) -> Path:
        return self.package_dir / "checkpoint.npz"

    @property
    def train_path(self) -> Path:
        return self.package_dir / "public" / "train.jsonl"

    @property
    def validation_path(self) -> Path:
        return self.package_dir / "public" / "validation.jsonl"


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
