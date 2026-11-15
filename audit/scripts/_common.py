"""审计脚本共享工具。

约定：
  * 所有路径相对仓库根目录解析，脚本可在任意 cwd 下运行。
  * 厂商（平台）代码以文件路径方式加载，避免三个包里同名的 `baselines` 包互相覆盖。
  * 只用 numpy + 标准库，与 harness 保持一致。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
ARCHIVES = ROOT / "archives"
DEMO = ROOT / "demo"

TASKS = {
    "A": {
        "dirname": "cpu-a-guide-20261110-v1",
        "archive": "cpu-a-guide-20261110-v1.zip",
        "label": "气体传感器漂移 · 六类识别",
        "primary": "macro_f1",
        "classes": ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
    },
    "B": {
        "dirname": "cpu-b-rul-20261110-v1",
        "archive": "cpu-b-rul-20261110-v1.zip",
        "label": "滚动轴承 · 剩余寿命（分钟）",
        "primary": "rmse",
        "classes": None,
    },
    "C": {
        "dirname": "cpu-c-guide-20261110-v1",
        "archive": "cpu-c-guide-20261110-v1.zip",
        "label": "涡扇发动机 · 剩余寿命（循环）",
        "primary": "rmse",
        "classes": None,
    },
}


# --------------------------------------------------------------------------
# 厂商代码加载
# --------------------------------------------------------------------------
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_VENDOR: dict = {}


def vendor():
    """返回 (model 模块, data 模块)。三个包的 model.py / data.py 字节相同。"""
    if not _VENDOR:
        base = task_dir("A") / "baselines"
        _VENDOR["model"] = _load_module(base / "model.py", "_vendor_model")
        _VENDOR["data"] = _load_module(base / "data.py", "_vendor_data")
    return _VENDOR["model"], _VENDOR["data"]


# --------------------------------------------------------------------------
# 数据访问
# --------------------------------------------------------------------------
def task_dir(task: str) -> Path:
    return DATASETS / TASKS[task]["dirname"]


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_split(task: str, split: str) -> list:
    return load_jsonl(task_dir(task) / "public" / f"{split}.jsonl")


def matrix(rows: list) -> np.ndarray:
    _, data = vendor()
    return np.array([list(data.flatten(row["inputs"])[1]) for row in rows], dtype=float)


def feature_paths(rows: list) -> list:
    _, data = vendor()
    return list(data.flatten(rows[0]["inputs"])[0])


def labels(rows: list, task: str) -> np.ndarray:
    if task == "A":
        index = {c: i for i, c in enumerate(TASKS["A"]["classes"])}
        return np.array([index[row["answer"]] for row in rows], dtype=np.int64)
    return np.array([float(row["answer"]) for row in rows], dtype=float)


def group_of(row: dict, task: str) -> str:
    """任务内的“设备/批次”分组键，用于看跨组泛化。"""
    meta = row.get("meta") or {}
    for key in ("bearing", "source_batch", "source_subset", "unit"):
        if key in meta:
            return str(meta[key])
    return "all"


def load_model(task: str, name: str = "checkpoint.npz"):
    model_module, _ = vendor()
    return model_module.Model.load(task_dir(task) / name)


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def macro_f1(pred: np.ndarray, truth: np.ndarray, n_classes: int):
    """返回 (macro_f1, 各类 F1 列表)。与平台定义一致：tp 为 0 的类记 0 分。"""
    per_class = []
    for k in range(n_classes):
        tp = int(((pred == k) & (truth == k)).sum())
        fp = int(((pred == k) & (truth != k)).sum())
        fn = int(((pred != k) & (truth == k)).sum())
        per_class.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(sum(per_class) / n_classes), per_class


def rmse(pred, truth) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float)) ** 2)))


def mae(pred, truth) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - np.asarray(truth, dtype=float))))


def evaluate(task: str, model, x: np.ndarray, y: np.ndarray):
    """统一评估入口，返回与 harness 相同口径的指标字典。"""
    if task == "A":
        proba = model.predict(x)
        pred = proba.argmax(axis=1)
        f1, per_class = macro_f1(pred, y, 6)
        return {"macro_f1": f1, "accuracy": float((pred == y).mean()), "per_class_f1": per_class, "pred": pred}
    pred = model.predict(x)
    return {"rmse": rmse(pred, y), "mae": mae(pred, y), "pred": pred}


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def sub(title: str) -> None:
    print(f"\n-- {title} " + "-" * max(0, 74 - len(title)))


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"
