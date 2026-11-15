#!/usr/bin/env python3
"""02 — 基线复现：平台给的 checkpoint 在各数据划分上到底有多少分。

同时打印分类任务的混淆矩阵，以及回归任务逐设备/逐子集的表现，
为后面"基线本身是否合理"的判断提供事实。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import _common as C


def task_metrics(task: str) -> None:
    C.header(f"任务 {task} — {C.TASKS[task]['label']}")
    info = json.loads((C.task_dir(task) / "baseline-info.json").read_text(encoding="utf-8"))
    training = json.loads((C.task_dir(task) / "training.json").read_text(encoding="utf-8"))
    model = C.load_model(task)

    train_rows = C.read_split(task, "train")
    val_rows = C.read_split(task, "validation")
    xtr, ytr = C.matrix(train_rows), C.labels(train_rows, task)
    xva, yva = C.matrix(val_rows), C.labels(val_rows, task)

    train_metrics = C.evaluate(task, model, xtr, ytr)
    val_metrics = C.evaluate(task, model, xva, yva)

    print(f"  参数量 {sum(v.size for v in model.params.values())}  "
          f"hidden={model.params['w1'].shape[1]}  lr={model.meta['learning_rate']}  "
          f"steps={model.meta['steps']}")
    key = C.TASKS[task]["primary"]

    C.sub("复现 vs 平台记录（training.json 的 validation_before / validation_after）")
    before = training["validation_before"]
    after = training["validation_after"]
    print(f"  平台记录的 validation_before: {key} = {before.get(key)}")
    print(f"  平台记录的 validation_after : {key} = {after.get(key)}")
    print(f"  本仓库复算的 validation    : {key} = {val_metrics[key]:.10f}")
    print(f"  → {'✅ 与 validation_after 一致' if abs(val_metrics[key] - after[key]) < 1e-9 else '❌ 不一致'}")
    print(f"  参数量记录 {info['parameter_count']} / 实际 {sum(v.size for v in model.params.values())}"
          f"  {'✅' if info['parameter_count'] == sum(v.size for v in model.params.values()) else '❌'}")
    print(f"  label_status = {info['label_status']}   official_scoring_ready = {training['official_scoring_ready']}"
          f"   purpose = {training['purpose']}")

    if task == "A":
        C.sub("验证集混淆矩阵（行=真实，列=预测）")
        names = C.TASKS["A"]["classes"]
        matrix = np.zeros((6, 6), dtype=int)
        for truth, pred in zip(yva, val_metrics["pred"]):
            matrix[truth, pred] += 1
        print("  " + " " * 13 + "".join(f"{n[:9]:>11}" for n in names))
        for i, name in enumerate(names):
            print(f"  {name:<13}" + "".join(f"{v:>11}" for v in matrix[i]))
        C.sub("逐类 F1")
        for name, f1 in zip(names, val_metrics["per_class_f1"]):
            support = int((yva == names.index(name)).sum())
            print(f"  {name:<14} F1={f1:.4f}  验证样本数={support}")
        return

    C.sub("逐组 RMSE（验证集）")
    groups = defaultdict(list)
    for row, pred, truth in zip(val_rows, val_metrics["pred"], yva):
        groups[C.group_of(row, task)].append((float(pred), float(truth)))
    for name, pairs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        pred = np.array([p for p, _ in pairs])
        truth = np.array([t for _, t in pairs])
        print(f"  {name:<14} n={len(pairs):>6,}  RMSE={C.rmse(pred, truth):>9.3f}  "
              f"MAE={C.mae(pred, truth):>9.3f}  均值预测={pred.mean():>9.3f}  均值真值={truth.mean():>9.3f}")


def main() -> int:
    for task in "ABC":
        task_metrics(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
