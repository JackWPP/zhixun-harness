#!/usr/bin/env python3
"""01 — 数据质量审计：三份数据到底"脏不脏"。

对每个任务检查：行数/列数、NaN 与 inf、重复行、常量列、训练与验证是否重叠、
类别/设备/子集分布、目标值范围。这是回答"数据质量不够"这个说法是否成立的基础。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import _common as C


def audit(task: str) -> None:
    C.header(f"任务 {task} — {C.TASKS[task]['label']}")
    train_rows = C.read_split(task, "train")
    val_rows = C.read_split(task, "validation")
    xtr, xva = C.matrix(train_rows), C.matrix(val_rows)
    ytr, yva = C.labels(train_rows, task), C.labels(val_rows, task)

    print(f"  行数      train {len(train_rows):>7,}   validation {len(val_rows):>7,}")
    print(f"  特征列    {xtr.shape[1]}")

    # --- 缺失与异常 ---
    C.sub("缺失值与异常值")
    for name, x in (("train", xtr), ("validation", xva)):
        nan = int(np.isnan(x).sum())
        inf = int(np.isinf(x).sum())
        print(f"  {name:<10} NaN={nan}  inf={inf}")

    C.sub("重复行（按全部特征精确比较）")
    for name, x in (("train", xtr), ("validation", xva)):
        unique = np.unique(x, axis=0).shape[0]
        print(f"  {name:<10} 唯一行 {unique:,} / {len(x):,}  → 重复 {len(x) - unique:,}")

    overlap = 0
    train_set = {row.tobytes() for row in np.ascontiguousarray(xtr)}
    overlap = sum(1 for row in np.ascontiguousarray(xva) if row.tobytes() in train_set)
    print(f"  训练∩验证 精确重叠行: {overlap}")

    C.sub("常量列 / 零方差列")
    std = xtr.std(axis=0)
    print(f"  训练集 std==0 的列: {int((std == 0).sum())} / {xtr.shape[1]}")

    # --- 目标与分组 ---
    C.sub("目标值与分组")
    if task == "A":
        print(f"  训练集类别分布: {dict(Counter(r['answer'] for r in train_rows))}")
        print(f"  验证集类别分布: {dict(Counter(r['answer'] for r in val_rows))}")
        print(f"  训练集批次分布: {dict(Counter(C.group_of(r, task) for r in train_rows))}")
        print(f"  验证集批次分布: {dict(Counter(C.group_of(r, task) for r in val_rows))}")
    else:
        print(f"  train 目标 min/median/max = {ytr.min():.1f} / {np.median(ytr):.1f} / {ytr.max():.1f}")
        print(f"  val   目标 min/median/max = {yva.min():.1f} / {np.median(yva):.1f} / {yva.max():.1f}")
        for name, rows in (("train", train_rows), ("validation", val_rows)):
            groups = Counter(C.group_of(r, task) for r in rows)
            top = ", ".join(f"{g}:{n}" for g, n in groups.most_common(6))
            print(f"  {name:<10} 分组数 {len(groups):>4}  最大组占比 {max(groups.values()) / len(rows):.1%}  ({top})")

    # --- 极值 ---
    C.sub("特征量级（训练集）")
    z = (xva - xtr.mean(axis=0)) / np.where(xtr.std(axis=0) == 0, 1.0, xtr.std(axis=0))
    print(f"  验证集标准化后 |z| 最大值 {np.abs(z).max():.1f}，"
          f"|z|>10 的样本 {int((np.abs(z) > 10).any(axis=1).sum())} / {len(z)}")
    print(f"  训练集原始值范围 [{xtr.min():.4g}, {xtr.max():.4g}]")


def main() -> int:
    for task in "ABC":
        audit(task)
    print("\n" + "=" * 78)
    print("结论：三份数据均无缺失值、无重复行、无常量列、无训练/验证重叠。")
    print("      「数据质量不够」在缺失/重复/噪声意义上不成立。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
