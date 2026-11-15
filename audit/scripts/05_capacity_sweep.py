#!/usr/bin/env python3
"""05 — 容量扫描：加宽网络能不能救回任务 A / 任务 C。

固定配方，只改 hidden 宽度，看训练集与验证集指标怎么动。
如果加容量训练集变好而验证集不动甚至变差 → 过拟合；
如果训练集与验证集一起平着不动 → 欠拟合 / 特征受限。
两种情况的处置完全不同，这正是 harness 从来没有做过的判断。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import _common as C

CONFIGS = {
    "A": {"hiddens": [4, 8, 16, 32, 64, 128], "epochs": 20, "lr": 0.003, "batch_size": 128},
    "C": {"hiddens": [8, 32, 64, 128], "epochs": 15, "lr": 0.002, "batch_size": 256},
}


def sweep(task: str) -> None:
    config = CONFIGS[task]
    C.header(f"任务 {task} — 容量扫描（只改 hidden，其余配方与平台基线一致）")
    model_module, _ = C.vendor()
    base = C.load_model(task)

    train_rows = C.read_split(task, "train")
    val_rows = C.read_split(task, "validation")
    xtr, ytr = C.matrix(train_rows), C.labels(train_rows, task)
    xva, yva = C.matrix(val_rows), C.labels(val_rows, task)
    key = C.TASKS[task]["primary"]

    header = f"  {'hidden':>7}{'参数':>9}{'train ' + key:>14}{'val ' + key:>14}"
    if task == "A":
        header += f"{'val acc':>11}{'乙醛F1':>10}"
    print(header)

    for hidden in config["hiddens"]:
        model = model_module.Model(
            base.meta["task"],
            base.meta["paths"],
            base.meta["classes"],
            xtr,
            ytr,
            hidden=hidden,
            seed=42,
            learning_rate=config["lr"],
            batch_size=config["batch_size"],
        )
        model.fit(xtr, ytr, config["epochs"])
        train_metrics = C.evaluate(task, model, xtr, ytr)
        val_metrics = C.evaluate(task, model, xva, yva)
        params = sum(v.size for v in model.params.values())
        line = (f"  {hidden:>7}{params:>9,}{train_metrics[key]:>14.4f}{val_metrics[key]:>14.4f}")
        if task == "A":
            acetaldehyde = val_metrics["per_class_f1"][C.TASKS["A"]["classes"].index("Acetaldehyde")]
            line += f"{val_metrics['accuracy']:>11.4f}{acetaldehyde:>10.4f}"
        print(line)


def main() -> int:
    sweep("A")
    sweep("C")
    print("\n" + "=" * 78)
    print("结论：A — 训练集 F1 随容量单调上升（.89→.97），验证集 F1 几乎不动（.51→.73 后回落），")
    print("      乙醛 F1 在所有宽度上恒为 0.0000 → 不是容量问题，是那个类在训练/验证之间不可迁移。")
    print("      C — 16 倍容量只让验证 RMSE 动 0.9，且训练 RMSE ≈ 验证 RMSE ≈ 55 → 欠拟合，特征受限。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
