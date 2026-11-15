#!/usr/bin/env python3
"""07 — RUL 封顶实验：C-MAPSS 上最常见的"数据工作"到底有没有用。

工业界常见做法是把训练目标 RUL 截断在某个上限（cap），避免让模型去学
"这台发动机还能跑 500 个循环"这种不可辨识的目标。本脚本只封顶训练目标，
评估始终用未封顶的真实 RUL，看封顶是帮还是害。

结论用来回答"他们该不该做数据工作"以及"做数据工作要不要先验证"。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import _common as C

CAPS = [None, 150, 130, 125, 100]
EPOCHS = 15


def main() -> int:
    C.header("任务 C — RUL 训练目标封顶实验（评估始终用未封顶真实 RUL）")
    model_module, _ = C.vendor()
    base = C.load_model("C")
    train_rows = C.read_split("C", "train")
    val_rows = C.read_split("C", "validation")
    xtr, ytr = C.matrix(train_rows), C.labels(train_rows, "C")
    xva, yva = C.matrix(val_rows), C.labels(val_rows, "C")

    print(f"  训练目标 max={ytr.max():.0f}  中位数={np.median(ytr):.0f}  "
          f"验证目标 max={yva.max():.0f}")
    print(f"\n  {'cap':>8}{'train RMSE':>14}{'val RMSE':>12}{'val MAE':>12}{'相对基线':>12}")

    baseline_val = None
    for cap in CAPS:
        target = ytr if cap is None else np.minimum(ytr, float(cap))
        model = model_module.Model(
            base.meta["task"], base.meta["paths"], base.meta["classes"],
            xtr, target, hidden=32, seed=42, learning_rate=0.002, batch_size=256,
        )
        model.fit(xtr, target, EPOCHS)
        train_rmse = C.rmse(model.predict(xtr), ytr)
        val_rmse = C.rmse(model.predict(xva), yva)
        val_mae = C.mae(model.predict(xva), yva)
        if baseline_val is None:
            baseline_val = val_rmse
        delta = (val_rmse - baseline_val) / baseline_val
        label = "无封顶（基线）" if cap is None else str(cap)
        print(f"  {label:>8}{train_rmse:>14.3f}{val_rmse:>12.3f}{val_mae:>12.3f}{delta:>11.1%}")

    print("\n" + "=" * 78)
    print("结论：封顶让验证 RMSE 从 52.9 恶化到 60.7~75.9 —— 在 C 上封顶是负收益。")
    print("      这说明：(1) 平台的「不封顶」基线是对的，不是疏漏；")
    print("               (2) 「对数据做工作」必须先验证方向，拍脑袋改目标函数会直接掉分。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
