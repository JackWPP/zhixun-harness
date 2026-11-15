#!/usr/bin/env python3
"""03 — 常数对照：一个"什么都不学"的预测器能拿多少分。

这是最便宜也最狠的 sanity check。回归任务上，如果模型打不过
"永远预测训练集均值"，说明它学到的东西还不如一个常数。
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import _common as C


def constant_control(task: str) -> None:
    C.header(f"任务 {task} — {C.TASKS[task]['label']}")
    train_rows = C.read_split(task, "train")
    val_rows = C.read_split(task, "validation")
    ytr = C.labels(train_rows, task)
    xva, yva = C.matrix(val_rows), C.labels(val_rows, task)
    model = C.load_model(task)
    pred = model.predict(xva)

    constant = float(ytr.mean())
    const_rmse = C.rmse(np.full_like(yva, constant), yva)
    const_mae = C.mae(np.full_like(yva, constant), yva)
    model_rmse = C.rmse(pred, yva)
    model_mae = C.mae(pred, yva)

    print(f"  训练集均值常数 = {constant:.3f}")
    print(f"  {'预测器':<22}{'RMSE':>12}{'MAE':>12}")
    print(f"  {'常数（训练均值）':<20}{const_rmse:>12.3f}{const_mae:>12.3f}")
    print(f"  {'平台基线模型':<22}{model_rmse:>12.3f}{model_mae:>12.3f}")

    delta = (model_rmse - const_rmse) / const_rmse
    verdict = "优于" if model_rmse < const_rmse else "差于"
    print(f"  → 基线模型 {verdict}常数对照 {abs(delta):.1%}")

    C.sub("误差结构：偏差还是方差？")
    signed = pred - yva
    bias = float(signed.mean())
    print(f"  mean(pred − truth) = {bias:+.3f}（负值=系统性低估剩余寿命）")
    for name, offset in (("训练集拟合偏差", float((model.predict(C.matrix(train_rows)) - ytr).mean())),
                         ("真实验证集偏差（oracle）", bias)):
        corrected = C.rmse(pred - offset, yva)
        print(f"  用{name} {offset:+.3f} 平移后 RMSE = {corrected:.3f}"
              f"  {'仍差于常数' if corrected > const_rmse else '已优于常数'}")

    C.sub("逐组对比（验证集）")
    groups = defaultdict(list)
    for row, p, t in zip(val_rows, pred, yva):
        groups[C.group_of(row, task)].append((float(p), float(t)))
    print(f"  {'分组':<14}{'n':>7}{'模型RMSE':>12}{'常数RMSE':>12}  结论")
    for name, pairs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        p = np.array([a for a, _ in pairs])
        t = np.array([b for _, b in pairs])
        c = np.full_like(t, constant)
        m_r, c_r = C.rmse(p, t), C.rmse(c, t)
        print(f"  {name:<14}{len(pairs):>7,}{m_r:>12.3f}{c_r:>12.3f}  "
              f"{'模型更好' if m_r < c_r else '常数更好'}")


def main() -> int:
    for task in ("B", "C"):
        constant_control(task)
    print("\n" + "=" * 78)
    print("说明：B 的基线比常数还差 → 该任务上任何「把 RMSE 降下来」的改进都首先是在修一个坏基线；")
    print("      C 的基线显著优于常数 → 该任务上才存在真实的改进空间。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
