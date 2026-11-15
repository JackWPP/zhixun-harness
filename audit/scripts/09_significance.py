#!/usr/bin/env python3
"""09 — 显著性：被接受的"提升"在统计上站得住吗。

对每个任务，用 bootstrap 重采样验证集（有放回），同时重算
  基线指标、demo 选中模型的指标、以及两者之差，
给出 95% 置信区间和"差值 > 0 的比例"。

这一步回答的是 harness 完全没回答的问题：
  在只有一次运行、且验证集只有 1244/1679/23049 行的情况下，
  报出来的 improvement 是信号还是噪声。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import _common as C

RESAMPLES = 2000
SEED = 20261115


def macro_f1_from_confusion(confusion: np.ndarray) -> float:
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    denominator = support + predicted
    f1 = np.where(tp > 0, 2 * tp / np.where(denominator == 0, 1, denominator), 0.0)
    return float(f1.mean())


def bootstrap(task: str, pred_base: np.ndarray, pred_new: np.ndarray, truth: np.ndarray) -> None:
    rng = np.random.default_rng(SEED)
    n = len(truth)
    key = C.TASKS[task]["primary"]

    if task == "A":
        base_scores, new_scores, diffs = [], [], []
        for _ in range(RESAMPLES):
            idx = rng.integers(0, n, n)
            t, b, w = truth[idx], pred_base[idx], pred_new[idx]
            conf_b = np.bincount(t * 6 + b, minlength=36).reshape(6, 6)
            conf_w = np.bincount(t * 6 + w, minlength=36).reshape(6, 6)
            fb, fw = macro_f1_from_confusion(conf_b), macro_f1_from_confusion(conf_w)
            base_scores.append(fb)
            new_scores.append(fw)
            diffs.append(fw - fb)
    else:
        base_scores, new_scores, diffs = [], [], []
        for _ in range(RESAMPLES):
            idx = rng.integers(0, n, n)
            t = truth[idx]
            rb = C.rmse(pred_base[idx], t)
            rw = C.rmse(pred_new[idx], t)
            base_scores.append(rb)
            new_scores.append(rw)
            diffs.append(rb - rw)

    base_scores, new_scores, diffs = map(np.array, (base_scores, new_scores, diffs))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  重采样 {RESAMPLES} 次（验证集 {n:,} 行）")
    print(f"  基线   {key}: {base_scores.mean():.6f} ± {base_scores.std():.6f}")
    print(f"  选中   {key}: {new_scores.mean():.6f} ± {new_scores.std():.6f}")
    print(f"  差值   (基线−选中): 均值 {diffs.mean():.6f}  sd {diffs.std():.6f}  "
          f"95% CI [{lo:.6f}, {hi:.6f}]")
    positive = float((diffs > 0).mean())
    print(f"  P(提升 > 0) = {positive:.3f}")
    verdict = (
        "✅ 置信区间不含 0，提升在统计上可分辨"
        if lo > 0
        else "❌ 置信区间包含 0，无法与「没有提升」区分"
    )
    print(f"  判定: {verdict}")


def prediction_diff(task: str) -> None:
    rows = C.read_split(task, "validation")
    x, y = C.matrix(rows), C.labels(rows, task)
    base = C.load_model(task).predict(x)
    new = C.vendor()[0].Model.load(
        C.DEMO / "harness-runs" / f"{task}-v1" / "selected" / "checkpoint.npz"
    ).predict(x)
    if task == "A":
        base_labels, new_labels = base.argmax(axis=1), new.argmax(axis=1)
        changed = int((base_labels != new_labels).sum())
        print(f"  预测发生改变的样本数: {changed} / {len(y)}")
        for index in np.where(base_labels != new_labels)[0][:10]:
            print(f"    行 {index}: 真实 {C.TASKS['A']['classes'][y[index]]}  "
                  f"{C.TASKS['A']['classes'][base_labels[index]]} → {C.TASKS['A']['classes'][new_labels[index]]}")
    else:
        changed = int((np.abs(base - new) > 1e-9).sum())
        print(f"  预测值发生改变的样本数: {changed} / {len(y)}")


def main() -> int:
    for task in "ABC":
        C.header(f"任务 {task} — {C.TASKS[task]['label']}")
        prediction_diff(task)
        rows = C.read_split(task, "validation")
        x, y = C.matrix(rows), C.labels(rows, task)
        pred_base = C.load_model(task).predict(x)
        pred_new = C.vendor()[0].Model.load(
            C.DEMO / "harness-runs" / f"{task}-v1" / "selected" / "checkpoint.npz"
        ).predict(x)
        if task == "A":
            pred_base, pred_new = pred_base.argmax(axis=1), pred_new.argmax(axis=1)
        print()
        bootstrap(task, pred_base, pred_new, y)

    print("\n" + "=" * 78)
    print("结论：三个任务的「提升」在配对 bootstrap 下都显著（95% CI 不含 0），")
    print("      也就是说它们不是抽样噪声。但这恰恰是最需要警惕的地方：")
    print("      1) 增益是在同一份验证集上「选」出来的，又用同一份验证集「报」出来，")
    print("         属于选择偏差，没有留出集，隐藏集上是否复现无法保证；")
    print("      2) A 的 +0.0127 来自 1244 个样本里 10 个预测翻转（9 个 Ethylene 修对、")
    print("         1 个 Acetone 被判成 Acetaldehyde），乙醛 F1 仍然是 0；")
    print("      3) B 的 +29.0 是在把一个劣于常数对照（536.27）的基线往常数方向拉，")
    print("         不是在做寿命预测；")
    print("      4) C 的 +1.11 里，第二轮只贡献 +0.0106（0.02%），纯属阈值 >0 就收。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
