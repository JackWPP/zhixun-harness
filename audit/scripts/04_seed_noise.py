#!/usr/bin/env python3
"""04 — 种子噪声：把"提升"放回随机性里看。

固定配方、固定数据、固定超参，只改 shuffle 种子，跑 10 次，
看 demo 里被接受的"提升"是否落在种子噪声区间内。

实现上刻意复刻 harness 的候选执行方式（逐 epoch 验证、保留最佳 epoch），
这样对照组与 demo 里那些候选才是同一口径。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import _common as C

sys.path.insert(0, str(C.ROOT))

from harness.data import prepare_bundle  # noqa: E402
from harness.strategies import build_strategy_by_id  # noqa: E402
from harness.training import train_epoch  # noqa: E402

SEEDS = list(range(42, 52))
EPOCHS = 5


def train_with_best_epoch(task, model, bundle, strategy, workdir: Path):
    """逐 epoch 训练并保留验证集最优的那一轮——与 harness 的候选执行口径一致。"""
    key = C.TASKS[task]["primary"]
    best_metric, best_path = None, workdir / "best.npz"
    for _ in range(strategy.epochs):
        train_epoch(model, bundle.train, strategy)
        primary = C.evaluate(task, model, bundle.validation.x, bundle.validation.y)[key]
        better = best_metric is None or (
            primary > best_metric if task == "A" else primary < best_metric
        )
        if better:
            best_metric = primary
            # 厂商 checkpoint 采用 "xb" 独占写，不能覆盖已有文件
            if best_path.exists():
                best_path.unlink()
            model.save(best_path)
    return C.vendor()[0].Model.load(best_path)


def sweep(task: str, strategy_id: str, accepted_gain: float) -> float:
    C.header(f"任务 {task} — 策略 {strategy_id} 的种子噪声（{len(SEEDS)} 个种子，{EPOCHS} epochs）")
    archive = C.ARCHIVES / C.TASKS[task]["archive"]
    key = C.TASKS[task]["primary"]

    with tempfile.TemporaryDirectory(prefix=f"audit-seed-{task}-") as tmp:
        bundle = prepare_bundle(archive, task, Path(tmp) / "source")
        results = []
        for seed in SEEDS:
            model = C.load_model(task)
            model.meta["seed"] = seed
            model.meta["epochs"] = 0
            strategy = build_strategy_by_id(
                strategy_id,
                task,
                low_lr=max(float(model.meta["learning_rate"]) * 0.3, 1e-6),
                batch_size=int(model.meta["batch_size"]),
                epochs=EPOCHS,
            )
            workdir = Path(tmp) / f"seed-{seed}"
            workdir.mkdir()
            best = train_with_best_epoch(task, model, bundle, strategy, workdir)
            metrics = C.evaluate(task, best, bundle.validation.x, bundle.validation.y)
            results.append(metrics[key])
            print(f"    seed={seed:<3} {key}={metrics[key]:.6f}")

    values = np.array(results)
    mean, std = float(values.mean()), float(values.std(ddof=1))
    print(f"\n  均值 {mean:.6f}  标准差 {std:.6f}  极差 {values.max() - values.min():.6f}")
    print(f"  区间 [{values.min():.6f}, {values.max():.6f}]")
    print(f"  demo 接受的增益 {accepted_gain:+.6f} = {abs(accepted_gain) / std:.2f} σ")

    if task == "A":
        reached = int((values >= mean + accepted_gain).sum())
        print(f"  只靠换种子就能达到或超过「对照组均值+增益」的次数: {reached} / {len(values)}")
    else:
        reached = int((values <= mean - abs(accepted_gain)).sum())
        print(f"  只靠换种子就能把 RMSE 降到「对照组均值−增益」以下的次数: {reached} / {len(values)}")
    verdict = (
        "增益超出种子噪声 → 该增益不是重排运气"
        if reached == 0
        else f"增益落在种子噪声内（{reached}/{len(values)} 次可复现）→ 不能排除重排运气"
    )
    print(f"  判定: {verdict}")
    return std, reached


def main() -> int:
    sd_a, hit_a = sweep("A", "conservative", 0.012642)
    sd_b, hit_b = sweep("B", "regularized", 8.748955)
    print("\n" + "=" * 78)
    print(f"A: 增益 {0.012642 / sd_a:.2f} σ（种子 sd={sd_a:.6f}），换种子复现 {hit_a}/10")
    print(f"B: 增益 {8.748955 / sd_b:.2f} σ（种子 sd={sd_b:.6f}），换种子复现 {hit_b}/10")
    print()
    print("注意：种子噪声只是噪声来源之一，而且往往是最小的一项。")
    print("      验证集本身的抽样噪声（1244 行 / 1679 行）远大于此，见 09_significance.py。")
    print("      因此本脚本的作用是排除「纯粹的重排运气」，而不是证明改进有效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
