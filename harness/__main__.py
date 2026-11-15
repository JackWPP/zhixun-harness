"""Usage:
  python -m harness run --task A --archive package.zip --output NEW_DIR [--decision-mode rule|small_model|hybrid]
  python -m harness train-recommender --runs-dir harness-runs --output strategy_recommender.npz
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU 工业模型诊断、候选续训与验证门控 Harness（集成 RuleDrivenHarnessTrainer）")
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("run", help="诊断 -> 选策略(规则/小模型) -> RuleDrivenHarnessTrainer 续训 -> 验证门控 -> HTML 报告")
    command.add_argument("--task", choices=("A", "B", "C"), required=True)
    command.add_argument("--archive", required=True, help="明确指定的原始 benchmark zip；只读")
    command.add_argument("--output", required=True, help="必须是新目录，父目录须存在")
    command.add_argument("--epochs", type=int, default=5, help="每候选最多训练轮数，默认5")
    command.add_argument("--max-candidates", type=int, default=3, help="每轮最多1–4个候选")
    command.add_argument("--rounds", type=int, default=1, help="最多1–20轮，拒绝更新后提前结束")
    command.add_argument("--patience", type=int, default=2, help="连续无主指标改善多少轮后早停")
    command.add_argument("--min-improvement", type=float, default=0.0, help="主指标绝对改善阈值，必须严格超过")
    command.add_argument("--max-train-seconds", type=float, default=60.0, help="所有候选共享训练时间预算（不含诊断/验证/IO）")
    command.add_argument("--threads", type=int, choices=range(1, 9), default=1)
    command.add_argument(
        "--decision-mode",
        choices=("rule", "small_model", "hybrid"),
        default="rule",
        help="候选策略选择模式：rule(默认规则路由)、small_model(策略推荐小模型) 或 hybrid(混合)",
    )
    command.add_argument("--strategy-model", default=None, help="可选的预训练策略推荐小模型 .npz 路径")

    train_cmd = sub.add_parser("train-recommender", help="从历史 harness-runs 目录提取经验轨迹并训练策略推荐小模型")
    train_cmd.add_argument("--runs-dir", required=True, help="包含 A-v1/B-v1/C-v1 等运行子目录的父目录")
    train_cmd.add_argument("--output", required=True, help="输出的策略推荐小模型 .npz 文件路径")
    train_cmd.add_argument("--epochs", type=int, default=150, help="小模型训练轮数")

    args = parser.parse_args(argv)

    if args.command == "train-recommender":
        from .strategies import StrategyRecommenderMLP
        from .training import ExperienceBuffer

        runs_root = Path(args.runs_dir)
        run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir() and (p / "summary.json").is_file()])
        if not run_dirs and (runs_root / "summary.json").is_file():
            run_dirs = [runs_root]
        buffer = ExperienceBuffer.from_run_directories(run_dirs)
        model = StrategyRecommenderMLP()
        stats = model.fit_from_buffer(buffer, epochs=args.epochs)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(out_path)
        print(json.dumps({"status": "completed", "output": str(out_path.resolve()), **stats}, ensure_ascii=False, indent=2))
        return 0

    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["BASELINE_CPU_THREADS"] = str(args.threads)
    from .runner import run
    try:
        result = run(
            args.task,
            args.archive,
            args.output,
            args.epochs,
            args.max_candidates,
            args.rounds,
            args.min_improvement,
            args.max_train_seconds,
            args.patience,
            decision_mode=args.decision_mode,
            strategy_model_path=args.strategy_model,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": result["status"],
                "task": result["task"],
                "trainer_engine": result.get("configuration", {}).get("trainer_engine"),
                "decision_mode": result.get("configuration", {}).get("decision_mode"),
                "baseline": result.get("baseline", {}).get("metrics"),
                "selected": result.get("selected"),
                "cost": result["cost"],
                "report": os.path.abspath(os.path.join(args.output, "report.html")),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
