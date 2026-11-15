"""Command line interface.

    python -m rsi.cli run    --tasks A,B,C --rounds 10 --llm auto --llm-env .scratch/llm.env
    python -m rsi.cli run    --tasks A --rounds 3 --llm off          # fully deterministic
    python -m rsi.cli predict --checkpoint <path> --postprocess '<json>'
    python -m rsi.cli report  --run-dir .scratch/rsi-runs/<run-id>
    python -m rsi.cli serve   --state .scratch/rsi-runs/<run-id>/state.json --port 7861
    python -m rsi.cli selftest --task A
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rsi.runner import Runner


def load_env_file(path: str) -> None:
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"env file not found: {source}")
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def cmd_run(args) -> int:
    if args.llm_env:
        load_env_file(args.llm_env)
    keys = [item.strip().upper() for item in args.tasks.split(",") if item.strip()]
    runner = Runner(keys=keys, rounds=args.rounds, llm_mode=args.llm, seed=args.seed,
                    state_dir=args.state_dir, wall_cap=args.wall_cap, cpu_cap=args.cpu_cap,
                    experiments_cap=args.experiments_cap, swanlab=args.swanlab)
    summary = runner.run()
    print(json.dumps({"run_id": summary["run_id"],
                      "run_dir": str(runner.run_dir),
                      "llm_available": summary["llm"]["available"],
                      "budget": summary["budget"],
                      "datasets": [{"key": d["key"],
                                    "primary": d["metrics"].get("primary"),
                                    "rounds": len(d["rounds"])}
                                   for d in summary["datasets"]]}, ensure_ascii=False, indent=2))
    print(f"\nstate: {runner.recorder.state_path}")
    print(f"summary: {runner.run_dir / 'summary.json'}")
    print(f"view:  python -m rsi.dashboard --state {runner.recorder.state_path} --port 7861")
    return 0


def cmd_predict(args) -> int:
    from rsi.predict import main as predict_main

    return predict_main(args.predict_args)


def cmd_report(args) -> int:
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"summary.json not found under {run_dir}")
    data = json.loads(summary_path.read_text())
    llm_budget = data['llm']['budget']
    print(f"run: {data['run_id']}  llm={data['llm']['backend']}  "
          f"tokens={llm_budget.get('tokens_used', 0)}/{llm_budget.get('tokens_cap', 0)}")
    for dataset in data["datasets"]:
        metrics = dataset["metrics"]
        print(f"\n[{dataset['key']}] primary={metrics.get('primary'):.6f} "
              f"rounds={len(dataset['rounds'])} complete={dataset['record_complete']}")
        print(f"  baseline_sha={dataset['baseline_checkpoint_sha256'][:26]}...")
        print(f"  final_sha   ={dataset['checkpoint_sha256'][:26]}...")
        decisions = [r["decision"] for r in dataset["rounds"]]
        print(f"  decisions: promote={decisions.count('promote')} "
              f"retain={decisions.count('retain')} rollback={decisions.count('rollback')}")
        for record in dataset["rounds"]:
            print(f"  r{record['index']:02d} {record['decision']:9s} "
                  f"{record['validation_metrics']} {record['change_summary'][:80]}")
    return 0


def cmd_serve(args) -> int:
    cmd = [sys.executable, "-m", "rsi.dashboard", "--state", str(args.state),
           "--port", str(args.port)]
    return subprocess.call(cmd)


def cmd_selftest(args) -> int:
    """Quick offline verification: checkpoint loads, prediction matches, gate works."""
    from rsi.inference import apply_postprocess
    from rsi.metrics import from_predictions, primary
    from rsi.model import ExtendedModel, checkpoint_sha256
    from rsi.splits import build_splits
    from rsi.tasks import resolve_tasks

    config = resolve_tasks([args.task])[0]
    model = ExtendedModel.from_checkpoint(config.checkpoint_path)
    print(f"[ok] checkpoint loaded task={model.meta['task']} params={sum(v.size for v in model.params.values())}")
    print(f"[ok] sha256 {checkpoint_sha256(config.checkpoint_path)}")
    splits = build_splits(config, Path(args.state_dir) / "selftest" / config.key, seed=42)
    for name in ("calibration", "selection", "report"):
        dataset = splits[name]
        predicted = apply_postprocess({"type": "none"}, model, dataset.x)
        values = from_predictions(config.task, predicted, dataset.y, config.classes)
        print(f"[ok] {name:11s} rows={len(dataset):5d} primary={primary(config.task, values):.6f}")
    if splits.get("probe") is not None:
        predicted = apply_postprocess({"type": "none"}, model, splits["probe"].x)
        values = from_predictions(config.task, predicted, splits["probe"].y, config.classes)
        print(f"[ok] probe rows={len(splits['probe'])} primary={primary(config.task, values):.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsi", description="zhixun RSI harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the autonomous loop")
    run.add_argument("--tasks", default="A,B,C")
    run.add_argument("--rounds", type=int, default=10)
    run.add_argument("--llm", choices=("auto", "on", "off"), default="auto")
    run.add_argument("--llm-env", default=os.environ.get("RSI_LLM_ENV", ""))
    run.add_argument("--state-dir", default=os.environ.get("RSI_STATE_DIR", ".scratch/rsi-runs"))
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--wall-cap", type=float, default=float(os.environ.get("RSI_WALL_CAP", 21600)))
    run.add_argument("--cpu-cap", type=float, default=float(os.environ.get("RSI_CPU_CAP", 36000)))
    run.add_argument("--experiments-cap", type=int, default=int(os.environ.get("RSI_EXPERIMENTS_CAP", 40)),
                     help="per-task experiment budget (total = cap x number of tasks)")
    run.add_argument("--swanlab", action="store_true")
    run.set_defaults(func=cmd_run)

    predict = sub.add_parser("predict", help="official-compatible predictor (stdin JSONL)")
    predict.add_argument("predict_args", nargs=argparse.REMAINDER)
    predict.set_defaults(func=cmd_predict)

    report = sub.add_parser("report", help="print a completed run summary")
    report.add_argument("--run-dir", required=True)
    report.set_defaults(func=cmd_report)

    serve = sub.add_parser("serve", help="serve the dashboard for a state.json")
    serve.add_argument("--state", required=True)
    serve.add_argument("--port", type=int, default=7861)
    serve.set_defaults(func=cmd_serve)

    selftest = sub.add_parser("selftest", help="offline sanity checks")
    selftest.add_argument("--task", default="A", choices=("A", "B", "C"))
    selftest.add_argument("--state-dir", default=".scratch/rsi-selftest")
    selftest.set_defaults(func=cmd_selftest)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
