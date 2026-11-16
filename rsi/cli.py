"""Command line interface.

    python -m rsi.cli run    --tasks A,B,C --rounds 10 --llm auto --llm-env .scratch/llm.env
    python -m rsi.cli run    --tasks A --rounds 3 --llm off          # fully deterministic
    python -m rsi.cli predict --checkpoint <path> --postprocess '<json>'
    python -m rsi.cli report  --run-dir .scratch/rsi-runs/<run-id>
    python -m rsi.cli serve   --state .scratch/rsi-runs/<run-id>/state.json --port 7861
    python -m rsi.cli selftest --task A
    python -m rsi.cli probe  --task A --path <private.jsonl> --checkpoint <ckpt.npz>
    python -m rsi.cli synth  --task A --batches 4 --output .scratch/synth/A.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rsi.runner import Runner
from rsi.tasks import describe_tasks, register_task_specs


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


def _register_specs(args) -> None:
    spec = getattr(args, "task_spec", None)
    added = register_task_specs(spec)
    if added:
        print(f"[spec] registered tasks: {', '.join(added)}")


def cmd_run(args) -> int:
    _register_specs(args)
    if args.llm_env:
        load_env_file(args.llm_env)
    if args.token_cap is not None:
        os.environ["RSI_TOKEN_CAP"] = str(int(args.token_cap))
    keys = [item.strip().upper() for item in args.tasks.split(",") if item.strip()]
    runner = Runner(keys=keys, rounds=args.rounds, llm_mode=args.llm, seed=args.seed,
                    state_dir=args.state_dir, wall_cap=args.wall_cap, cpu_cap=args.cpu_cap,
                    experiments_cap=args.experiments_cap, swanlab=args.swanlab,
                    controller=args.controller, memory=not args.no_memory,
                    memory_dir=args.memory_dir or None, db_path=args.db_path or None,
                    round_turns=args.round_turns, judge=not args.no_judge)
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


def cmd_stream(args) -> int:
    """Batch-stream mode: labelled batches in, decision + predictions out."""
    _register_specs(args)
    if args.llm_env:
        load_env_file(args.llm_env)
    if args.token_cap is not None:
        os.environ["RSI_TOKEN_CAP"] = str(int(args.token_cap))
    keys = [item.strip().upper() for item in args.tasks.split(",") if item.strip()]
    from rsi.stream import run_stream

    report = run_stream(
        keys=keys,
        source=args.source,
        n_batches=args.batches,
        state_dir=args.state_dir,
        controller=args.controller,
        llm_mode=args.llm,
        seed=args.seed,
        parallel_tasks=args.parallel_tasks,
        memory=not args.no_memory,
        experiments_cap=args.experiments_cap,
        batch_wall=args.batch_wall,
        retention_every=args.retention_every,
        round_turns=args.round_turns,
        adapt_history=args.adapt_history,
        adapt_epochs=args.adapt_epochs,
        adapt_lr=args.adapt_lr,
        adapt_weight=args.adapt_weight,
        recenter_weight=args.recenter_weight,
        pair_calibration=not args.no_pair_calibration,
        pair=[int(v) for v in args.pair.split(",")],
        arm=args.arm,
        arrival_start=args.arrival_start,
        arrival_end=args.arrival_end,
        hidden_start=args.hidden_start,
        hidden_end=args.hidden_end,
        dev_fraction=args.dev_fraction,
        label_every=args.label_every,
        label_fraction=args.label_fraction,
        perturb=args.perturb,
        missing_class=args.missing_class,
        repeat=args.repeat,
        external_root=args.external_root or None,
        external_start=args.external_start,
        external_end=args.external_end,
        holdout=args.holdout,
    )
    summary = {
        "protocol": report["protocol"],
        "source": report["source"],
        "n_batches": report["n_batches"],
        "report": str(Path(args.state_dir) / "stream_report.json"),
        "tasks": {
            key: {
                "batches": task.get("final", {}).get("batches"),
                "first_primary": task.get("final", {}).get("first_primary"),
                "final_primary": task.get("final", {}).get("final_primary"),
                "hidden_mean_primary": task.get("final", {}).get("hidden_mean_primary"),
                "hidden_metrics": task.get("final", {}).get("hidden_metrics"),
                "errors": task.get("errors", []),
            }
            for key, task in report.get("tasks", {}).items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
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
    _register_specs(args)
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


def cmd_probe(args) -> int:
    _register_specs(args)
    from rsi.adapters import evaluate_private_file, SchemaError
    from rsi.tasks import resolve_tasks

    config = resolve_tasks([args.task.upper()])[0]
    checkpoint = args.checkpoint or str(config.checkpoint_path)
    postprocess = {"type": "none"}
    if args.postprocess:
        raw = args.postprocess.strip()
        if os.path.isfile(raw):
            data = json.loads(Path(raw).read_text(encoding="utf-8"))
            postprocess = data.get("postprocess", data)
        else:
            postprocess = json.loads(raw)
    try:
        report = evaluate_private_file(config, args.path, checkpoint, postprocess, args.output)
    except SchemaError as exc:
        print(json.dumps({"ok": False, "error": f"schema: {exc}"}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"ok": True, "primary": report["primary"],
                      "metrics": report["metrics"],
                      "rows": report["dataset"]["rows"],
                      "groups": [{"group": row["group"], "rows": row["rows"], "primary": row["primary"]}
                                 for row in report["per_group"]],
                      "training_isolation": report["training_isolation"]},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_synth(args) -> int:
    _register_specs(args)
    from rsi.adapters import synthesize, write_synthetic
    from rsi.tasks import resolve_tasks

    config = resolve_tasks([args.task.upper()])[0]
    rows = synthesize(config, args.source or config.train_path, batches=args.batches,
                      seed=args.seed, drift=args.drift, noise=args.noise)
    output = write_synthetic(rows, args.output)
    print(json.dumps({"ok": True, "output": str(output), "rows": len(rows),
                      "batches": args.batches, "drift": args.drift, "noise": args.noise},
                     ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsi", description="zhixun RSI harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the autonomous loop")
    run.add_argument("--task-spec", action="append", metavar="PATH",
                     help="portable task.json (file or directory); repeatable, also RSI_TASK_SPECS")
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
    run.add_argument("--controller", choices=("auto", "kernel", "pipeline"),
                     default=os.environ.get("RSI_CONTROLLER", "auto"),
                     help="auto: kernel when an LLM endpoint is available, else pipeline")
    run.add_argument("--no-memory", action="store_true",
                     help="disable persistent memory / sqlite index (ablation)")
    run.add_argument("--memory-dir", default=os.environ.get("RSI_MEMORY_DIR", ""))
    run.add_argument("--db-path", default=os.environ.get("RSI_DB_PATH", ""))
    run.add_argument("--round-turns", type=int, default=int(os.environ.get("RSI_ROUND_TURNS", 20)),
                     help="max PI agent turns inside one autonomous round")
    run.add_argument("--token-cap", type=int, default=None,
                     help="LLM token budget for the whole run (0 = unlimited, dev default)")
    run.add_argument("--no-judge", action="store_true", help="disable the deterministic goal judge")
    run.set_defaults(func=cmd_run)

    stream = sub.add_parser("stream", help="batch-stream protocol (track S official / track X external)")
    stream.add_argument("--task-spec", action="append", metavar="PATH",
                        help="portable task.json (file or directory); repeatable")
    stream.add_argument("--tasks", default="A")
    stream.add_argument("--source", choices=("official", "external-a", "private-sim"), default="official")
    stream.add_argument("--batches", type=int, default=6)
    stream.add_argument("--state-dir", default=os.environ.get("RSI_STREAM_DIR", ".scratch/rsi-stream"))
    stream.add_argument("--llm", choices=("auto", "on", "off"), default="off")
    stream.add_argument("--llm-env", default=os.environ.get("RSI_LLM_ENV", ""))
    stream.add_argument("--controller", choices=("auto", "kernel", "pipeline"),
                        default=os.environ.get("RSI_CONTROLLER", "auto"))
    stream.add_argument("--seed", type=int, default=42)
    stream.add_argument("--parallel-tasks", type=int, default=1,
                        help="run independent task streams in parallel threads")
    stream.add_argument("--experiments-cap", type=int, default=int(os.environ.get("RSI_EXPERIMENTS_CAP", 40)))
    stream.add_argument("--batch-wall", type=float, default=float(os.environ.get("RSI_BATCH_WALL", 1800)))
    stream.add_argument("--retention-every", type=int, default=2,
                        help="re-evaluate deployed model on all earlier batches every N batches (0=off)")
    stream.add_argument("--adapt-history", action="store_true",
                        help="continue-train the parent on cumulative labelled history before each batch")
    stream.add_argument("--adapt-epochs", type=int, default=30,
                        help="epochs for the history-adaptation continuation (default 30)")
    stream.add_argument("--adapt-lr", type=float, default=0.0005,
                        help="learning rate for the history-adaptation continuation")
    stream.add_argument("--adapt-weight", type=float, default=0.5,
                        help="recenter blend weight after history adaptation")
    stream.add_argument("--arm", default="auto",
                        choices=["auto", "frozen", "recenter", "pair", "adapt", "full"],
                        help="adaptation arm under test (default: auto route selection)")
    stream.add_argument("--arrival-start", type=int, default=3)
    stream.add_argument("--arrival-end", type=int, default=7)
    stream.add_argument("--hidden-start", type=int, default=8)
    stream.add_argument("--hidden-end", type=int, default=10)
    stream.add_argument("--dev-fraction", type=float, default=0.25,
                        help="fraction of revealed history reserved as labelled dev")
    stream.add_argument("--label-every", type=int, default=1,
                        help="reveal labels only every N arrival batches (1=always)")
    stream.add_argument("--label-fraction", type=float, default=1.0,
                        help="fraction of a revealed batch added to history")
    stream.add_argument("--perturb", default="none",
                        choices=["none", "gain", "offset", "noise", "gain_offset"])
    stream.add_argument("--missing-class", default="")
    stream.add_argument("--repeat", type=int, default=1)
    stream.add_argument("--recenter-weight", type=float, default=0.4)
    stream.add_argument("--no-pair-calibration", action="store_true",
                        help="disable history-calibrated pair threshold (A: Ethanol vs Acetaldehyde)")
    stream.add_argument("--pair", default=os.environ.get("RSI_STREAM_PAIR", "0,3"),
                        help="class-index pair for threshold calibration (default 0,3)")
    stream.add_argument("--round-turns", type=int, default=int(os.environ.get("RSI_ROUND_TURNS", 20)))
    stream.add_argument("--token-cap", type=int, default=None)
    stream.add_argument("--no-memory", action="store_true")
    stream.add_argument("--external-root", default=os.environ.get("RSI_GAS_ROOT", ""),
                        help="directory with UCI gas batch*.dat for --source external-a")
    stream.add_argument("--external-start", type=int, default=3)
    stream.add_argument("--external-end", type=int, default=10)
    stream.add_argument("--holdout", type=float, default=0.3,
                        help="fraction of an external batch used for evaluation (track X)")
    stream.set_defaults(func=cmd_stream)

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

    probe = sub.add_parser("probe", help="evaluate a private/hidden-style JSONL probe")
    probe.add_argument("--task-spec", action="append", metavar="PATH", help="portable task.json")
    probe.add_argument("--task", required=True, choices=("A", "B", "C"))
    probe.add_argument("--path", required=True, help="JSONL file following the official schema")
    probe.add_argument("--checkpoint", default="", help="defaults to the official baseline checkpoint")
    probe.add_argument("--postprocess", default="", help="JSON string or a JSON file with a postprocess key")
    probe.add_argument("--output", default="", help="optional report path (json)")
    probe.set_defaults(func=cmd_probe)

    synth = sub.add_parser("synth", help="generate deterministic synthetic drift probes")
    synth.add_argument("--task-spec", action="append", metavar="PATH", help="portable task.json")
    synth.add_argument("--task", required=True, choices=("A", "B", "C"))
    synth.add_argument("--batches", type=int, default=4)
    synth.add_argument("--source", default="", help="source split; defaults to official public train")
    synth.add_argument("--seed", type=int, default=42)
    synth.add_argument("--drift", type=float, default=2.5)
    synth.add_argument("--noise", type=float, default=0.25)
    synth.add_argument("--output", required=True)
    synth.set_defaults(func=cmd_synth)

    selftest = sub.add_parser("selftest", help="offline sanity checks")
    selftest.add_argument("--task-spec", action="append", metavar="PATH", help="portable task.json")
    selftest.add_argument("--task", default="A", choices=("A", "B", "C"))
    selftest.add_argument("--state-dir", default=".scratch/rsi-selftest")
    selftest.set_defaults(func=cmd_selftest)

    tasks = sub.add_parser("tasks", help="list registered tasks (built-in + task.json specs)")
    tasks.add_argument("--task-spec", action="append", metavar="PATH", help="portable task.json")
    tasks.set_defaults(func=cmd_tasks)
    return parser


def cmd_tasks(args) -> int:
    _register_specs(args)
    print(describe_tasks())
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
