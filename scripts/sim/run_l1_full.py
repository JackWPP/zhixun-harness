#!/usr/bin/env python3
"""L1 full-scale batch-stream rehearsal.

Runs each (task, arm, seed) lane as an isolated `rsi.cli stream --source full`
subprocess, then compresses the evidence (decision/evaluation metrics) and
deletes the bulky per-batch packages.

Usage:
    PYTHONPATH=. ./.venv/bin/python scripts/sim/run_l1_full.py \
        --arms frozen,recenter,pair,adapt,full --seeds 42,43 --jobs 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "bin" / "python")


def lane_command(job: dict, args) -> list[str]:
    cmd = [
        PY, "-m", "rsi.cli", "stream",
        "--tasks", job["task"],
        "--source", "full",
        "--batches", str(args.batches),
        "--hidden-tail", str(args.hidden_tail),
        "--arm", job["arm"],
        "--seed", str(job["seed"]),
        "--state-dir", job["state_dir"],
        "--experiments-cap", str(args.experiments_cap),
        "--batch-wall", str(args.batch_wall),
        "--retention-every", str(args.retention_every),
        "--dev-fraction", str(args.dev_fraction),
        "--label-every", str(args.label_every),
        "--label-fraction", str(args.label_fraction),
        "--perturb", args.perturb,
        "--repeat", str(args.repeat),
        "--llm", getattr(args, "llm", "off"),
        "--controller", getattr(args, "controller", "pipeline"),
    ]
    if getattr(args, "llm_env", ""):
        cmd.extend(["--llm-env", args.llm_env])
    if getattr(args, "token_cap", None) is not None:
        cmd.extend(["--token-cap", str(args.token_cap)])
    if getattr(args, "round_turns", None) is not None:
        cmd.extend(["--round-turns", str(args.round_turns)])
    if args.no_memory:
        cmd.append("--no-memory")
    return cmd


def compress_lane(job: dict) -> dict:
    lane = Path(job["state_dir"])
    report_path = lane / "stream_report.json"
    summary = {"task": job["task"], "arm": job["arm"], "seed": job["seed"],
               "state_dir": str(lane), "ok": False, "error": None,
               "batches": [], "final": {}, "hidden": []}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary
    task = report.get("tasks", {}).get(job["task"], {})
    summary["ok"] = not task.get("errors")
    summary["error"] = "; ".join(task.get("errors", [])[:3]) or None
    for batch in task.get("batches", []):
        summary["batches"].append({
            "index": batch.get("index"),
            "phase": batch.get("phase"),
            "arm": batch.get("arm"),
            "revealed": batch.get("revealed"),
            "history_rows": batch.get("history_rows"),
            "dev_primary": batch.get("dev_primary"),
            "prediction_metrics": batch.get("prediction_metrics"),
            "prediction_count": batch.get("prediction_count"),
            "decision": batch.get("decision"),
            "adapt_gate": batch.get("adapt_gate"),
            "wall_seconds": batch.get("wall_seconds"),
            "recenter_weight": batch.get("recenter_weight"),
            "perturb": (batch.get("meta") or {}).get("perturb"),
        })
    summary["hidden"] = [
        {"index": b.get("index"),
         "prediction_metrics": b.get("prediction_metrics"),
         "prediction_count": b.get("prediction_count")}
        for b in summary["batches"] if b.get("phase") == "hidden"
    ]
    summary["final"] = task.get("final", {})
    # Keep compact evidence, drop bulky packages/runs.
    task_root = lane / "tasks" / job["task"]
    if task_root.exists():
        for batch_dir in sorted(task_root.glob("batch_*")):
            shutil.rmtree(batch_dir, ignore_errors=True)
    return summary


def run_lane(job: dict, args) -> dict:
    lane = Path(job["state_dir"])
    lane.mkdir(parents=True, exist_ok=True)
    log_path = lane / "stream.log"
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                lane_command(job, args), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": str(ROOT), "RSI_ALLOW_EXTENDED_DATA": "1"},
                timeout=args.lane_timeout,
            )
            job["returncode"] = proc.returncode
        except subprocess.TimeoutExpired:
            job["returncode"] = -9
            log.write("\n[lane] TIMEOUT\n")
    summary = compress_lane(job)
    summary["returncode"] = job.get("returncode")
    summary["wall_seconds"] = round(time.time() - started, 2)
    (lane / "lane_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lane] {job['task']}/{job['arm']}/s{job['seed']} rc={summary['returncode']} "
          f"ok={summary['ok']} wall={summary['wall_seconds']}s "
          f"batches={len(summary['batches'])} errors={summary['error']}", flush=True)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="A,B,C")
    parser.add_argument("--arms-A", default="frozen,recenter,pair,adapt,full")
    parser.add_argument("--arms-B", default="frozen,recenter,adapt,full")
    parser.add_argument("--arms-C", default="frozen,recenter,adapt")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--hidden-tail", type=int, default=2)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--label-every", type=int, default=1)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--perturb", default="none")
    parser.add_argument("--repeat", type=int, default=1,
                        help="repeat the whole stream N times (long-stream variant)")
    parser.add_argument("--experiments-cap", type=int, default=2)
    parser.add_argument("--batch-wall", type=float, default=900.0)
    parser.add_argument("--lane-timeout", type=float, default=5400.0)
    parser.add_argument("--retention-every", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--llm", default="off", choices=("auto", "on", "off"))
    parser.add_argument("--controller", default="pipeline",
                        choices=("auto", "kernel", "pipeline"))
    parser.add_argument("--llm-env", default="")
    parser.add_argument("--token-cap", type=int, default=None)
    parser.add_argument("--round-turns", type=int, default=None)
    parser.add_argument("--seed-tag", default="base",
                        help="sub-directory tag, e.g. base / scarcity / smallb / long")
    parser.add_argument("--root", default=".scratch/sim/L1")
    args = parser.parse_args(argv)

    arms_by_task = {"A": [a for a in args.arms_A.split(",") if a],
                    "B": [a for a in args.arms_B.split(",") if a],
                    "C": [a for a in args.arms_C.split(",") if a]}
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = [t.strip().upper() for t in args.tasks.split(",") if t.strip()]
    jobs = []
    for task in tasks:
        for arm in arms_by_task.get(task, []):
            for seed in seeds:
                name = f"{task}-{arm}-s{seed}"
                jobs.append({
                    "task": task, "arm": arm, "seed": seed,
                    "state_dir": str(Path(args.root) / args.seed_tag / name),
                })
    print(f"[L1] {len(jobs)} lanes, jobs={args.jobs}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(run_lane, job, args): job for job in jobs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                job = futures[future]
                results.append({"task": job["task"], "arm": job["arm"], "seed": job["seed"],
                                "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    out = Path(args.root) / args.seed_tag / "L1_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "lanes": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    # compact table
    print("\n[L1] summary (arrival mean / hidden mean / worst arrival)")
    for task in tasks:
        for arm in arms_by_task.get(task, []):
            for seed in seeds:
                summary = next((r for r in results if r.get("task") == task
                                and r.get("arm") == arm and r.get("seed") == seed), None)
                if not summary:
                    continue
                vals = [b["prediction_metrics"].get("primary")
                        for b in summary.get("batches", [])
                        if b.get("prediction_metrics") and b.get("decision") is not None]
                vals = [v for v in vals if v is not None]
                hidden = [h["prediction_metrics"].get("primary") for h in summary.get("hidden", [])
                          if h.get("prediction_metrics")]
                if not vals:
                    print(f"  {task} {arm:<10} s{seed}: no records ({summary.get('error')})")
                    continue
                mean = sum(vals) / len(vals)
                worst = min(vals) if task != "B" and task != "C" else max(vals)
                hidden_txt = f" hidden={sum(hidden)/len(hidden):.4f}" if hidden else ""
                print(f"  {task} {arm:<10} s{seed}: mean={mean:.4f} worst={worst:.4f}"
                      f" n={len(vals)}{hidden_txt}")
    print(f"\n[L1] summary -> {out}")
    return 0 if all(r.get("ok") for r in results if r.get("batches")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
