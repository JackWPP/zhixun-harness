#!/usr/bin/env python3
"""Sequential campaign runner for the full RSI simulation.

Reads ``scripts/sim/campaigns.json``, expands ``arms x seeds`` and launches
``scripts/sim/full_env.py`` once per run.  A manifest is written incrementally
so interrupted campaigns can be resumed (completed entries are skipped unless
``--force`` is given).

Usage::

    python scripts/sim/campaign.py --only C1
    python scripts/sim/campaign.py --only C0 --dry-run
    python scripts/sim/campaign.py                 # everything, sequential
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FULL_ENV = REPO / "scripts" / "sim" / "full_env.py"
MATRIX = REPO / "scripts" / "sim" / "campaigns.json"
DEFAULT_OUT = REPO / ".scratch" / "sim" / "runs"
MANIFEST = REPO / ".scratch" / "sim" / "campaign-manifest.jsonl"
LOG_DIR = REPO / ".scratch" / "sim" / "logs"


def load_env_files() -> None:
    """Load 0600 scratch env files without echoing their values."""
    for name in ("pytrio.env", "swanlab.env", "llm.env"):
        path = REPO / ".scratch" / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def expand(matrix: dict, only: list[str] | None) -> list[dict]:
    runs: list[dict] = []
    for campaign, spec in matrix.items():
        if only and campaign not in only:
            continue
        for seed in spec.get("seeds", [1]):
            for arm in spec.get("arms", [{"tag": "default", "policy": "on"}]):
                tag = "%s-%s-s%d" % (campaign, arm.get("tag", "default"), seed)
                run = {
                    "campaign": campaign,
                    "tag": tag,
                    "seed": seed,
                    "scenario": spec["scenario"],
                    "tasks": spec.get("tasks", "A"),
                    "rounds": int(spec.get("rounds", 3)),
                    "provider": spec.get("provider", "real"),
                    "controller": spec.get("controller", "kernel"),
                    "policy": arm.get("policy", "on"),
                    "pair_delta": arm.get("pair_delta"),
                    "hidden_every_round": bool(spec.get("hidden_every_round", False)),
                }
                runs.append(run)
    return runs


def load_manifest() -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not MANIFEST.is_file():
        return done
    for line in MANIFEST.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("status") == "done":
            done[item["tag"]] = item
    return done


def append_manifest(item: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_command(run: dict) -> list[str]:
    command = [
        sys.executable, str(FULL_ENV),
        "--scenario", run["scenario"],
        "--tasks", run["tasks"],
        "--rounds", str(run["rounds"]),
        "--provider", run["provider"],
        "--controller", run["controller"],
        "--policy", run["policy"],
        "--tag", run["tag"],
        "--swanlab", "auto",
    ]
    if run.get("hidden_every_round"):
        command.append("--hidden-every-round")
    if run.get("pair_delta") is not None:
        command.extend(["--pair-delta", str(run["pair_delta"])])
    return command


def parse_run_root(stdout: str) -> str:
    match = re.search(r"\[sim\] run_root=(\S+)", stdout)
    return match.group(1) if match else ""


def parse_swanlab(stdout: str) -> str:
    match = re.search(r"swanlab[^\n]*?(https://\S+)", stdout)
    if match:
        return match.group(1)
    match = re.search(r"(https://swanlab\.cn/\S+)", stdout)
    return match.group(1) if match else ""


def execute(run: dict, timeout: float, retries: int = 1) -> dict:
    command = build_command(run)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / ("campaign-%s.log" % run["tag"])
    started = time.time()
    last_error = ""
    for attempt in range(retries + 1):
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n=== %s attempt %d: %s ===\n" % (time.strftime("%H:%M:%S"), attempt + 1,
                                                        " ".join(command)))
            log.flush()
            try:
                process = subprocess.run(command, cwd=str(REPO), stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True,
                                         timeout=max(60.0, timeout))
                log.write(process.stdout[-20000:])
                log.flush()
                if process.returncode != 0:
                    last_error = "exit=%d" % process.returncode
                    continue
                run_root = parse_run_root(process.stdout)
                summary_path = Path(run_root) / "summary.json"
                summary = {}
                if summary_path.is_file():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                return {
                    "tag": run["tag"], "campaign": run["campaign"], "seed": run["seed"],
                    "run": run, "run_root": run_root, "status": "done",
                    "wall_s": round(time.time() - started, 1),
                    "swanlab": parse_swanlab(process.stdout),
                    "tasks": {key: {"hidden_proxy": value.get("hidden_proxy"),
                                    "rounds": [(entry["round"], entry["validation"],
                                                entry.get("hidden_proxy"))
                                               for entry in value.get("rounds", [])]}
                              for key, value in summary.get("tasks", {}).items()},
                }
            except subprocess.TimeoutExpired:
                last_error = "timeout after %.0fs" % timeout
                log.write("\nTIMEOUT: %s\n" % last_error)
                log.flush()
    return {"tag": run["tag"], "campaign": run["campaign"], "seed": run["seed"],
            "run": run, "status": "failed", "error": last_error,
            "wall_s": round(time.time() - started, 1)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="run full-sim experiment campaigns")
    parser.add_argument("--only", default="", help="comma separated campaign ids (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun completed entries")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    load_env_files()
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    only = [item.strip() for item in args.only.split(",") if item.strip()] or None
    runs = expand(matrix, only)
    done = load_manifest()
    pending = [run for run in runs if args.force or run["tag"] not in done]
    print("[campaign] planned=%d pending=%d (done=%d)" % (len(runs), len(pending), len(done)))
    for run in pending:
        print("  -", run["tag"], run["scenario"], run["tasks"], "r=%d" % run["rounds"],
              "policy=%s" % run["policy"],
              "delta=%s" % run["pair_delta"] if run.get("pair_delta") is not None else "")
    if args.dry_run:
        return 0

    failures = 0
    for index, run in enumerate(pending, 1):
        print("[campaign] %d/%d %s" % (index, len(pending), run["tag"]), flush=True)
        item = execute(run, args.timeout)
        append_manifest(item)
        if item["status"] != "done":
            failures += 1
            print("[campaign] FAILED %s: %s" % (run["tag"], item.get("error")), file=sys.stderr)
        else:
            hidden = {key: value["hidden_proxy"] for key, value in item["tasks"].items()}
            print("[campaign] done %s wall=%.1fs hidden=%s" % (run["tag"], item["wall_s"], hidden))
    print("[campaign] finished pending=%d failures=%d manifest=%s" %
          (len(pending), failures, MANIFEST))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
