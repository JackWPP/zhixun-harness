#!/usr/bin/env python3
"""Aggregate L1/L2/L3 full-scale rehearsal evidence into a markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTION = {"A": "higher", "B": "lower", "C": "lower"}


def _load_json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _primary(batch: dict):
    metrics = batch.get("prediction_metrics") or {}
    return metrics.get("primary")


def _lane_stats(lane: dict) -> dict:
    task = lane["task"]
    vals = [_primary(b) for b in lane.get("batches", []) if b.get("phase") != "hidden"]
    vals = [v for v in vals if isinstance(v, (int, float))]
    hidden = [_primary(h) for h in lane.get("hidden", [])]
    hidden = [v for v in hidden if isinstance(v, (int, float))]
    if not vals:
        return {"task": task, "arm": lane.get("arm"), "seed": lane.get("seed"),
                "n": 0, "mean": None, "worst": None, "hidden_mean": None,
                "decisions": {}, "adapt_kept": None, "wall": lane.get("wall_seconds")}
    decisions: dict[str, int] = {}
    for batch in lane.get("batches", []):
        key = str(batch.get("decision") or "pending")
        decisions[key] = decisions.get(key, 0) + 1
    kept = [b.get("adapt_gate", {}) for b in lane.get("batches", [])
            if isinstance(b.get("adapt_gate"), dict) and b.get("adapt_gate", {}).get("attempted")]
    return {
        "task": task,
        "arm": lane.get("arm"),
        "seed": lane.get("seed"),
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "worst": min(vals) if DIRECTION[task] == "higher" else max(vals),
        "hidden_mean": (sum(hidden) / len(hidden)) if hidden else None,
        "decisions": decisions,
        "adapt_kept": f"{sum(1 for k in kept if k.get('kept'))}/{len(kept)}" if kept else None,
        "wall": lane.get("wall_seconds"),
        "error": lane.get("error"),
    }


def _l1(path: Path) -> list[dict]:
    data = _load_json(path, {})
    return [_lane_stats(lane) for lane in data.get("lanes", [])]


def _l3(root: Path) -> list[dict]:
    rows = []
    for task in ("A", "B", "C"):
        path = root / f"kernel-{task}" / "L1_summary.json"
        data = _load_json(path, {})
        for lane in data.get("lanes", []):
            rows.append({**_lane_stats(lane), "kernel": True})
    return rows


def _l2(path: Path) -> dict:
    return _load_json(path, {})


def _fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build(root: Path) -> dict:
    l1 = _l1(root / "L1" / "base" / "L1_summary.json")
    l3 = _l3(root / "L3")
    l2 = _l2(root / "L2" / "report.json")
    issues: list[dict] = []
    path = root / "issues.jsonl"
    if path.is_file():
        issues = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    return {"l1": l1, "l2": l2, "l3": l3, "issues": issues}


def markdown(report: dict) -> str:
    lines = ["# Full-scale batch-stream rehearsal", ""]
    lines.append("## L1 — protocol-level full-data streams")
    lines.append("")
    lines.append("| task | arm | seed | batches | mean primary | worst | hidden mean | decisions | adapt kept | wall(s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for lane in report["l1"]:
        decisions = ",".join(f"{k}:{v}" for k, v in sorted(lane.get("decisions", {}).items()))
        lines.append(f"| {lane['task']} | {lane['arm']} | {lane['seed']} | {lane['n']} | "
                     f"{_fmt(lane['mean'])} | {_fmt(lane['worst'])} | {_fmt(lane['hidden_mean'])} | "
                     f"{decisions or '-'} | {lane['adapt_kept'] or '-'} | {_fmt(lane['wall'], 1)} |")
    lines.append("")
    lines.append("## L2 — workspace-v2 batch path (fake host, real entrypoint)")
    lines.append("")
    for task, block in sorted(report["l2"].items()):
        lines.append(f"### {task} — {'PASS' if block.get('ok') else 'FAIL'}")
        if not block.get("ok"):
            lines.append(f"error: {block.get('error')}")
        rounds = block.get("rounds", [])
        if rounds:
            lines.append("")
            lines.append("| round | batch | hidden | history rows | prediction rows | metric |")
            lines.append("|---|---|---|---|---|---|")
            for item in rounds:
                metric = item.get("metrics", {})
                key = "macro_f1" if task == "A" else "rmse"
                lines.append(f"| {item['round_index']} | {item.get('batch_no')} | {int(item.get('hidden', False))} | "
                             f"{item.get('history_rows')} | {item.get('prediction_rows')} | "
                             f"{key}={_fmt(metric.get(key))} |")
        lines.append("")
    if report["l3"]:
        lines.append("## L3 — kernel-LLM batch stream")
        lines.append("")
        lines.append("| task | arm | seed | batches | mean | worst | hidden mean | wall(s) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for lane in report["l3"]:
            lines.append(f"| {lane['task']} | {lane['arm']} | {lane['seed']} | {lane['n']} | "
                         f"{_fmt(lane['mean'])} | {_fmt(lane['worst'])} | {_fmt(lane['hidden_mean'])} | "
                         f"{_fmt(lane['wall'], 1)} |")
        lines.append("")
    lines.append("## Issues found and fixed")
    lines.append("")
    if report["issues"]:
        for issue in report["issues"]:
            lines.append(f"- **{issue.get('id')}** [{issue.get('kind')}] {issue.get('symptom')}  ")
            lines.append(f"  fix: {issue.get('fix')}  ")
            lines.append(f"  evidence: {issue.get('evidence')}  ")
            lines.append(f"  status: {issue.get('status', 'fixed')}")
    else:
        lines.append("No issues recorded yet.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".scratch/sim")
    parser.add_argument("--out", default="docs/evidence/full_sim/full_sim_report.md")
    args = parser.parse_args(argv)
    report = build(Path(args.root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(report), encoding="utf-8")
    (out.parent / "full_sim_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
