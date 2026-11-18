#!/usr/bin/env python3
"""Aggregate full-simulation runs into CSV / Markdown / SVG tables.

Reads every ``.scratch/sim/runs/*/summary.json`` plus the campaign manifest and
the offline strategy-lab reports, then prints:

* per-arm hidden-proxy mean/std (last round),
* per-round hidden learning curves (when ``--hidden-every-round`` was used),
* a small dependency-free SVG bar chart.

Usage::

    python scripts/sim/analyze.py
    python scripts/sim/analyze.py --out .scratch/sim/lab
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / ".scratch" / "sim" / "runs"
MANIFEST = REPO / ".scratch" / "sim" / "campaign-manifest.jsonl"
LAB = REPO / ".scratch" / "sim" / "lab"


def _mean_std(values: list[float]) -> tuple[float, float]:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return float("nan"), float("nan")
    return statistics.fmean(clean), (statistics.stdev(clean) if len(clean) > 1 else 0.0)


def load_runs() -> list[dict]:
    items = []
    if not RUNS.is_dir():
        return items
    for summary_path in sorted(RUNS.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        root = summary_path.parent
        tasks = {}
        for key, value in (summary.get("tasks") or {}).items():
            rounds = value.get("rounds") or []
            tasks[key] = {
                "hidden_proxy": value.get("hidden_proxy"),
                "hidden_curve": [entry.get("hidden_proxy") for entry in rounds],
                "val_curve": [entry.get("validation") for entry in rounds],
                "chosen": [entry.get("chosen") for entry in rounds],
                "calls": sum(int(entry.get("calls") or 0) for entry in rounds),
            }
        items.append({
            "run_root": str(root),
            "name": root.name,
            "scenario": summary.get("scenario"),
            "tag": summary.get("tag") or "",
            "policy": summary.get("policy") or "",
            "pair_delta": summary.get("pair_delta"),
            "rounds": summary.get("rounds"),
            "tasks": tasks,
        })
    return items


def load_manifest() -> list[dict]:
    if not MANIFEST.is_file():
        return []
    items = []
    for line in MANIFEST.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def arm_label(item: dict) -> str:
    parts = [item["scenario"] or "?"]
    if item.get("tag"):
        parts.append(item["tag"])
    if item.get("policy"):
        parts.append("policy=%s" % item["policy"])
    if item.get("pair_delta") is not None:
        parts.append("delta=%s" % item["pair_delta"])
    return " ".join(parts)


def svg_bars(rows: list[tuple[str, float, float]], width: int = 960) -> str:
    height = 60 + 34 * max(1, len(rows))
    bars = []
    max_value = max((value for _, value, _ in rows if value == value), default=0.0) or 1.0
    for index, (label, value, error) in enumerate(rows):
        y = 50 + index * 34
        usable = width - 430
        bar = 0.0 if value != value else usable * max(0.0, value) / max_value
        bars.append(
            '<text x="8" y="%d" font-size="13" font-family="Menlo,monospace" fill="#334">%s</text>'
            '<rect x="330" y="%d" width="%.1f" height="20" fill="#2f6fed" rx="3"/>'
            '<text x="%.1f" y="%d" font-size="12" font-family="Menlo,monospace" fill="#111">'
            '%.4f ± %.4f</text>' % (
                y + 15, _escape(label[:44]), y, max(2.0, bar), 338 + max(2.0, bar), y + 15,
                value, error))
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
            'viewBox="0 0 %d %d"><rect width="100%%" height="100%%" fill="#fff"/>'
            '<text x="8" y="30" font-size="16" font-weight="700" '
            'font-family="Helvetica,Arial">hidden proxy by arm (last round)</text>%s</svg>'
            % (width, height, width, height, "".join(bars)))


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="aggregate full-sim runs")
    parser.add_argument("--out", default=str(LAB))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    manifest = load_manifest()
    if not runs:
        print("[analyze] no runs found under", RUNS)
        return 1

    # Group by campaign tag prefix (e.g. C1-policy-on-s1 -> scenario + arm).
    groups: dict[str, list[dict]] = {}
    for item in runs:
        if not item["tag"]:
            continue  # pre-campaign ad-hoc runs are listed in the CSV only
        key = "%s|%s|%s" % (item["scenario"], item["tag"].split("-s")[0],
                            item["policy"])
        groups.setdefault(key, []).append(item)

    rows = []
    for item in runs:
        values = [task["hidden_proxy"] for task in item["tasks"].values()
                  if task["hidden_proxy"] is not None]
        row = {
            "run": item["name"],
            "scenario": item["scenario"],
            "tag": item["tag"],
            "policy": item["policy"],
            "pair_delta": item["pair_delta"] if item["pair_delta"] is not None else "",
            "rounds": item["rounds"],
            "tasks": ",".join(sorted(item["tasks"])),
            "hidden_mean": statistics.fmean(values) if values else "",
            "calls": sum(task["calls"] for task in item["tasks"].values()),
        }
        for key in sorted(item["tasks"]):
            row["%s_hidden" % key] = item["tasks"][key]["hidden_proxy"]
            row["%s_val_last" % key] = (item["tasks"][key]["val_curve"] or [None])[-1]
            row["%s_chosen_last" % key] = (item["tasks"][key]["chosen"] or [""])[-1]
        rows.append(row)

    csv_path = out / "analysis.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Per-arm summary (mean/std across seeds/repeats).
    summary_rows = []
    for key, items in groups.items():
        scenario, arm, policy = key.split("|", 2)
        task_values: dict[str, list[float]] = {}
        for item in items:
            for task_key, task in item["tasks"].items():
                if task["hidden_proxy"] is not None:
                    task_values.setdefault(task_key, []).append(task["hidden_proxy"])
        for task_key, values in sorted(task_values.items()):
            mean, std = _mean_std(values)
            summary_rows.append((scenario, arm, policy, task_key, len(values), mean, std))
    summary_rows.sort(key=lambda row: (row[0], row[1], row[3]))

    bar_rows = [(f"{scenario}/{arm}/{task}", mean, std)
                for scenario, arm, policy, task, _, mean, std in summary_rows]
    svg_path = out / "hidden_by_arm.svg"
    svg_path.write_text(svg_bars(bar_rows), encoding="utf-8")

    lines = ["# Full-sim analysis", "",
             "Generated from %d run summaries (%d manifest entries)." % (len(runs), len(manifest)),
             "", "## Per-arm hidden proxy (last round)", "",
             "| scenario | arm | task | n | hidden mean | std |", "|---|---|---|---|---|---|"]
    for scenario, arm, policy, task, count, mean, std in summary_rows:
        lines.append("| %s | %s | %s | %d | %.4f | %.4f |"
                     % (scenario, arm, task, count, mean, std))
    curves = []
    for item in runs:
        for task_key, task in item["tasks"].items():
            curve = task.get("hidden_curve") or []
            if len([value for value in curve if value is not None]) >= 2:
                curves.append((item["tag"], task_key, curve))
    if curves:
        lines += ["", "## Hidden learning curves", ""]
        for tag, task_key, curve in curves:
            points = " -> ".join("%.4f" % value if value is not None else "n/a" for value in curve)
            lines.append("- `%s` %s: %s" % (tag, task_key, points))
    lines += ["", "## Offline lab", ""]
    for name in ("sweep.json", "comparison_table.json"):
        path = LAB / "official" / name
        if path.is_file():
            lines.append("- %s: `%s`" % (name, path))
    markdown = "\n".join(lines) + "\n"
    (out / "analysis.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("[analyze] wrote", csv_path, svg_path, out / "analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
