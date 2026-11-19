#!/usr/bin/env python3
"""Summarize the P3 (E7-E12) campaign runs into a Markdown table.

The ``summary.json`` files store a per-round hidden proxy; for drift campaigns
the meaningful number is the mean over drops, not just the last one.  This
script groups runs by arm tag (``E7-gate-on-s1`` -> ``E7-gate-on``) and reports
mean±std across seeds for both the mean-over-drops and the last drop.

Usage::

    python scripts/sim/summarize_p3.py
    python scripts/sim/summarize_p3.py --write docs/39_SPRINT_0.14.0.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / ".scratch" / "sim" / "runs"


def _arm(name: str) -> str:
    return re.sub(r"-s\d+$", "", name)


_LATEST: dict[str, str] = {}


def load() -> list[dict]:
    best: dict[str, tuple[str, dict]] = {}
    for path in sorted(RUNS.glob("*/summary.json")):
        name = path.parent.name
        match = re.match(r"rsi-sim-(?:official-seq|official|external|schema)-(.+?)-\d{8}-\d{6}$", name)
        if not match:
            continue
        tag = match.group(1)
        if not re.match(r"E(?:7|8|9|10|11|12)-", tag):
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Only the newest run per tag counts: interrupted/retried attempts
        # leave older run dirs behind and must not double-count poles.
        stamp = name.rsplit("-", 2)[-2] + "-" + name.rsplit("-", 1)[-1]
        if tag not in best or stamp > best[tag][0]:
            best[tag] = (stamp, summary)
    rows = []
    for tag, (_, summary) in sorted(best.items()):
        for task_key, value in (summary.get("tasks") or {}).items():
            rounds = value.get("rounds") or []
            hidden = [entry.get("hidden_proxy") for entry in rounds
                      if entry.get("hidden_proxy") is not None]
            if not hidden:
                continue
            chosen = []
            for entry in rounds:
                choice = entry.get("chosen")
                if isinstance(choice, dict):
                    chosen.append(str(choice.get("kind")))
                elif choice:
                    chosen.append(str(choice))
            rows.append({
                "arm": _arm(tag),
                "tag": tag,
                "task": task_key,
                "mean_hidden": statistics.fmean(hidden),
                "last_hidden": hidden[-1],
                "rounds": len(hidden),
                "calls": sum(int(entry.get("calls") or 0) for entry in rounds),
                "chosen": chosen,
            })
    return rows


def _fmt(values: list[float]) -> str:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return "n/a"
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return "%.4f ± %.4f" % (mean, std)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", default="", help="optional Markdown file to write")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args(argv)
    rows = load()
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["arm"], []).append(row)
    lines = ["| arm | task | poles | mean hidden (mean±sd) | last drop | calls | chosen arms |",
             "|---|---|---|---|---|---|---|"]
    for arm in sorted(groups):
        items = groups[arm]
        task = items[0]["task"]
        means = [item["mean_hidden"] for item in items]
        lasts = [item["last_hidden"] for item in items]
        calls = sum(item["calls"] for item in items)
        chosen: list[str] = []
        for item in items:
            for kind in item["chosen"]:
                if kind not in chosen:
                    chosen.append(kind)
        lines.append("| %s | %s | %d | %s | %s | %d | %s |" % (
            arm, task, len(items), _fmt(means), _fmt(lasts), calls,
            ", ".join(chosen) or "-"))
    table = "\n".join(lines)
    print(table)
    if args.write:
        path = REPO / args.write
        mode = "a" if args.append else "w"
        with path.open(mode, encoding="utf-8") as handle:
            if args.append:
                handle.write("\n" + table + "\n")
            else:
                handle.write(table + "\n")
        print("[summarize] wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
