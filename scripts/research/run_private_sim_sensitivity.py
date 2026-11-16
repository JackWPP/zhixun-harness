"""Sensitivity experiments E3-E6 (A) and official honest replay E9."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsi.stream.runner import run_stream  # noqa: E402

OUT = Path(".scratch/ps-results/sensitivity.json")
ROOT = Path(".scratch/ps")

CASES = [
    ("E3a_label_every2", "A", "private-sim", "adapt", {"label_every": 2}),
    ("E3b_label_fraction50", "A", "private-sim", "adapt", {"label_fraction": 0.5}),
    ("E3c_scarce_both", "A", "private-sim", "adapt", {"label_every": 2, "label_fraction": 0.5}),
    ("E4_missing_aa", "A", "private-sim", "adapt", {"missing_class": "Acetaldehyde"}),
    ("E5_gain", "A", "private-sim", "adapt", {"perturb": "gain"}),
    ("E5_offset", "A", "private-sim", "adapt", {"perturb": "offset"}),
    ("E5_noise", "A", "private-sim", "adapt", {"perturb": "noise"}),
    ("E5_gain_offset", "A", "private-sim", "adapt", {"perturb": "gain_offset"}),
    ("E6_repeat2", "A", "private-sim", "adapt", {"repeat": 2}),
    ("E9_official_frozen", "A", "official", "frozen", {"n_batches": 8}),
    ("E9_official_pair", "A", "official", "pair", {"n_batches": 8}),
    ("E9_official_adapt", "A", "official", "adapt", {"n_batches": 8}),
    ("E9_official_full", "A", "official", "full", {"n_batches": 8, "experiments_cap": 1}),
]


def _row(tag, task, source, arm, report, elapsed):
    data = report["tasks"][task]
    final = data.get("final") or {}
    return {
        "tag": tag, "task": task, "source": source, "arm": arm,
        "arrival": [round((b.get("prediction_metrics") or {}).get("primary") or 0.0, 6)
                    for b in data.get("batches", [])],
        "hidden": [(b["batch"], round((b.get("prediction_metrics") or {}).get("primary") or 0.0, 6))
                   for b in data.get("hidden", [])],
        "hidden_mean": final.get("hidden_mean_primary"),
        "mean_primary": final.get("mean_primary"),
        "history_rows": final.get("history_rows"),
        "errors": data.get("errors", []),
        "elapsed": elapsed,
    }


def main() -> int:
    results = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": {}}
    if OUT.exists():
        try:
            results = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            pass
    for index, (tag, task, source, arm, extra) in enumerate(CASES):
        if tag in results["rows"] and not results["rows"][tag].get("errors"):
            print(f"[sens] {tag} already done", flush=True)
            continue
        n_batches = int(extra.pop("n_batches", 5))
        print(f"[sens] {tag} -> {task}/{source}/{arm} {extra}", flush=True)
        started = time.time()
        state_dir = ROOT / f"sens_{index:02d}_{tag}"
        report = run_stream(keys=[task], source=source, n_batches=n_batches,
                            state_dir=str(state_dir), arm=arm, seed=42,
                            controller="pipeline", llm_mode="off", memory=False,
                            experiments_cap=int(extra.pop("experiments_cap", 1)),
                            retention_every=2, **extra)
        elapsed = round(time.time() - started, 1)
        results["rows"][tag] = _row(tag, task, source, arm, report, elapsed)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2))
        row = results["rows"][tag]
        print(f"[sens] {tag} arrival={row['arrival']} hidden={row['hidden']} "
              f"hidden_mean={row['hidden_mean']} wall={elapsed}s", flush=True)
    print(json.dumps(results["rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
