"""Rerun adapt/full arms with the dev-validated adaptation gate enabled.

Compares the gated arms against the already-recorded ungated matrix
(``.scratch/ps-results/matrix.json``).  The gate should keep the large gains on
the external UCI stream while rolling back the harmful adaptations on the
official per-chunk stream.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsi.stream.runner import run_stream  # noqa: E402

ROOT = Path(".scratch/ps-gated")
OUT = Path(".scratch/ps-results/gated.json")

CASES = [
    # task, source, arm, seed, n_batches
    ("A", "private-sim", "adapt", 42, 5),
    ("A", "private-sim", "adapt", 43, 5),
    ("A", "private-sim", "full", 42, 5),
    ("B", "private-sim", "adapt", 42, 8),
    ("B", "private-sim", "full", 42, 8),
    ("C", "private-sim", "adapt", 42, 8),
    ("A", "official", "adapt", 42, 8),
    ("A", "official", "full", 42, 8),
]


def _summary(task: str, source: str, arm: str, seed: int, report: dict) -> dict:
    data = report["tasks"][task]
    final = data.get("final") or {}
    arrival = [round((b.get("prediction_metrics") or {}).get("primary") or 0.0, 6)
               for b in data.get("batches", [])]
    hidden = [(b["batch"], round((b.get("prediction_metrics") or {}).get("primary") or 0.0, 6))
              for b in data.get("hidden", [])]
    gates = [b.get("adapt_gate") for b in data.get("batches", []) if b.get("adapt_gate")]
    kept = sum(1 for g in gates if g.get("kept"))
    return {
        "task": task,
        "source": source,
        "arm": arm,
        "seed": seed,
        "arrival": arrival,
        "hidden": hidden,
        "hidden_mean": round(sum(v for _, v in hidden) / len(hidden), 6) if hidden else None,
        "mean_primary": final.get("mean_primary"),
        "history_rows": final.get("history_rows"),
        "errors": data.get("errors", []),
        "wall_seconds": final.get("wall_seconds"),
        "gate_attempted": len(gates),
        "gate_kept": kept,
    }


def main() -> int:
    results = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": {}}
    if OUT.exists():
        try:
            results = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            pass
    for task, source, arm, seed, n_batches in CASES:
        tag = f"{task}/{source}/{arm}/{seed}"
        if tag in results["rows"]:
            print(f"[gated] skip {tag} (already recorded)", flush=True)
            continue
        state_dir = ROOT / f"{task}_{source}_{arm}_{seed}"
        print(f"[gated] {tag} -> {state_dir}", flush=True)
        started = time.time()
        kwargs = {}
        if source == "private-sim" and task == "A":
            kwargs.update(arrival_start=3, arrival_end=7, hidden_start=8, hidden_end=10)
        report = run_stream(keys=[task], source=source, n_batches=n_batches,
                            state_dir=str(state_dir), arm=arm, seed=seed,
                            controller="pipeline", llm_mode="off", memory=False,
                            experiments_cap=1, retention_every=2, **kwargs)
        row = _summary(task, source, arm, seed, report)
        row["elapsed"] = round(time.time() - started, 1)
        results["rows"][tag] = row
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[gated] {tag} hidden={row['hidden']} hidden_mean={row['hidden_mean']} "
              f"gate kept={row['gate_kept']}/{row['gate_attempted']} wall={row['elapsed']}s",
              flush=True)
    print(json.dumps(results["rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
