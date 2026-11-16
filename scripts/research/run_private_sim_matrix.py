"""Run the honest private-sim arm matrix (A/B/C) and print a comparison table."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsi.stream.runner import run_stream  # noqa: E402

ROOT = Path(".scratch/ps")
OUT = Path(".scratch/ps-results/matrix.json")

MATRIX = {
    "A": {"arms": ["frozen", "recenter", "pair", "adapt", "full"], "seeds": [42, 43, 44],
          "n_batches": 5, "heavy": ["adapt", "full"]},
    "B": {"arms": ["frozen", "recenter", "adapt", "full"], "seeds": [42], "n_batches": 8},
    "C": {"arms": ["frozen", "recenter", "adapt", "full"], "seeds": [42], "n_batches": 8},
}


def _summary(task: str, arm: str, seed: int, report: dict) -> dict:
    data = report["tasks"][task]
    final = data.get("final") or {}
    arrival = [round((b.get("prediction_metrics") or {}).get("primary") or 0.0, 6)
               for b in data.get("batches", [])]
    hidden = [(b["batch"], round((b.get("prediction_metrics") or {}).get("primary") or 0.0, 6))
              for b in data.get("hidden", [])]
    return {"task": task, "arm": arm, "seed": seed, "arrival": arrival,
            "hidden": hidden, "hidden_mean": round(sum(v for _, v in hidden) / len(hidden), 6) if hidden else None,
            "mean_primary": final.get("mean_primary"), "history_rows": final.get("history_rows"),
            "errors": data.get("errors", []), "wall_seconds": final.get("wall_seconds")}


def main() -> int:
    results = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": {}}
    if OUT.exists():
        try:
            results = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            pass
    for task, cfg in MATRIX.items():
        arms = cfg["arms"]
        seeds = cfg["seeds"]
        for arm in arms:
            use_seeds = seeds
            if task == "A" and arm in cfg.get("heavy", []) and len(seeds) > 2:
                use_seeds = seeds[:2]
            if task == "A" and arm not in cfg.get("heavy", []) and len(seeds) > 2:
                use_seeds = [seeds[0]]
            for seed in use_seeds:
                tag = f"{task}/{arm}/{seed}"
                state_dir = ROOT / f"{task}_{arm}_{seed}"
                print(f"[matrix] {tag} -> {state_dir}", flush=True)
                started = time.time()
                report = run_stream(keys=[task], source="private-sim", n_batches=cfg["n_batches"],
                                    state_dir=str(state_dir), arm=arm, seed=seed,
                                    controller="pipeline", llm_mode="off", memory=False,
                                    experiments_cap=1, retention_every=2)
                row = _summary(task, arm, seed, report)
                row["elapsed"] = round(time.time() - started, 1)
                results["rows"][tag] = row
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(results, indent=2))
                print(f"[matrix] {tag} hidden={row['hidden']} mean={row['mean_primary']} "
                      f"wall={row['elapsed']}s", flush=True)
    print(json.dumps(results["rows"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
