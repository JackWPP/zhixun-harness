"""Summarize the PyTRiO kernel-adaptation campaign (K1/K2) from run summaries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / ".scratch" / "sim" / "runs"


def load_runs(prefix: str) -> list[dict]:
    rows = []
    for path in sorted(RUNS.glob("rsi-sim-official-seq-%s-*" % prefix)):
        summary = path / "summary.json"
        if not summary.is_file():
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        rows.append({"dir": path.name, **data})
    return rows


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "K1"
    rows = load_runs(prefix)
    if not rows:
        print("no runs for prefix", prefix)
        return 1
    out = []
    for row in rows:
        tag = row.get("tag") or row["dir"]
        provider = row.get("provider_usage") or {}
        for key, task in (row.get("tasks") or {}).items():
            rounds = task.get("rounds") or []
            hidden = [float(r.get("hidden_proxy")) for r in rounds
                      if r.get("hidden_proxy") is not None]
            calls = [int(r.get("calls") or 0) for r in rounds]
            walls = [float(r.get("wall_s") or 0) for r in rounds]
            chosen = [str(r.get("chosen")) for r in rounds]
            out.append({
                "tag": tag, "task": key, "rounds": len(rounds),
                "hidden_mean": round(sum(hidden) / len(hidden), 4) if hidden else None,
                "hidden_max": round(max(hidden), 4) if hidden else None,
                "calls": sum(calls), "calls_mean": round(sum(calls) / len(calls), 2) if calls else 0,
                "wall_s": round(sum(walls), 1),
                "chosen": sorted(set(chosen)),
                "prompt_tokens": provider.get("prompt_tokens"),
                "completion_tokens": provider.get("completion_tokens"),
            })
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
