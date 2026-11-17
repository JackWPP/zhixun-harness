#!/usr/bin/env python3
"""L3 kernel-LLM batch rehearsal (unlimited token budget).

Runs a small number of full-data batches through the kernel controller so the
LLM agent has to observe drift, propose experiments, execute them and record a
memory entry between batches. Budgets are deliberately loose (token cap 0).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_l1():
    path = ROOT / "scripts" / "sim" / "run_l1_full.py"
    spec = importlib.util.spec_from_file_location("sim_l1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    l1 = _load_l1()
    plan = (("A", 2), ("B", 1), ("C", 1))
    for task, batches in plan:
        print(f"\n===== L3 kernel task {task} batches={batches} =====", flush=True)
        code = l1.main([
            "--tasks", task,
            "--batches", str(batches),
            "--jobs", "1",
            "--seeds", "42",
            "--arms-A", "full", "--arms-B", "full", "--arms-C", "full",
            "--root", ".scratch/sim/L3",
            "--seed-tag", f"kernel-{task}",
            "--llm", "on",
            "--controller", "kernel",
            "--llm-env", ".scratch/llm.env",
            "--token-cap", "0",
            "--round-turns", "16",
            "--experiments-cap", "8",
            "--batch-wall", "1800",
            "--lane-timeout", "10800",
            "--retention-every", "0",
        ])
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
