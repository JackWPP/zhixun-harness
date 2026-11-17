#!/usr/bin/env python3
"""Prepare and sanity-check the full-dataset batch streams (A/B/C).

Usage:
    PYTHONPATH=. python3 scripts/sim/prepare_full_batches.py --batches 10 \
        --out .scratch/sim/batches_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsi.stream.sources import full_batches
from rsi.tasks import resolve_tasks


def audit(config, batches, expected_rows: int) -> dict:
    problems: list[str] = []
    seen_pred: set[str] = set()
    per_batch = []
    revealed_rows = 0
    hidden_rows = 0
    for batch in batches:
        ids = {str(r.get("id")) for r in batch.prediction}
        if len(ids) != len(batch.prediction):
            problems.append(f"{config.key} batch {batch.index}: duplicate prediction ids")
        if ids & seen_pred:
            problems.append(f"{config.key} batch {batch.index}: prediction rows repeated")
        seen_pred |= ids
        if ids & {str(r.get("id")) for r in batch.train}:
            problems.append(f"{config.key} batch {batch.index}: current batch leaked into train")
        if ids & {str(r.get("id")) for r in batch.dev}:
            problems.append(f"{config.key} batch {batch.index}: current batch leaked into dev")
        if any("answer" in r for r in batch.prediction):
            problems.append(f"{config.key} batch {batch.index}: prediction rows contain labels")
        if any("answer" not in r for r in batch.train):
            problems.append(f"{config.key} batch {batch.index}: train rows missing labels")
        if any("answer" not in r for r in batch.dev):
            problems.append(f"{config.key} batch {batch.index}: dev rows missing labels")
        if len(set(batch.answers)) != len(batch.answers):
            problems.append(f"{config.key} batch {batch.index}: duplicate answer ids")
        hidden = bool(batch.meta.get("hidden"))
        if hidden and batch.reveal:
            problems.append(f"{config.key} batch {batch.index}: hidden batch reveals rows")
        if hidden:
            hidden_rows += len(batch.prediction)
        else:
            revealed_rows += len(batch.reveal)
        per_batch.append({
            "index": int(batch.index),
            "phase": batch.meta.get("phase", "arrival"),
            "batch_no": batch.meta.get("batch"),
            "hidden": hidden,
            "train_rows": len(batch.train),
            "dev_rows": len(batch.dev),
            "prediction_rows": len(batch.prediction),
            "reveal_rows": len(batch.reveal),
            "history_rows": len(batch.train) + len(batch.dev),
        })
    if len(seen_pred) != expected_rows:
        problems.append(
            f"{config.key}: streamed {len(seen_pred)} rows, expected {expected_rows}"
        )
    return {
        "key": config.key,
        "task": config.task,
        "n_batches": len(batches),
        "expected_rows": expected_rows,
        "streamed_rows": len(seen_pred),
        "revealed_rows": revealed_rows,
        "hidden_rows": hidden_rows,
        "problems": problems,
        "batches": per_batch,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="A,B,C")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--hidden-tail", type=int, default=2)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perturb", default="none",
                        choices=["none", "gain", "offset", "noise", "gain_offset"])
    parser.add_argument("--out", default=".scratch/sim/batches_manifest.json")
    args = parser.parse_args(argv)

    keys = [k.strip().upper() for k in args.tasks.split(",") if k.strip()]
    configs = resolve_tasks(keys)
    report = {"batches": args.batches, "hidden_tail": args.hidden_tail,
              "dev_fraction": args.dev_fraction, "seed": args.seed,
              "perturb": args.perturb, "tasks": {}}
    failed = 0
    for config in configs:
        batches = full_batches(
            config, n_batches=args.batches, dev_fraction=args.dev_fraction,
            seed=args.seed, hidden_tail=args.hidden_tail, perturb=args.perturb,
        )
        if config.key == "A":
            expected = sum(len(b.prediction) for b in batches) + 57  # UCI probe rows are all streamed
            expected = sum(len(b.prediction) for b in batches)
        else:
            expected = len([1 for _ in open(config.validation_path, encoding="utf-8")])
        audit_result = audit(config, batches, expected)
        report["tasks"][config.key] = audit_result
        failed += len(audit_result["problems"])
        print(f"[{config.key}] batches={len(batches)} rows={audit_result['streamed_rows']} "
              f"revealed={audit_result['revealed_rows']} hidden={audit_result['hidden_rows']} "
              f"problems={len(audit_result['problems'])}")
        for row in audit_result["batches"]:
            print(f"    b{row['index']:02d} {row['phase']:<7} hidden={int(row['hidden'])} "
                  f"train={row['train_rows']:>6} dev={row['dev_rows']:>5} "
                  f"pred={row['prediction_rows']:>6} reveal={row['reveal_rows']:>6}")
        for problem in audit_result["problems"][:10]:
            print(f"    ! {problem}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {out}")
    print("PASS" if failed == 0 else f"FAIL ({failed} problems)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
