#!/usr/bin/env python3
"""Track X four-arm evidence exporter.

Reads a finished batch-stream run (``rsi.cli stream``) and builds the four-arm
comparison used in docs/12:

1. ``frozen``       – official baseline checkpoint, no adaptation (per batch);
2. ``recenter``     – frozen checkpoint + train-free batch re-centering;
3. ``full_harness`` – the model actually deployed after the batch decision;
4. ``oracle``       – retrain from the parent on all labelled rows seen so far
                      (upper bound; uses batch labels, testbed only).

The oracle arm is intentionally expensive and is only run when ``--oracle`` is
passed. Nothing here is used for the scored submission path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rsi.data import load_jsonl_dataset
from rsi.inference import apply_postprocess, resolve_postprocess
from rsi.metrics import from_predictions, primary
from rsi.model import ExtendedModel, checkpoint_sha256
from rsi.strategies import ExperimentContext, run_strategy


def _eval(model, dataset, postprocess=None):
    predicted = model.predict(dataset.x)
    if postprocess:
        predicted = apply_postprocess(postprocess, model, dataset.x)
    metrics = from_predictions(task_mode, predicted, dataset.y, classes)
    metrics["primary"] = primary(task_mode, metrics)
    metrics["rows"] = len(dataset)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-dir", required=True, help="stream state dir (tasks/<key>/batch_*)")
    ap.add_argument("--task", default="A")
    ap.add_argument("--oracle", action="store_true", help="also run oracle retrain (slow)")
    ap.add_argument("--oracle-epochs", type=int, default=100)
    ap.add_argument("--oracle-lr", type=float, default=1e-3)
    ap.add_argument("--out", default="docs/evidence/testbed_arms.md")
    args = ap.parse_args()

    global task_mode, classes
    from rsi.tasks import resolve_tasks

    config = resolve_tasks([args.task])[0]
    task_mode, classes = config.task, config.classes
    task_root = Path(args.stream_dir) / "tasks" / args.task
    batch_dirs = sorted(p for p in task_root.glob("batch_*") if p.is_dir())
    if not batch_dirs:
        raise SystemExit(f"no batches under {task_root}")

    rows = []
    for batch_dir in batch_dirs:
        index = int(batch_dir.name.split("_")[1])
        package = batch_dir / "package"
        validation = load_jsonl_dataset(package / "public" / "validation.jsonl",
                                        config.task, config.classes)
        train = load_jsonl_dataset(package / "public" / "train.jsonl",
                                   config.task, config.classes)
        decision = json.loads((batch_dir / "decision.json").read_text())

        frozen_ckpt = decision["parent"]["checkpoint"]
        frozen = ExtendedModel.from_checkpoint(frozen_ckpt)
        frozen_metrics = _eval(frozen, validation)

        recenter_spec = resolve_postprocess({"type": "recenter", "mode": "batch_mean", "weight": 0.4},
                                            frozen)
        recenter_metrics = _eval(frozen, validation, recenter_spec)

        chosen_ckpt = decision["chosen"]["checkpoint"]
        chosen_post = decision["chosen"].get("postprocess")
        chosen = ExtendedModel.from_checkpoint(chosen_ckpt)
        harness_metrics = _eval(chosen, validation, chosen_post)

        oracle_metrics = None
        if args.oracle:
            ctx = ExperimentContext(
                task_key=config.key, task=config.task, parent=frozen,
                train=train, calibration=validation, selection=validation,
                report=validation, seed=42, classes=tuple(classes),
            )
            params = {"epochs": args.oracle_epochs, "data": "train",
                      "learning_rate": args.oracle_lr, "class_balance": True}
            try:
                candidate = run_strategy(ctx, "continue_train", params)
                oracle_metrics = _eval(candidate.model, validation)
            except Exception as exc:  # oracle is best-effort evidence
                oracle_metrics = {"error": f"{exc.__class__.__name__}: {exc}"}

        rows.append({
            "batch": index,
            "frozen": frozen_metrics,
            "recenter": recenter_metrics,
            "full_harness": harness_metrics,
            "oracle": oracle_metrics,
            "decision": decision["decision"],
            "parent_sha": decision["parent_checkpoint_sha256"],
            "chosen_sha": decision["checkpoint_sha256"],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    def fmt(value):
        return "–" if value is None else f"{value:.4f}"

    lines = [f"# Track X four-arm evidence — task {args.task}", "",
             f"source: `{args.stream_dir}`  ·  batches: {len(rows)}  ·  "
             f"primary = {config.primary_metric}", "",
             "| batch | frozen | recenter | full harness | oracle | decision |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| {b} | {f} | {r} | {h} | {o} | {d} |".format(
            b=row["batch"], f=fmt(row["frozen"]["primary"]),
            r=fmt(row["recenter"]["primary"]), h=fmt(row["full_harness"]["primary"]),
            o=fmt((row["oracle"] or {}).get("primary")), d=row["decision"],
        ))
    out.write_text("\n".join(lines) + "\n")
    print(json.dumps({"batches": len(rows), "markdown": str(out), "json": str(json_out)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
