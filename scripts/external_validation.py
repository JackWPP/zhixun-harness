#!/usr/bin/env python3
"""Reproducible external validation on the public UCI gas drift archive (task A).

The official task-A package only ships two batches.  This script turns the other
eight public batches (1-10) into JSONL probes and evaluates a checkpoint on each
of them, producing a markdown table and a JSON file.  It is evidence only:
Guard forbids using these files for training, and the checkpoint is frozen while
the probes are read.

    ./.venv/bin/python scripts/external_validation.py \
        --checkpoint datasets/cpu-a-guide-20261110-v1/checkpoint.npz \
        --postprocess .scratch/rsi-canonical/<run>/summary.json \
        --output docs/evidence/uci_gas_drift.md
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsi.adapters import evaluate_private, evaluate_private_file  # noqa: E402
from rsi.data import read_uci_gas_dat  # noqa: E402
from rsi.tasks import resolve_tasks  # noqa: E402

CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def batch_to_jsonl(dat_path: Path, output: Path) -> Path:
    dataset = read_uci_gas_dat(dat_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, (row, label) in enumerate(zip(dataset.x, dataset.y)):
            handle.write(json.dumps({
                "id": f"{dat_path.stem}_{index:06d}",
                "inputs": {"features": [float(value) for value in row]},
                "answer": CLASSES[int(label)],
            }) + "\n")
    return output


def postprocess_from(path: str) -> dict:
    if not path:
        return {"type": "none"}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "postprocess" in payload:
        return payload["postprocess"]
    if isinstance(payload, dict) and "datasets" in payload:
        for dataset in payload["datasets"]:
            if dataset.get("key") == "A":
                return dataset.get("postprocess", {"type": "none"})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gas-root", default="extra/uci-gas/Dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", default="datasets/cpu-a-guide-20261110-v1/checkpoint.npz")
    parser.add_argument("--postprocess", default="", help="summary.json or postprocess JSON")
    parser.add_argument("--output", default="docs/evidence/uci_gas_drift.md")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    config = resolve_tasks(["A"])[0]
    dat_files = sorted(Path(args.gas_root).glob("batch*.dat"),
                       key=lambda path: int(path.stem.replace("batch", "")))
    if not dat_files:
        raise SystemExit(f"no batch*.dat under {args.gas_root}")
    postprocess = postprocess_from(args.postprocess)

    with tempfile.TemporaryDirectory(prefix="uci-probe-") as tmp:
        rows = []
        for dat in dat_files:
            jsonl = batch_to_jsonl(dat, Path(tmp) / f"{dat.stem}.jsonl")
            base = evaluate_private(config, jsonl, args.baseline_checkpoint, {"type": "none"})
            final = evaluate_private(config, jsonl, args.checkpoint, postprocess)
            rows.append({
                "batch": dat.stem,
                "rows": base["dataset"]["rows"],
                "baseline_macro_f1": base["primary"],
                "final_macro_f1": final["primary"],
                "delta": final["primary"] - base["primary"],
            })
        combined_base = sum(row["baseline_macro_f1"] * row["rows"] for row in rows) / sum(row["rows"] for row in rows)
        combined_final = sum(row["final_macro_f1"] * row["rows"] for row in rows) / sum(row["rows"] for row in rows)

    lines = [
        "# UCI Gas Sensor Drift — 外部独立验证（扩展数据，非训练）",
        "",
        "官方 task A 仅含 batch1/batch2。本表用 UCI 公开 archive 的 batch1–batch10 逐批评估；",
        "扩展数据只读、不参与训练（Guard 白名单排除），对应赛题 extended_data_validation 证据。",
        "",
        f"- baseline checkpoint: `{args.baseline_checkpoint}`",
        f"- final checkpoint: `{args.checkpoint}`",
        f"- postprocess: `{json.dumps(postprocess, ensure_ascii=False)}`",
        "",
        "| batch | rows | baseline macro-F1 | final macro-F1 | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['batch']} | {row['rows']} | {row['baseline_macro_f1']:.6f} | "
                     f"{row['final_macro_f1']:.6f} | {row['delta']:+.6f} |")
    lines += ["", f"**micro weighted total**: baseline {combined_base:.6f} → final {combined_final:.6f} "
                  f"({combined_final - combined_base:+.6f})", ""]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps({"rows": rows, "combined": {
            "baseline_macro_f1": combined_base, "final_macro_f1": combined_final},
            "postprocess": postprocess}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
