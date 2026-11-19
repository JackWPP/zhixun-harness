#!/usr/bin/env python3
"""Predict-path robustness probe.

Feeds 50 deliberately malformed rows through the deployment harness
(``harness.py predict``) using a real selected artifact and checks that every
line still produces schema-valid JSON.  The participant is expected to degrade
per row (fallback / degenerate prediction), never to crash the stream.

Usage::

    python scripts/sim/anomaly.py --run .scratch/sim/runs/rsi-sim-official-C1-policy-on-s1-...
    python scripts/sim/anomaly.py --artifact path/to/checkpoint.npz --task A
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "docker" / "app"
SDK_PARENT = REPO / "vendor" / "contest_harness-1.0.4"
VALIDATION = REPO / "datasets" / "cpu-a-guide-20261110-v1" / "public" / "validation.jsonl"
CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def _rows(limit: int = 12) -> list[dict]:
    rows = []
    with VALIDATION.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def build_cases() -> list[dict]:
    base = _rows()
    cases: list[dict] = []
    for row in base:
        cases.append({"id": row["id"], "inputs": row["inputs"], "note": "valid"})
    template = base[0]
    features = list(template["inputs"].get("features", []))
    mutations = [
        ("missing inputs key", {"id": "bad-1", "inputs": {}}),
        ("inputs null coerced to {}", {"id": "bad-2", "inputs": {}}),
        ("missing features", {"id": "bad-3", "inputs": {"other": 1.0}}),
        ("extra key", {"id": "bad-4", "inputs": {"features": features, "extra": [1, 2]}}),
        ("null feature", {"id": "bad-5", "inputs": {"features": [None] * len(features)}}),
        ("string numbers", {"id": "bad-6",
                            "inputs": {"features": [str(value) for value in features]}}),
        ("short features", {"id": "bad-7", "inputs": {"features": features[:3]}}),
        ("long features", {"id": "bad-8", "inputs": {"features": features + [0.0] * 5}}),
        ("nested dict", {"id": "bad-9", "inputs": {"features": {"0": 1.0}}}),
        ("empty inputs", {"id": "bad-10", "inputs": {}}),
        ("nan string", {"id": "bad-11",
                        "inputs": {"features": ["nan"] * len(features)}}),
        ("bool features", {"id": "bad-12",
                           "inputs": {"features": [True] * len(features)}}),
    ]
    for _ in range(4):
        for label, row in mutations:
            case = dict(row)
            case["id"] = "%s-%d" % (case["id"], len(cases))
            case["note"] = label
            cases.append(case)
    return cases[:50]


def resolve_artifact(run: str | None) -> tuple[Path, str]:
    if run:
        root = Path(run)
        if root.is_file():
            return root, "A"
        candidates = sorted(root.glob("tasks/A/artifacts/*/checkpoint.npz"))
        if not candidates:
            raise SystemExit("no A artifact under %s" % root)
        return candidates[-1], "A"
    runs = sorted((REPO / ".scratch" / "sim" / "runs").glob("rsi-sim-official-*/tasks/A/artifacts/*/checkpoint.npz"))
    if not runs:
        raise SystemExit("no artifact found; pass --run or --artifact")
    return runs[-1], "A"


def validate_line(task: str, payload: dict, classes: list[str]) -> str:
    if not isinstance(payload, dict) or "outputs" not in payload:
        return "missing outputs"
    if task == "classification":
        outputs = payload["outputs"]
        if not isinstance(outputs, list) or not outputs:
            return "empty outputs"
        labels, scores = [], []
        for item in outputs:
            label = str(item.get("label", ""))
            score = item.get("score")
            if not label:
                return "empty label"
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                return "bad score"
            if not 0.0 <= float(score) <= 1.0:
                return "score out of range"
            labels.append(label)
            scores.append(float(score))
        if len(set(labels)) != len(labels):
            return "duplicate labels"
        top = max(scores)
        if scores.count(top) != 1:
            return "tied top score"
        return ""
    outputs = payload["outputs"]
    if not isinstance(outputs, dict) or "value" not in outputs:
        return "missing value"
    value = outputs["value"]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "bad value"
    if float(value) < 0.0:
        return "negative value"
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="malformed predict-row robustness probe")
    parser.add_argument("--run", default="")
    parser.add_argument("--artifact", default="")
    parser.add_argument("--task", default="A")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    artifact, task_key = (Path(args.artifact), args.task) if args.artifact else resolve_artifact(args.run)
    artifact = artifact.resolve()
    if not artifact.is_file():
        raise SystemExit("artifact not found: %s" % artifact)
    classes = CLASSES if task_key == "A" else []
    cases = build_cases()
    work = Path(tempfile.mkdtemp(prefix="anomaly-"))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(SDK_PARENT), str(APP)])
    env["CONTEST_ARTIFACT_PATH"] = str(artifact)
    env["CONTEST_ARTIFACT_MANIFEST"] = str(artifact.with_name("submission_meta.json"))
    env["CONTEST_PREDICTION_CONTEXT"] = json.dumps({
        "protocol": "contest.predict-context.v1", "dataset_key": task_key,
        "task": "classification" if task_key == "A" else "regression",
        "artifact": {"checkpoint_paths": ["checkpoint.npz"],
                     "code_paths": ["submission_meta.json"]},
        "artifact_id": "anomaly-probe",
    })
    payload = "".join(json.dumps({"task": "classification" if task_key == "A" else "regression",
                                  "inputs": case.get("inputs") or {}}, ensure_ascii=False) + "\n"
                      for case in cases)
    process = subprocess.run([sys.executable, str(APP / "harness.py"), "predict"],
                             input=payload, text=True, capture_output=True,
                             cwd=str(work), env=env, timeout=args.timeout)
    print("[anomaly] exit=%d lines_out=%d" % (process.returncode,
                                              len(process.stdout.splitlines())))
    if process.stderr.strip():
        print("[anomaly] stderr tail:", process.stderr.strip()[-1500:], file=sys.stderr)
    failures = []
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            failures.append((index, "invalid JSON: %s" % error))
            continue
        problem = validate_line("classification" if task_key == "A" else "regression",
                                parsed, classes)
        if problem:
            failures.append((index, problem))
    covered = min(len(cases), len(lines))
    if len(lines) < len(cases):
        failures.append((len(lines), "missing output lines (%d/%d)" % (len(lines), len(cases))))
    report = {"artifact": str(artifact), "cases": len(cases), "outputs": len(lines),
              "invalid": len(failures), "failures": failures[:10],
              "stderr_tail": process.stderr.strip()[-500:]}
    out = REPO / ".scratch" / "sim" / "lab" / "anomaly.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[anomaly] valid=%d invalid=%d -> %s" %
          (covered - len(failures) if covered >= len(failures) else 0, len(failures), out))
    for index, problem in failures[:10]:
        print("  line %d: %s" % (index, problem))
    return 0 if process.returncode == 0 and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
