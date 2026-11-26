"""Measure three serial workspace-v2 C rounds on complete public projections.

Use one Python environment with NumPy 1.26.4 for all variants. The baseline
must be upstream 17cc307b. Each emulator starts with fresh state and the same
official parent. It exercises the local fake gateway, upload/finish and a
three-row prediction protocol smoke check. This is not a platform benchmark.
Local repository, interpreter and temporary run paths are never serialized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


BASELINE_COMMIT = "17cc307b10caed360405d739bdfb09f457910bad"
THREAD_NAMES = (
    "BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
PACKAGE = Path("datasets/cpu-c-guide-20261110-v1")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(repository: Path) -> dict:
    emulator = repository / "scripts/emulate_workspace_v2.py"
    sdk = re.search(r'contest_harness-(\d+\.\d+\.\d+)', emulator.read_text())
    if sdk is None:
        raise RuntimeError("Cannot identify emulator SDK version")
    files = {emulator}
    for relative in ("rsi", "scripts/sim", "docker/app", str(PACKAGE / "baselines"),
                     f"vendor/contest_harness-{sdk.group(1)}"):
        files.update((repository / relative).rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(repository).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return {"source_sha256": digest.hexdigest(), "sdk_version": sdk.group(1)}


def data_identity(repository: Path) -> dict:
    result = {}
    for name in ("public/train.jsonl", "public/validation.jsonl", "checkpoint.npz"):
        path = repository / PACKAGE / name
        record = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        if path.suffix == ".jsonl":
            with path.open("rb") as handle:
                record["rows"] = sum(bool(line.strip()) for line in handle)
        result[name] = record
    return result


def run_variant(label: str, repository: Path, environment: dict, inputs: dict) -> dict:
    runtime = json.loads(subprocess.check_output(
        [sys.executable, "-c",
         "import json,platform,numpy; print(json.dumps({"
         "'python':platform.python_version(), 'numpy':numpy.__version__,"
         "'system':platform.system(), 'machine':platform.machine()}))"],
        cwd=repository, env=environment, text=True,
    ))
    if runtime["numpy"] != "1.26.4":
        raise RuntimeError("This comparison requires NumPy 1.26.4")
    identity = source_identity(repository)
    print(json.dumps({"starting_variant": label}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    process = subprocess.run(
        [sys.executable, str(repository / "scripts/emulate_workspace_v2.py"),
         "--task", "C", "--rounds", "1", "--json"],
        cwd=repository, env=environment, capture_output=True, text=True, timeout=300,
    )
    elapsed = time.perf_counter() - started
    if process.returncode:
        # Emulator output contains local runtime paths; do not publish it.
        raise RuntimeError(f"Variant {label} failed with exit code {process.returncode}")
    report = json.loads(process.stdout)
    if not report.get("ok") or len(report["rounds"]) != 1:
        raise RuntimeError(f"Variant {label} did not complete one protocol round")
    run_directory = Path(report["root"])
    metadata = json.loads((run_directory / "artifact-01/submission_meta.json").read_text())
    state = json.loads((run_directory / "state-01/state.json").read_text())
    resource = {key: metadata["resource"][key] for key in (
        "cpu_seconds", "wall_seconds", "train_rows", "validation_rows", "gateway_calls")}
    for key, projection in (("train_rows", "public/train.jsonl"),
                            ("validation_rows", "public/validation.jsonl")):
        if resource[key] != inputs[projection]["rows"]:
            raise RuntimeError(f"Variant {label} did not use the complete public projection")
    parent_sha = metadata["parent_checkpoint_sha256"].removeprefix("sha256:")
    if parent_sha != inputs["checkpoint.npz"]["sha256"]:
        raise RuntimeError(f"Variant {label} used a different parent checkpoint")
    if identity != source_identity(repository) or inputs != data_identity(repository):
        raise RuntimeError(f"Source or input data changed during variant {label}")
    candidates = [{"kind": item["kind"], "primary": item["primary"]}
                  for item in state["candidates"]]
    return {
        "variant": label, "runtime": runtime, "source_identity": identity,
        "configuration": {key: environment[key] for key in sorted(environment)
                          if key.startswith("RSI_V2_") or key in THREAD_NAMES},
        "end_to_end_seconds": elapsed, "reported_training_resource": resource,
        "parent_checkpoint_sha256": parent_sha,
        "candidate_checkpoint_sha256": metadata["candidate_checkpoint_sha256"],
        "chosen": metadata["chosen"],
        "probe_retain_applied": metadata["probe_retain_applied"],
        "candidate_evaluations": candidates,
        "gateway_calls_verified": len(report["rounds"][0]["model_call_ids"]),
        "prediction_lines": report["rounds"][0]["predictions"],
        "protocol_round_completed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--candidate-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate = args.candidate_repo.resolve()
    baseline = args.baseline_repo.resolve()
    baseline_commit = subprocess.check_output(
        ["git", "-C", str(baseline), "rev-parse", "HEAD"], text=True,
    ).strip()
    if baseline_commit != BASELINE_COMMIT:
        raise RuntimeError("Baseline repository is not the specified upstream revision")
    inputs = data_identity(baseline)
    if inputs != data_identity(candidate):
        raise RuntimeError("Baseline and candidate data or parent checkpoints differ")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("RSI_", "CONTEST_", "EMU_"))
                   and key != "PYTHONPATH"}
    environment.update({name: "1" for name in THREAD_NAMES})
    environment.update(
        RSI_V2_CONTROLLER="auto", RSI_V2_CAPACITY_ARM="ls10",
        RSI_V2_WARMUP_ROUNDS="0", RSI_V2_B_RETAIN="0",
        RSI_V2_EXPERIMENT_CACHE="1", RSI_V2_C_MAX_ALPHA="0.25",
    )
    variants = (
        ("upstream_late_retain", baseline,
         {"RSI_V2_PROBE_RETAIN": "C", "RSI_V2_SCRATCH_ARM": "on", "RSI_V2_C_EFFICIENT": "0"}),
        ("current_early_retain", candidate,
         {"RSI_V2_PROBE_RETAIN": "C", "RSI_V2_SCRATCH_ARM": "off", "RSI_V2_C_EFFICIENT": "1"}),
        ("current_conservative_c", candidate,
         {"RSI_V2_PROBE_RETAIN": "", "RSI_V2_SCRATCH_ARM": "off", "RSI_V2_C_EFFICIENT": "1"}),
    )
    output = args.output or candidate / "outputs/resource_benchmark_current.json"
    results = []
    payload = {
        "author": "Arthur", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": baseline_commit,
        "task": "C", "rounds_per_variant": 1, "execution_order": "serial",
        "complete": False, "inputs": inputs,
        "limitations": [
            "Single observation per variant on a local CPU; no repeated-run variance estimate.",
            "Complete official public train/validation projections and a fresh official parent per variant; no hidden data.",
            "Fake gateway, artifact/state upload, round finish and three-row prediction protocol check are included.",
            "End-to-end wall time includes emulator setup and prediction; reported CPU/wall end before final artifact/state upload.",
            "Reported CPU is participant process_time, not total host/container CPU; timings are not platform scores.",
            "Upstream and current paths also differ in SDK/protocol implementation; this is a complete-path comparison.",
            "Cold state for every variant; no cross-round cache reuse is measured.",
        ],
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for label, repository, overrides in variants:
        result = run_variant(label, repository, environment | overrides, inputs)
        if results and result["runtime"] != results[0]["runtime"]:
            raise RuntimeError("Runtime changed between benchmark variants")
        if results:
            original = results[0]
            result["reduction_vs_upstream_percent"] = {
                "reported_cpu_seconds": 100.0 * (1.0 - result["reported_training_resource"]["cpu_seconds"]
                    / original["reported_training_resource"]["cpu_seconds"]),
                "end_to_end_seconds": 100.0 * (1.0 - result["end_to_end_seconds"]
                    / original["end_to_end_seconds"]),
            }
        results.append(result)
        payload["complete"] = len(results) == len(variants)
        payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
