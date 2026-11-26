"""Review one parent-based B continuation on complete official public data.

Run from any directory with the repository's Python environment::

    python experiments/b_continuation_review.py

This performs one 15-epoch continuation, never the legacy 600-epoch scratch
search. Public validation is measurement-only. Reported local timings are
diagnostics, not a resource-efficiency benchmark or a platform score estimate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time


# Set numerical-runtime limits before the first NumPy import.
for _name in ("BASELINE_CPU_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
              "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rsi.model import ExtendedModel
from rsi.platform import plans, submission
from rsi.platform.b_safety import bearing_guard
from rsi.platform.experiment_cache import ExperimentCache, file_sha256, source_sha256


PACKAGE = ROOT / "datasets" / "cpu-b-rul-20261110-v1"
_BEARING = re.compile(r"^xjtu_(Bearing\d+_\d+)_s\d+$")


def bearing_id(row: dict) -> str:
    match = _BEARING.fullmatch(str(row.get("id", "")))
    identifier = match.group(1) if match else None
    supplied = row.get("meta", {}).get("bearing")
    if identifier is None or supplied != identifier:
        raise ValueError("Missing or inconsistent bearing identity in public row")
    return identifier


def read_rows(path: Path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    bearings = np.asarray([bearing_id(row) for row in rows])
    return rows, bearings, hashlib.sha256(raw).hexdigest()


def measured(predictions, expected) -> dict:
    error = np.asarray(predictions, dtype=float) - np.asarray(expected, dtype=float)
    if not len(error) or not np.all(np.isfinite(error)):
        raise ValueError("Metrics require non-empty finite predictions and labels")
    return {"rows": int(len(error)), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error))}


def compare(parent, candidate, expected) -> dict:
    baseline = measured(parent, expected)
    changed = measured(candidate, expected)
    delta = {metric: changed[metric] - baseline[metric] for metric in ("rmse", "mae")}
    return {"parent": baseline, "candidate": changed, "candidate_minus_parent": delta,
            "relative_change_percent": {metric: 100.0 * delta[metric] / baseline[metric]
                                        if baseline[metric] else None for metric in delta},
            "rmse_and_mae_noninferior": all(delta[metric] <= 0 for metric in delta)}


def interpolation_exploration(parent, trained, ctx, validation_bearings) -> list[dict]:
    """Evaluate two predeclared parameter interpolations; never fit or select."""
    results = []
    baseline = ctx.parent_predicted
    identifiers = [str(row["id"]) for row in ctx.val_rows]
    for alpha in (0.25, 0.5):
        interpolated = trained.clone()
        for name in parent.params:
            interpolated.params[name] = parent.params[name] + alpha * (trained.params[name] - parent.params[name])
        # This is an inference-only diagnostic, not an adopted checkpoint. Reset
        # inconsistent Adam moments rather than implying a trained trajectory.
        interpolated.reset_optimizer()
        predicted = interpolated.predict(ctx.x_val)
        values = measured(predicted, ctx.y_val)
        guard = bearing_guard(baseline, predicted, ctx.y_val, identifiers, min_rows=10)
        by_bearing = {}
        for bearing in sorted(set(validation_bearings)):
            mask = validation_bearings == bearing
            by_bearing[str(bearing)] = compare(baseline[mask], predicted[mask], ctx.y_val[mask])
        aggregate = compare(baseline, predicted, ctx.y_val)
        aggregate_pass = bool(values["rmse"] < ctx.parent_values["rmse"] - ctx.margin
                              and values["mae"] <= ctx.parent_values["mae"])
        results.append({
            "alpha": alpha, "aggregate": aggregate, "by_validation_bearing": by_bearing,
            "aggregate_pass": aggregate_pass, "bearing_guard_pass": bool(guard["passed"]),
            "would_pass_combined_guard": bool(aggregate_pass and guard["passed"]),
            "bearing_guard": guard, "additional_fit_calls": 0,
            "adopted": False,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "b_continuation_review.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=.001)
    args = parser.parse_args()
    os.environ["RSI_V2_B_EPOCHS"] = str(args.epochs)
    os.environ["RSI_V2_B_LR"] = str(args.learning_rate)
    parameters = plans.b_continuation_parameters()  # Validate the production limits.

    train_path = PACKAGE / "public" / "train.jsonl"
    validation_path = PACKAGE / "public" / "validation.jsonl"
    parent_path = PACKAGE / "checkpoint.npz"
    parent_sha = file_sha256(parent_path)
    train, train_bearings, train_sha = read_rows(train_path)
    validation, validation_bearings, validation_sha = read_rows(validation_path)
    train_ids = {str(row["id"]) for row in train}
    validation_ids = {str(row["id"]) for row in validation}
    overlapping_ids = sorted(train_ids & validation_ids)
    overlapping_bearings = sorted(set(train_bearings) & set(validation_bearings))
    if overlapping_ids or overlapping_bearings:
        raise ValueError("Train and validation must be disjoint by row and whole bearing")
    if len(train_ids) != len(train) or len(validation_ids) != len(validation):
        raise ValueError("Public projections contain duplicate row identifiers")

    parent = ExtendedModel.from_checkpoint(parent_path)
    parent.meta["_source_path"] = str(parent_path)
    feature_paths = [json.loads(path) if isinstance(path, str) else path
                     for path in parent.meta["paths"]]
    if not all(path and path[0] == ["key", "features"] for path in feature_paths):
        raise ValueError("B model inputs must stay inside the public sensor feature tree")
    ctx = plans.PlanContext.from_rows(
        "B", parent, train, validation, metric="rmse", direction="lower",
        margin=float(submission.TASK_DEFAULTS["B"]["margin"]),
        experiment_cache=ExperimentCache(enabled=False))
    baseline = plans.retain_plan(ctx)
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    candidate = plans.b_continuation_plan(ctx)
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    if not candidate.get("ok"):
        raise RuntimeError(candidate.get("error") or "B continuation failed")
    trained = candidate["model"]

    checks = {
        "parameter_shapes_preserved": all(parent.params[name].shape == trained.params[name].shape
                                          for name in parent.params),
        "input_mean_preserved": np.array_equal(parent.mean, trained.mean),
        "input_scale_preserved": np.array_equal(parent.scale, trained.scale),
        "target_mean_preserved": parent.meta["target_mean"] == trained.meta["target_mean"],
        "target_scale_preserved": parent.meta["target_scale"] == trained.meta["target_scale"],
        "hidden_width_preserved": parent.meta["hidden"] == trained.meta["hidden"],
        "feature_paths_preserved": parent.meta["paths"] == trained.meta["paths"],
        "seed_preserved": parent.meta["seed"] == trained.meta["seed"],
        "epoch_history_continued": trained.meta["epochs"] == parent.meta["epochs"] + parameters["epochs"],
        "optimizer_steps_continued": trained.meta["steps"] == (
            parent.meta["steps"] + parameters["epochs"] *
            ((len(train) + parent.meta["batch_size"] - 1) // parent.meta["batch_size"])),
        "parent_file_unchanged": file_sha256(parent_path) == parent_sha,
    }
    if not all(checks.values()):
        raise AssertionError({name: passed for name, passed in checks.items() if not passed})
    with tempfile.TemporaryDirectory(prefix="b-continuation-review-") as temporary:
        candidate_path = Path(temporary) / "candidate.npz"
        trained.save(candidate_path)
        candidate_sha = file_sha256(candidate_path)

    aggregate = compare(baseline["predicted"], candidate["predicted"], ctx.y_val)
    per_bearing = {}
    for bearing in sorted(set(validation_bearings)):
        mask = validation_bearings == bearing
        per_bearing[str(bearing)] = compare(baseline["predicted"][mask], candidate["predicted"][mask],
                                            ctx.y_val[mask])
    # Reproduce the deterministic chooser, aggregate RMSE/MAE check, and the
    # required whole-bearing guard. A good aggregate must not mask a bad bearing.
    choice = submission._choose([baseline, candidate], "lower", ctx.margin, task_key="B")
    aggregate_guard = all(candidate["values"].get(key) is not None
                          and np.isfinite(candidate["values"][key])
                          and candidate["values"][key] <= ctx.parent_values[key]
                          for key in ("rmse", "mae"))
    bearing_report = bearing_guard(
        baseline["predicted"], candidate["predicted"], ctx.y_val,
        [str(row["id"]) for row in validation], min_rows=10)
    aggregate_pass = choice == 1 and aggregate_guard
    passed = aggregate_pass and bearing_report["passed"]
    interpolations = interpolation_exploration(parent, trained, ctx, validation_bearings)
    report = {
        "author": "Arthur", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one actual-parent B continuation on complete official public train/validation",
        "configuration": parameters,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "system": platform.system(), "machine": platform.machine(),
                    "configured_threads": 1, "source_sha256": source_sha256()},
        "inputs": {"package": str(PACKAGE.relative_to(ROOT)),
                   "train_rows": len(train), "validation_rows": len(validation),
                   "train_file_sha256": train_sha, "validation_file_sha256": validation_sha,
                   "parent_checkpoint_sha256": parent_sha},
        "data_boundary": {
            "train_bearings": sorted(set(train_bearings)),
            "validation_bearings": sorted(set(validation_bearings)),
            "overlapping_bearings": overlapping_bearings, "overlapping_row_ids": overlapping_ids,
            "fit_source": "public/train.jsonl", "validation_used_for_fit": False,
            "bearing_and_snapshot_not_model_inputs": True,
        },
        "parent_lineage": {"checks": {key: bool(value) for key, value in checks.items()},
                           "parent_hidden": parent.meta["hidden"], "candidate_hidden": trained.meta["hidden"],
                           "parent_epochs": parent.meta["epochs"], "candidate_epochs": trained.meta["epochs"],
                           "parent_steps": parent.meta["steps"], "candidate_steps": trained.meta["steps"],
                           "seed": trained.meta["seed"], "candidate_checkpoint_sha256": candidate_sha},
        "aggregate": aggregate, "by_validation_bearing": per_bearing,
        "current_gate": {"rmse_gain_must_exceed": ctx.margin,
                         "deterministic_choice": "continue_train" if choice else "retain",
                         "aggregate_rmse_mae_guard_passed": bool(aggregate_guard),
                         "aggregate_pass": bool(aggregate_pass),
                         "bearing_guard_pass": bool(bearing_report["passed"]),
                         "passes_current_default_selection": bool(passed),
                         "final_choice": "continue_train" if passed else "retain",
                         "all_bearings_noninferior": all(row["rmse_and_mae_noninferior"]
                                                         for row in per_bearing.values()),
                         "bearing_guard_required": True, "bearing_guard": bearing_report},
        "fixed_interpolation_exploration": {
            "purpose": "Two predeclared diagnostic points, no validation-fitted alpha or automatic adoption",
            "alphas": [0.25, 0.5], "results": interpolations,
            "fit_calls_within_this_run": 1, "additional_fit_calls_for_interpolations": 0,
            "reproduction_note": "The initial review persisted JSON only. This review reran the same 15-epoch continuation once to recover the model, then shared it across both interpolation points.",
        },
        "experiment_cache": candidate.get("experiment_cache"),
        "local_timing_diagnostic": {"candidate_and_validation_cpu_seconds": cpu_seconds,
                                    "candidate_and_validation_wall_seconds": wall_seconds,
                                    "performance_benchmark": False,
                                    "note": "Single local run; not a platform resource-efficiency benchmark."},
        "limitations": [
            "Uses only the official public validation bearings; no hidden labels or platform score were accessed.",
            "This does not demonstrate preservation of the previous scratch model's hidden RMSE 155.172446.",
            "No legacy 600-epoch scratch training, multi-seed search, real gateway call or platform submission was run.",
            "Fixed interpolation points are exploratory validation measurements and are not enabled as the default B policy.",
            "The runtime context is local macOS, not the official 2-core/4GiB formal container.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregate": aggregate,
                      "current_gate": report["current_gate"],
                      "local_timing_diagnostic": report["local_timing_diagnostic"]},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
