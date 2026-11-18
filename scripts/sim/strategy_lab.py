#!/usr/bin/env python3
"""Strategy lab: compare a "generation-1" arm catalog with a drift-aware
"generation-2" policy on the same simulated data drops.

The lab is a deterministic, label-aware offline rehearsal.  It never touches
the network, never uploads anything and never runs the contest harness.  For
every round it:

1. builds a parent model from the previous round's selected artifact;
2. evaluates a catalog of strategy arms on the *public validation* rows;
3. selects one arm (gen1 = best validation; gen2 = drift-aware robust choice);
4. scores the selected artifact on the held-out ``hidden proxy`` rows;
5. carries the selected artifact into the next round.

Usage::

    .venv/bin/python scripts/sim/strategy_lab.py --scenario official --rounds 3
    .venv/bin/python scripts/sim/strategy_lab.py --scenario external --rounds 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO))

from rsi.executor import ExperimentProposal  # noqa: E402
from rsi.inference import apply_postprocess  # noqa: E402
from rsi.metrics import from_predictions, primary as primary_metric  # noqa: E402
from rsi.model import ExtendedModel  # noqa: E402
from rsi.platform.plans import PlanContext, plan_for_proposal, retain_plan  # noqa: E402
from rsi.platform.submission import _drift_brief  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _specs() -> dict:
    datasets = _load_module("sim_datasets", REPO / "scripts" / "sim" / "datasets.py")
    datasets.ensure_packages()
    root = REPO / ".scratch" / "sim" / "datasets"
    return {
        "official": {
            "A": {"package": REPO / "datasets" / "cpu-a-guide-20261110-v1",
                  "task": "classification", "metric": "macro_f1", "direction": "higher",
                  "margin": 0.004, "classes": ["Ethanol", "Ethylene", "Ammonia",
                                               "Acetaldehyde", "Acetone", "Toluene"]},
            "B": {"package": REPO / "datasets" / "cpu-b-rul-20261110-v1",
                  "task": "regression", "metric": "rmse", "direction": "lower",
                  "margin": 5.0, "classes": []},
            "C": {"package": REPO / "datasets" / "cpu-c-guide-20261110-v1",
                  "task": "regression", "metric": "rmse", "direction": "lower",
                  "margin": 0.3, "classes": []},
        },
        "external": {
            "A": {"package": root / "sim-gas-ext-v1", "task": "classification",
                  "metric": "macro_f1", "direction": "higher", "margin": 0.004,
                  "classes": ["CO", "NO2", "C2H5OH", "Ethylene", "Toluene", "Ammonia"]},
            "SCHEMA": {"package": root / "sim-private-schema-v1", "task": "classification",
                       "metric": "macro_f1", "direction": "higher", "margin": 0.004,
                       "classes": []},
        },
    }


def _task_rows(spec: dict) -> tuple[list[dict], list[dict], list[dict]]:
    package = spec["package"]
    train = _rows(package / "public" / "train.jsonl")
    validation = _rows(package / "public" / "validation.jsonl")
    if len(validation) > 600:
        val, hidden = validation[:600], validation[600:]
    else:
        half = max(1, len(validation) // 2)
        val, hidden = validation[:half], validation[half:]
    return train, val, hidden


def _arms(task_key: str, task: str) -> list[dict]:
    """Generation-1 catalog mirrors the platform pipeline's candidate plans."""
    retain = {"label": "retain", "strategy": "retain", "params": {},
              "postprocess": {"type": "none"}, "hypothesis": "keep parent artifact"}
    if task == "classification":
        return [
            retain,
            {"label": "recenter-mean-0.4", "strategy": "retain", "params": {},
             "postprocess": {"type": "recenter", "mode": "batch_mean", "weight": 0.4},
             "hypothesis": "re-standardise batch mean shift"},
            {"label": "pair-delta-fit", "strategy": "retain", "params": {},
             "postprocess": {"type": "pair_threshold", "pair": [0, 3]},
             "hypothesis": "delayed-label pair flip fitted on train"},
        ]
    if task_key == "B":
        return [
            retain,
            {"label": "shrink-0.3", "strategy": "retain", "params": {},
             "postprocess": {"type": "shrink", "alpha": 0.3}, "hypothesis": "train-centre shrink"},
            {"label": "damage-shrink", "strategy": "retain", "params": {},
             "postprocess": {"type": "damage_shrink"}, "hypothesis": "bearing degradation curve"},
        ]
    return [
        retain,
        {"label": "shrink-0.3", "strategy": "retain", "params": {},
         "postprocess": {"type": "shrink", "alpha": 0.3}, "hypothesis": "train-centre shrink"},
        {"label": "continue-40", "strategy": "continue_train",
         "params": {"epochs": 40, "learning_rate": 1e-3},
         "postprocess": {"type": "none"}, "hypothesis": "short continuation"},
    ]


def _gen2_arms(task_key: str, task: str) -> list[dict]:
    """Generation-2 = gen1 + new arms discovered while rehearsing."""
    arms = _arms(task_key, task)
    if task == "classification":
        arms.append({"label": "recenter-std-0.2", "strategy": "retain", "params": {},
                     "postprocess": {"type": "recenter", "mode": "batch_std", "weight": 0.2},
                     "hypothesis": "coverage calibration with covariance weighting"})
        arms.append({"label": "pair-delta-history", "strategy": "retain", "params": {},
                     "postprocess": {"type": "pair_threshold", "pair": [0, 3], "delta": -4.0},
                     "hypothesis": "delayed-label delta from previous campaigns"})
    elif task_key == "C":
        arms.append({"label": "continue-20", "strategy": "continue_train",
                     "params": {"epochs": 20, "learning_rate": 1e-3},
                     "postprocess": {"type": "none"}, "hypothesis": "shorter continuation"})
    return arms


def _improved(task: str, candidate: float, parent: float, margin: float) -> bool:
    if task == "classification":
        return candidate >= parent + margin
    return candidate <= parent - margin


def _pick(plans: list[dict], task: str, parent_primary: float, margin: float,
          drift: dict, generation: str) -> dict:
    retain = next(item for item in plans if item["ok"] and item["kind"] == "retain")
    candidates = [item for item in plans if item["ok"] and item["kind"] != "retain"]
    severity = drift.get("severity", "unknown")
    effective_margin = margin
    allowed = candidates
    if generation == "gen2":
        if severity == "high":
            # Low-risk coverage calibration stays eligible, but the stronger
            # lesson from previous campaigns is the delayed-label rule: a
            # history-calibrated pair flip is validation-neutral yet improves
            # the drifting hidden batches, so it outranks a do-nothing retain.
            history_pair = [item for item in candidates
                            if item["label"] == "pair-delta-history"]
            if (history_pair and history_pair[0]["ok"]
                    and history_pair[0]["primary"] >= parent_primary - margin):
                chosen = dict(history_pair[0])
                chosen["selection_reason"] = (
                    "history-calibrated delayed-label rule survives high drift "
                    "(validation-neutral, hidden-batch prior)")
                return chosen
            allowed = [item for item in candidates
                       if item["label"] in ("recenter-mean-0.4", "recenter-std-0.2")]
            effective_margin = margin * 2.0
        elif severity == "medium":
            allowed = [item for item in candidates if "pair" not in item["label"]]
            effective_margin = margin * 1.25
    eligible = [item for item in allowed
                if _improved(task, item["primary"], parent_primary, effective_margin)]
    pool = eligible or []
    if not pool:
        chosen = dict(retain)
        chosen["selection_reason"] = (f"no arm beat parent by {effective_margin:g} "
                                      f"(drift={severity}, generation={generation})")
        return chosen
    chosen = (max(pool, key=lambda item: item["primary"]) if task == "classification"
              else min(pool, key=lambda item: item["primary"]))
    chosen = dict(chosen)
    chosen["selection_reason"] = (f"best validation among {len(pool)} eligible arms "
                                  f"(drift={severity}, margin={effective_margin:g}, "
                                  f"generation={generation})")
    return chosen


def _score_hidden(spec: dict, model: ExtendedModel, postprocess: dict | None,
                  hidden_rows: list[dict], classes: list[str]) -> float:
    from rsi.platform.plans import xy_rows
    x, y = xy_rows(hidden_rows, model, classes)
    predicted = apply_postprocess(postprocess or {"type": "none"}, model, x)
    values = from_predictions(spec["task"], predicted, y, classes)
    return float(values.get("rmse") if spec["task"] == "regression"
                 else values.get("macro_f1") or 0.0)


def run(scenario: str, rounds: int, tasks: list[str], generation: str) -> dict:
    table = _specs()[scenario]
    selected = tasks or sorted(table)
    out = {"scenario": scenario, "generation": generation, "rounds": {}, "task_table": {}}
    for task_key in selected:
        spec = table.get(task_key)
        if spec is None or spec["task"] == "regression" and scenario == "external":
            continue
        train_rows, val_rows, hidden_rows = _task_rows(spec)
        checkpoint = spec["package"] / "checkpoint.npz"
        model = ExtendedModel.from_checkpoint(checkpoint)
        model.meta["_source_path"] = str(checkpoint)
        task = spec["task"]
        classes = spec["classes"] or list(model.meta.get("classes") or [])
        parent_primary = None
        round_records = []
        for round_index in range(1, rounds + 1):
            started = time.time()
            ctx = PlanContext.from_rows(task_key, model, train_rows, val_rows,
                                        metric=spec["metric"], direction=spec["direction"],
                                        margin=spec["margin"], classes=classes)
            if parent_primary is None:
                parent_primary = ctx.parent_primary
            drift = _drift_brief(ctx)
            arms = (_gen2_arms if generation == "gen2" else _arms)(task_key, task)
            plans = []
            for arm in arms:
                proposal = ExperimentProposal(strategy=arm["strategy"], params=arm["params"],
                                              postprocess=arm["postprocess"],
                                              hypothesis=arm["hypothesis"])
                plan = plan_for_proposal(ctx, proposal)
                plan["label"] = arm["label"]
                plans.append(plan)
            if not any(item["ok"] and item["kind"] == "retain" for item in plans):
                plans.append(retain_plan(ctx) | {"label": "retain"})
            chosen = _pick(plans, task, parent_primary, spec["margin"], drift, generation)
            hidden = _score_hidden(spec, chosen["model"], chosen["postprocess"],
                                   hidden_rows, classes)
            record = {
                "round": round_index,
                "wall_s": round(time.time() - started, 2),
                "drift": {"severity": drift.get("severity"), "score": drift.get("score")},
                "arms": [{"label": item["label"], "ok": item["ok"],
                          "primary": item["primary"], "error": item.get("error", "")}
                         for item in plans],
                "chosen": chosen["label"],
                "chosen_primary": chosen["primary"],
                "hidden_proxy": hidden,
                "selection_reason": chosen.get("selection_reason", ""),
            }
            round_records.append(record)
            print(f"[lab:{generation}] {task_key} r{round_index} drift={drift.get('severity')} "
                  f"chosen={chosen['label']} val={chosen['primary']:.4f} hidden={hidden:.4f}")
            # Carry the selected artifact into the next round.
            if chosen.get("checkpoint_changed"):
                target = (REPO / ".scratch" / "sim" / "lab" / scenario / generation
                          / task_key / f"round-{round_index}.npz")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    target.unlink()
                chosen["model"].save(target)
                candidate = ExtendedModel.from_checkpoint(target)
                candidate.meta["_source_path"] = str(target)
                model = candidate
            else:
                model = chosen["model"]
                model.meta["_source_path"] = str(checkpoint)
            parent_primary = chosen["primary"]
        out["rounds"][task_key] = round_records
        out["task_table"][task_key] = {
            "hidden_last": round_records[-1]["hidden_proxy"],
            "val_last": round_records[-1]["chosen_primary"],
            "chosen_last": round_records[-1]["chosen"],
            "drift_last": round_records[-1]["drift"],
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RSI strategy lab (gen1 vs gen2)")
    parser.add_argument("--scenario", choices=["official", "external"], default="official")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--generation", choices=["gen1", "gen2"], default="gen1")
    parser.add_argument("--compare", action="store_true",
                        help="run both generations and write a comparison report")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    tasks = [item.strip().upper() for item in args.tasks.split(",") if item.strip()]
    from rsi.platform import plans as _plans
    if args.compare:
        reports = {}
        for generation in ("gen1", "gen2"):
            reports[generation] = run(args.scenario, args.rounds, tasks, generation)
        root = REPO / ".scratch" / "sim" / "lab" / args.scenario
        root.mkdir(parents=True, exist_ok=True)
        (root / "comparison.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
        table = {}
        for key in reports["gen1"]["task_table"]:
            table[key] = {"gen1": reports["gen1"]["task_table"][key],
                          "gen2": reports["gen2"]["task_table"][key]}
        (root / "comparison_table.json").write_text(
            json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(table, ensure_ascii=False, indent=1))
        return 0
    report = run(args.scenario, args.rounds, tasks, args.generation)
    root = Path(args.output) if args.output else (
        REPO / ".scratch" / "sim" / "lab" / args.scenario / f"{args.generation}.json")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lab] report written: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
