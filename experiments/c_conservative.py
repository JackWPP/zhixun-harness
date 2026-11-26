"""Fixed C continuation/interpolation comparison; validation never enters fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.c_candidates import load_rows
from rsi.metrics import from_predictions
from rsi.model import ExtendedModel
from rsi.platform.c_safety import conservative_candidates, group_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/c_conservative_review.json"))
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1] / "datasets/cpu-c-guide-20261110-v1"
    parent = ExtendedModel.load(package / "checkpoint.npz")
    train_x, train_y, train_groups, train_sha = load_rows(package / "public/train.jsonl", parent)
    val_x, val_y, val_groups, val_sha = load_rows(package / "public/validation.jsonl", parent)
    assert not set(train_groups) & set(val_groups)
    manifest = json.loads((package / "training.json").read_text())
    assert "sha256:" + train_sha == manifest["train_sha256"]
    assert "sha256:" + val_sha == manifest["validation_sha256"]
    parent_sha = hashlib.sha256((package / "checkpoint.npz").read_bytes()).hexdigest()
    assert "sha256:" + parent_sha == manifest["checkpoint_sha256"]
    row_ids = [row["id"] for row in map(json.loads, (package / "public/validation.jsonl").read_text().splitlines())]
    parent_pred = parent.predict(val_x)
    trained = parent.clone()
    trained.meta["seed"] = 20261117
    trained.meta["learning_rate"] = .0003
    cpu_start = time.process_time()
    trained.fit(train_x, train_y, 10)
    fit_cpu_seconds = time.process_time() - cpu_start
    full_pred = trained.predict(val_x)
    records = []
    for name, alpha, model in [("parent", 0., parent), ("full_update", 1., trained),
                                *[("conservative_"+str(alpha), alpha, model)
                                  for alpha, model in conservative_candidates(parent, trained)]]:
        start = time.process_time()
        pred = model.predict(val_x)
        prediction_cpu_seconds = time.process_time() - start
        values = from_predictions("regression", pred, val_y, profile="C")
        guard = group_guard(parent_pred, pred, val_y, row_ids)
        output_mix = parent_pred + alpha * (full_pred-parent_pred)
        records.append({"name": name, "alpha": alpha, "validation": values, "group_guard": guard,
                        "prediction_cpu_seconds": prediction_cpu_seconds,
                        "output_mixture_reference": from_predictions("regression", output_mix, val_y, profile="C"),
                        "epochs": model.meta["epochs"], "adam_steps": model.meta["steps"],
                        "interpolation": model.meta.get("c_conservative")})
    result = {"author": "Arthur", "python": platform.python_version(), "numpy": np.__version__,
              "train_rows": len(train_y), "validation_rows": len(val_y),
              "train_sha256": train_sha, "validation_sha256": val_sha, "parent_sha256": parent_sha,
              "train_validation_engines_disjoint": True,
              "same_input_and_target_normalization": True,
              "fit": {"epochs": 10, "learning_rate": .0003, "seed": 20261117,
                      "fit_cpu_seconds": fit_cpu_seconds, "source": "public/train only"},
              "results": records,
              "limitations": ["Repeatedly inspected public validation; exploratory, not independent confirmatory evidence.",
                              "FD groups are validation audits, not hidden drift stages or prediction routes.",
                              "No hidden-test or platform score was measured; timings are one local CPU run.",
                              "Interpolated models restart Adam moments and steps while preserving actual training epochs."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "rows": len(val_y),
                      "results": [{"name": r["name"], "metrics": r["validation"],
                                   "group_safe": r["group_guard"]["passed"]} for r in records]}))


if __name__ == "__main__":
    main()
