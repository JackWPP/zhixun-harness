"""Probe the label-free streaming policy for task A on raw UCI batches.

Evaluation uses exactly the runtime path that ships in the harness
(`apply_postprocess_stream`), so the number here is the number the platform
would get for a given postprocess spec, not a batch-mode approximation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rsi.inference import apply_postprocess_stream  # noqa: E402
from rsi.eval.metrics import macro_f1_fixed  # noqa: E402
from rsi.model import ExtendedModel  # noqa: E402

GAS_ROOT = REPO / "extra" / "uci-gas" / "Dataset"
TASK_DIR = REPO / "datasets" / "cpu-a-guide-20261110-v1"
CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def read_batch(index: int):
    xs, ys = [], []
    with (GAS_ROOT / ("batch%d.dat" % index)).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            label = int(parts[0]) - 1
            features = np.zeros(128, dtype=np.float64)
            for item in parts[1:]:
                key, value = item.split(":", 1)
                features[int(key) - 1] = float(value)
            xs.append(features)
            ys.append(label)
    return np.asarray(xs), np.asarray(ys, dtype=np.int64)


def score(probabilities, expected):
    return macro_f1_fixed(probabilities, expected, 6)


SPECS = {
    "none": {"type": "none"},
    "recenter_mean": {"type": "recenter", "mode": "batch_mean", "weight": 1.0},
    "recenter_std": {"type": "recenter", "mode": "batch_std", "weight": 1.0},
    "recenter_std_clip2": {"type": "recenter", "mode": "batch_std", "weight": 1.0,
                            "std_ratio_clip": 2.0},
    "pair_median": {"type": "pair_stream", "pair": [0, 3], "window": 512, "min_count": 64,
                     "fallback_delta": -4.0},
    "recenter_std_pair": {"type": "pair_stream", "pair": [0, 3], "window": 512, "min_count": 64,
                           "fallback_delta": -4.0,
                           "base": {"type": "recenter", "mode": "batch_std", "weight": 1.0}},
    "pair_hist_m40": {"type": "pair_threshold", "pair": [0, 3], "delta": -4.0},
    "pair_hist_rs_m40": {"type": "pair_threshold", "pair": [0, 3], "delta": -4.0,
                          "base": {"type": "recenter", "mode": "batch_std", "weight": 1.0}},
    "pair_hist_rs_w8_m44": {"type": "pair_threshold", "pair": [0, 3], "delta": -4.4, "window": 8192,
                             "base": {"type": "recenter", "mode": "batch_std", "weight": 1.0}},
    "recenter_clip2_pair": {"type": "pair_stream", "pair": [0, 3], "window": 512, "min_count": 64,
                             "fallback_delta": -4.0,
                             "base": {"type": "recenter", "mode": "batch_std", "weight": 1.0,
                                      "std_ratio_clip": 2.0}},
}


def run(batches=range(2, 11), out=None):
    model = ExtendedModel.from_checkpoint(TASK_DIR / "checkpoint.npz")
    payload = {}
    print("batch  " + "  ".join("%-18s" % name for name in SPECS), flush=True)
    totals = {name: [] for name in SPECS}
    for index in batches:
        x, y = read_batch(index)
        row = {}
        for name, spec in SPECS.items():
            probabilities = apply_postprocess_stream(spec, model, x)
            value = score(probabilities, y)
            row[name] = value
            totals[name].append(value)
        payload[index] = row
        print("b%-3d   " % index + "  ".join("%-18.4f" % row[name] for name in SPECS), flush=True)
    print()
    for name in SPECS:
        values = totals[name]
        print("%-20s all=%.4f  b3-b10=%.4f" % (name, float(np.mean(values)), float(np.mean(values[1:]))))
    if out:
        Path(out).write_text(json.dumps({"protocol": "streaming replay, fixed-6 macro-F1",
                                          "per_batch": payload}, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print("wrote", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--out", default=".scratch/sim/cert/a_stream_probe.json")
    args = parser.parse_args()
    run(range(2, args.batches + 1), args.out)


if __name__ == "__main__":
    main()
