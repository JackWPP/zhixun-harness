"""Controlled capacity experiments for task A (UCI gas drift).

Protocol (faithful to the contest round flow):
  round r predicts UCI batch r after training on batches 1..r-1.
  All reported scores use the *fixed* six classes (missing classes count as 0).

This lab is intentionally independent of the platform simulator: it isolates the
question "do loss / hidden-layer changes help under the same forward protocol?"
and writes a paired block-bootstrap CI versus the CE baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rsi.capacity import CapacityModel, capacity_spec  # noqa: E402
from rsi.model import ExtendedModel  # noqa: E402

GAS_ROOT = REPO / "extra" / "uci-gas" / "Dataset"
TASK_DIR = REPO / "datasets" / "cpu-a-guide-20261110-v1"
CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
OUT = REPO / ".scratch" / "sim" / "cert"
SEEDS = (42, 43, 44)


def read_batch(index: int):
    path = GAS_ROOT / ("batch%d.dat" % index)
    xs, ys = [], []
    with path.open(encoding="utf-8", errors="replace") as handle:
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


def macro_f1(probabilities, expected, classes=6):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = np.argmax(probabilities, axis=1)
    hits = np.zeros(classes)
    totals = np.zeros(classes)
    for label in range(classes):
        mask = expected == label
        totals[label] = float(mask.sum())
        hits[label] = float((predicted[mask] == label).sum()) if mask.any() else 0.0
    f1 = []
    for label in range(classes):
        tp = hits[label]
        fp = float(((predicted == label) & (expected != label)).sum())
        fn = totals[label] - hits[label]
        if tp + fn == 0:
            f1.append(0.0)  # fixed-class convention: missing support stays 0
            continue
        denom = 2 * tp + fp + fn
        f1.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1))


def accuracy(probabilities, expected):
    return float((np.argmax(probabilities, axis=1) == expected).mean())


def bootstrap_ci(values, rounds=4000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(rounds)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def variant_specs():
    """Return [(name, capacity_spec, epochs, from_parent)]."""
    specs = []
    for name, loss in (
        ("ce", {"type": "ce"}),
        ("focal_g1", {"type": "focal", "gamma": 1.0}),
        ("focal_g2", {"type": "focal", "gamma": 2.0}),
        ("ls_05", {"type": "label_smoothing", "eps": 0.05}),
        ("ls_10", {"type": "label_smoothing", "eps": 0.1}),
        ("class_bal", {"type": "class_balanced", "beta": 0.999}),
        ("logit_adj", {"type": "logit_adjusted", "tau": 1.0}),
        ("pair_aux", {"type": "pair_aux", "lambda": 0.2, "pair": [0, 3]}),
    ):
        specs.append((name, {"hidden": 32, "loss": loss}, 40, True))
    for name, arch in (
        ("h64", {"hidden": 64}),
        ("h128", {"hidden": 128}),
        ("h64_h2", {"hidden": 64, "hidden2": 32}),
        ("relu32", {"hidden": 32, "activation": "relu"}),
        ("relu64", {"hidden": 64, "activation": "relu"}),
        ("two_relu", {"hidden": 64, "hidden2": 32, "activation": "relu"}),
    ):
        specs.append((name, arch, 40, True))
    specs.append(("scratch_ce", {"hidden": 32}, 150, False))
    specs.append(("scratch_h64", {"hidden": 64}, 150, False))
    return specs


def run(seeds=SEEDS, batches=8, quick=False):
    data = {index: read_batch(index) for index in range(1, batches + 1)}
    baseline = ExtendedModel.from_checkpoint(TASK_DIR / "checkpoint.npz")
    specs = variant_specs()
    if quick:
        specs = [spec for spec in specs if spec[0] in ("ce", "focal_g1", "ls_10", "h64", "scratch_ce")]
    results = {name: {"rounds": [], "per_seed": {}} for name, *_ in specs}
    for seed in seeds:
        for name, arch, epochs, from_parent in specs:
            per_round = []
            carried = None
            for round_index in range(2, batches + 1):
                train_x = np.concatenate([data[i][0] for i in range(1, round_index)])
                train_y = np.concatenate([data[i][1] for i in range(1, round_index)])
                eval_x, eval_y = data[round_index]
                if from_parent:
                    model = CapacityModel.from_model(baseline, capacity_spec(**arch), seed=seed)
                else:
                    model = CapacityModel("classification", baseline.meta["paths"], CLASSES,
                                          train_x, train_y, seed=seed,
                                          **capacity_spec(**arch))
                model.set_learning_rate(1e-3 if from_parent else 3e-3)
                model.fit(train_x, train_y, epochs)
                probabilities = model.predict(eval_x)
                score = macro_f1(probabilities, eval_y)
                per_round.append({"round": round_index, "batch": round_index,
                                  "macro_f1": score,
                                  "accuracy": accuracy(probabilities, eval_y),
                                  "n": int(len(eval_y))})
            record = {"seed": seed, "rounds": per_round}
            results[name]["per_seed"][str(seed)] = record
            results[name]["rounds"].append(per_round)
            mean_hidden = float(np.mean([r["macro_f1"] for r in per_round[1:]]))  # b3..b10
            print("%-12s seed=%d b2=%.4f hidden(b3-b10)=%.4f" % (
                name, seed, per_round[0]["macro_f1"], mean_hidden), flush=True)
    # Aggregate: compare each variant to the CE baseline with paired batch diffs.
    def batch_vector(name):
        rows = []
        for seed in seeds:
            for item in results[name]["per_seed"][str(seed)]["rounds"][1:]:
                rows.append((item["round"], item["macro_f1"]))
        return rows

    base = dict(batch_vector("ce"))
    summary = {}
    for name, *_ in specs:
        rows = batch_vector(name)
        diffs = [value - base.get(round_index, 0.0) for round_index, value in rows]
        low, high = bootstrap_ci(diffs)
        hidden_mean = float(np.mean([value for _, value in rows]))
        b2 = float(np.mean([results[name]["per_seed"][str(seed)]["rounds"][0]["macro_f1"] for seed in seeds]))
        summary[name] = {"b2": b2, "hidden_mean": hidden_mean, "delta_vs_ce": float(np.mean(diffs)),
                         "ci_low": low, "ci_high": high}
        print("%-12s b2=%.4f hidden=%.4f delta=%+.4f CI=[%+.4f,%+.4f]" % (
            name, b2, hidden_mean, summary[name]["delta_vs_ce"], low, high), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": "uci b1..b_{r-1} -> b_r, fixed-6 macro-F1, b3-b10 mean",
               "seeds": list(seeds), "batches": batches, "variants": summary,
               "detail": results}
    (OUT / "capacity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Capacity lab (task A, fixed-6 macro-F1)", "",
             "Protocol: train b1..b_{r-1}, predict b_r; hidden = b3-b10 mean; paired block bootstrap vs CE.", "",
             "| variant | b2 | hidden mean | delta vs CE | 95% CI |", "|---|---|---|---|---|"]
    for name in summary:
        item = summary[name]
        lines.append("| %s | %.4f | %.4f | %+.4f | [%+.4f, %+.4f] |" % (
            name, item["b2"], item["hidden_mean"], item["delta_vs_ce"], item["ci_low"], item["ci_high"]))
    (OUT / "capacity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "capacity.json")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = SEEDS[: max(1, args.seeds)]
    run(seeds=seeds, batches=args.batches, quick=args.quick)


if __name__ == "__main__":
    main()
