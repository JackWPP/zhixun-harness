#!/usr/bin/env python3
"""Official-compatible prediction entry point.

Reads one JSON object per line from stdin::

    {"task": "<task key>", "inputs": {...}}

and writes one prediction per line to stdout::

    {"outputs": [{"label": "...", "score": ...}, ...]}     # classification
    {"outputs": {"value": ...}}                              # regression

The default batch mode buffers all rows and applies the round's postprocess
(e.g. per-batch recentering) before answering. Use ``--mode stream`` for the
official line-by-line protocol: each prediction is flushed before reading the
next request, using only the current and preceding rows for calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from rsi.inference import (DamageShrinkStream, RecenterStream, apply_postprocess,
                           format_outputs, resolve_postprocess)
from rsi.data_official import flatten, json_value
from rsi.model import ExtendedModel


def _read_features(stream, model):
    """Validate requests lazily; never read ahead in interactive stream mode."""
    for number, line in enumerate(stream, 1):
        line = line.strip()
        if not line:
            continue
        try:
            yield _features(json_value(line), model)
        except ValueError as exc:
            raise ValueError(f"line {number}: {exc}") from exc


def _features(row, model):
    if not isinstance(row, dict) or "inputs" not in row or set(row) - {"task", "inputs"}:
        raise ValueError("prediction request requires only task and inputs")
    task = model.meta["task"]
    requested = row.get("task", task)
    # The official protocol uses classification/regression. Accept a dataset
    # alias only when it matches the checkpoint's own profile, so B cannot
    # silently select a C model merely because both perform regression.
    profile = model.meta.get("profile")
    alias_matches = (isinstance(requested, str) and isinstance(profile, str)
                     and requested.upper() == profile.upper())
    if requested != task and not alias_matches:
        raise ValueError(f"prediction task mismatch: {requested!r}; "
                         f"checkpoint task is {task!r}, profile is {profile!r}")
    paths, values = flatten(row["inputs"])
    if paths != model.meta["paths"]:
        raise ValueError("prediction feature schema differs from checkpoint "
                         f"(expected {len(model.meta['paths'])} paths, got {len(paths)})")
    return np.asarray(values, dtype=np.float64)


def _load_postprocess(spec_arg: str, model: ExtendedModel):
    if not spec_arg:
        return {"type": "none"}
    candidate = Path(spec_arg)
    spec = json.loads(candidate.read_text()) if candidate.is_file() else json.loads(spec_arg)
    if isinstance(spec, dict) and "postprocess" in spec and isinstance(spec["postprocess"], dict):
        spec = spec["postprocess"]
    if (spec or {}).get("type") == "damage_shrink" and "curve" not in spec:
        raise SystemExit("damage_shrink must be resolved at experiment time (curve/robust_scale "
                         "missing); ship the resolved postprocess.json instead of {fit: train}")
    return resolve_postprocess(spec, model)


def _predict_row_stream(model, postprocess, features, stream: RecenterStream):
    kind = postprocess.get("type", "none")
    if kind == "none":
        return model.predict(features.reshape(1, -1))[0]
    raw = model.predict(features.reshape(1, -1))[0]
    if kind == "shrink":
        alpha = float(postprocess["alpha"])
        center = float(postprocess["center"])
        return alpha * raw + (1.0 - alpha) * center
    if kind == "damage_shrink":
        return stream.transform(features, raw)
    if kind == "recenter":
        weight = float(postprocess.get("weight", 1.0))
        if weight <= 0.0:
            return raw
        calibrated_features = stream.calibrate(features)
        calibrated = model.predict(calibrated_features.reshape(1, -1))[0]
        output = (1.0 - weight) * raw + weight * calibrated
        if model.meta["task"] == "classification":
            output = np.clip(output, 1e-12, None)
            output = output / output.sum()
        return output
    raise ValueError(f"unsupported postprocess: {kind}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RSI harness official-compatible predictor")
    parser.add_argument("--checkpoint", required=True, help="candidate/parent checkpoint.npz")
    parser.add_argument("--postprocess", default="", help="JSON spec or path to a summary/postprocess JSON")
    parser.add_argument("--mode", choices=("batch", "stream"), default="batch")
    parser.add_argument("--stream-window", type=int, default=2048)
    args = parser.parse_args(argv)

    model = ExtendedModel.from_checkpoint(args.checkpoint)
    postprocess = _load_postprocess(args.postprocess, model)
    task = model.meta["task"]
    classes = list(model.meta.get("classes") or [])

    try:
        features = _read_features(sys.stdin, model)
        if args.mode == "stream":
            stream = None
            if postprocess.get("type") == "damage_shrink":
                stream = DamageShrinkStream(postprocess)
            elif postprocess.get("type") == "recenter":
                stream = RecenterStream(model.mean, model.scale,
                                        mode=postprocess.get("mode", "batch_mean"),
                                        window=args.stream_window)
            for row_features in features:
                predicted = _predict_row_stream(model, postprocess, row_features, stream)
                output = json.dumps(format_outputs(predicted, task, classes), allow_nan=False)
                sys.stdout.write(output + "\n")
                sys.stdout.flush()
            return 0

        # Batch mode deliberately computes calibration from the entire batch.
        # Validate every request first, so a bad schema cannot produce a
        # partially populated batch response.
        rows = list(features)
        if not rows:
            return 0
        predicted = apply_postprocess(postprocess, model, np.vstack(rows))
        for value in predicted:
            output = json.dumps(format_outputs(value, task, classes), allow_nan=False)
            sys.stdout.write(output + "\n")
        sys.stdout.flush()
    except ValueError as exc:
        sys.stderr.write(f"rsi-predict: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
