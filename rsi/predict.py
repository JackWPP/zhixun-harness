#!/usr/bin/env python3
"""Official-compatible prediction entry point.

Reads one JSON object per line from stdin::

    {"task": "<task key>", "inputs": {...}}

and writes one prediction per line to stdout::

    {"outputs": [{"label": "...", "score": ...}, ...]}     # classification
    {"outputs": {"value": ...}}                              # regression

The platform evaluates in batch mode, so the default mode buffers all rows and
applies the round's postprocess (e.g. per-batch recentering) before answering.
``--mode stream`` keeps the official line-by-line behaviour using a causal rolling
recentering window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from rsi.inference import RecenterStream, apply_postprocess, format_outputs, resolve_postprocess
from rsi.model import ExtendedModel


def _read_rows(stream):
    rows = []
    for line in stream:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _features(row):
    inputs = row.get("inputs") or {}
    if "features" in inputs:
        return np.asarray(inputs["features"], dtype=np.float64)
    if "sensors" in inputs:
        return np.asarray(inputs["sensors"], dtype=np.float64)
    raise ValueError(f"unsupported inputs keys: {sorted(inputs)}")


def _load_postprocess(spec_arg: str, model: ExtendedModel):
    if not spec_arg:
        return {"type": "none"}
    candidate = Path(spec_arg)
    spec = json.loads(candidate.read_text()) if candidate.is_file() else json.loads(spec_arg)
    if isinstance(spec, dict) and "postprocess" in spec and isinstance(spec["postprocess"], dict):
        spec = spec["postprocess"]
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

    rows = _read_rows(sys.stdin)
    if not rows:
        return 0

    if args.mode == "stream":
        stream = None
        if postprocess.get("type") == "recenter":
            stream = RecenterStream(model.mean, model.scale,
                                    mode=postprocess.get("mode", "batch_mean"),
                                    window=args.stream_window)
        for row in rows:
            row_task = row.get("task", task)
            if row_task != task:
                sys.stdout.write(json.dumps({"error": f"task mismatch: {row_task} != {task}"}) + "\n")
                sys.stdout.flush()
                continue
            features = _features(row)
            predicted = _predict_row_stream(model, postprocess, features, stream)
            sys.stdout.write(json.dumps(format_outputs(predicted, task, classes)) + "\n")
            sys.stdout.flush()
        return 0

    # batch mode: the platform evaluates batches; compute the batch statistic once
    tasks = [row.get("task", task) for row in rows]
    if any(item != task for item in tasks):
        sys.stderr.write("warning: mixed task keys in one batch; checkpoint task is "
                         f"{task}, got {sorted(set(tasks))}\n")
    x = np.vstack([_features(row) for row in rows])
    predicted = apply_postprocess(postprocess, model, x)
    for index in range(len(rows)):
        value = predicted[index]
        sys.stdout.write(json.dumps(format_outputs(value, task, classes)) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
