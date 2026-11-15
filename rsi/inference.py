"""Inference-time adaptation applied on top of a frozen checkpoint.

Nothing here modifies the checkpoint: the model stays a byte-compatible
continuation of the official baseline, while the predictor adapts per batch.
This keeps the "weights unchanged" compatibility rule and makes the adaptation
auditable: the postprocess spec is recorded in every round and shipped in the
final deployment manifest.

Supported postprocess types
---------------------------
none
    Plain checkpoint prediction.
recenter (or recenter_blend)
    Re-standardise the incoming feature batch onto the checkpoint's frozen
    training statistics (covariate-shift coverage calibration). ``weight``
    blends the calibrated probabilities/values with the raw prediction:
    ``out = (1 - weight) * raw + weight * calibrated``.
shrink
    Regression output calibration: ``y = alpha * prediction + (1 - alpha) * center``
    with ``center`` = the checkpoint's frozen train-target mean. Validation labels
    are intentionally NOT used: fitting the calibration centre on validation would
    violate the competition rule that validation/hidden data must not be fitted.
"""
from __future__ import annotations

import numpy as np

POSTPROCESS_KINDS = ("none", "recenter", "shrink")


def resolve_postprocess(spec, model, calibration=None):
    """Resolve a loose postprocess declaration into a concrete serialisable spec."""
    spec = dict(spec or {"type": "none"})
    kind = spec.get("type", "none")
    if kind not in POSTPROCESS_KINDS:
        raise ValueError(f"unsupported postprocess: {kind}")
    if kind == "none":
        return {"type": "none"}
    if kind == "recenter":
        mode = spec.get("mode", "batch_mean")
        if mode not in ("batch_mean", "batch_std"):
            raise ValueError("recenter mode must be batch_mean or batch_std")
        weight = float(spec.get("weight", 1.0))
        if not 0.0 <= weight <= 1.0:
            raise ValueError("recenter weight must be within [0, 1]")
        return {"type": "recenter", "mode": mode, "weight": round(weight, 6)}
    if kind == "shrink":
        if model.meta["task"] != "regression":
            raise ValueError("shrink calibration only applies to regression")
        alpha = float(spec.get("alpha", 0.5))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("shrink alpha must be within [0, 1]")
        center = float(model.meta.get("target_mean", 0.0))
        return {"type": "shrink", "alpha": round(alpha, 6), "center": round(center, 6),
                "center_source": "train_target_mean", "calibration_rows": 0}
    raise ValueError(f"unsupported postprocess: {kind}")  # pragma: no cover


def _recentered_features(model, x, mode):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=0)
    if mode == "batch_std":
        scale = np.std(x, axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        return (x - mean) / scale * model.scale + model.mean
    return x - mean + model.mean


def apply_postprocess(postprocess, model, x):
    """Predict a full feature matrix with the resolved postprocess applied.

    Returns the same object type as ``Model.predict``: probabilities for
    classification, a 1-D value vector for regression.
    """
    spec = postprocess or {"type": "none"}
    kind = spec.get("type", "none")
    x = np.asarray(x, dtype=np.float64)
    if kind == "none":
        return model.predict(x)
    if kind == "recenter":
        raw = model.predict(x)
        weight = float(spec.get("weight", 1.0))
        if weight <= 0.0:
            return raw
        calibrated = model.predict(_recentered_features(model, x, spec.get("mode", "batch_mean")))
        output = (1.0 - weight) * raw + weight * calibrated
        if model.meta["task"] == "classification":
            output = np.clip(output, 1e-12, None)
            output = output / output.sum(axis=1, keepdims=True)
        return output
    if kind == "shrink":
        alpha = float(spec["alpha"])
        center = float(spec["center"])
        return alpha * model.predict(x) + (1.0 - alpha) * center
    raise ValueError(f"unsupported postprocess: {kind}")  # pragma: no cover


def format_outputs(predicted, task, classes):
    """Official JSONL prediction record for one sample (matches predict.py)."""
    if task == "regression":
        return {"outputs": {"value": float(predicted)}}
    return {"outputs": [{"label": label, "score": float(score)}
                        for label, score in zip(classes, predicted)]}


class RecenterStream:
    """Causal rolling re-standardiser for interactive streaming prediction.

    Batch scoring should use :func:`apply_postprocess`, which uses the full
    incoming batch (the platform evaluates in batch mode). This class exists so
    the predictor can still answer line-by-line without waiting for EOF.
    """

    def __init__(self, reference_mean, reference_scale, mode="batch_mean", window=2048, min_count=8):
        self.reference_mean = np.asarray(reference_mean, dtype=np.float64)
        self.reference_scale = np.asarray(reference_scale, dtype=np.float64)
        self.mode = mode
        self.window = max(int(window), 2)
        self.min_count = max(int(min_count), 2)
        self.rows = []
        self.count = 0

    def calibrate(self, row):
        row = np.asarray(row, dtype=np.float64)
        self.rows.append(row)
        if len(self.rows) > self.window:
            self.rows = self.rows[-self.window:]
        self.count += 1
        if self.count < self.min_count:
            return row
        stack = np.asarray(self.rows, dtype=np.float64)
        mean = stack.mean(axis=0)
        if self.mode == "batch_std":
            scale = stack.std(axis=0)
            scale = np.where(scale < 1e-9, 1.0, scale)
            return (row - mean) / scale * self.reference_scale + self.reference_mean
        return row - mean + self.reference_mean
