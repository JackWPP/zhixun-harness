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
damage_shrink
    Bearing-style degradation calibration for RUL regression. A causal per-segment
    damage score ``score = max(rel[dim0], rel[dim1])`` with
    ``rel = (x - mean(first `early` rows of the segment)) / max(|mean|, floor)``
    is mapped through a step table fitted on **official train labels only**, then
    blended with a train-centre shrink::

        y = (1 - w) * (a * raw + (1 - a) * center) + w * curve(score)

    Segment boundaries are detected causally from the input stream (a sharp level
    drop starts a new segment). If the stream looks unstable (too many resets),
    the predictor falls back to the plain train-centre shrink.
"""
from __future__ import annotations

import numpy as np

POSTPROCESS_KINDS = ("none", "recenter", "shrink", "damage_shrink", "pair_threshold")
DAMAGE_SCORE_DIMS = (0, 17)
DAMAGE_DEFAULT_DETECTOR = {"rel": 0.15, "ratio": 0.10, "min_rows": 8, "window": 20}


def _group_ranges(groups):
    """Contiguous index ranges for one sequence per group (input order preserved)."""
    if not groups:
        return [(0, 0)]
    ranges = []
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            ranges.append((start, i))
            start = i
    return ranges


def _segment_scores(x, robust_scale, early=10, dims=DAMAGE_SCORE_DIMS):
    """Causal rel scores for one contiguous segment."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    floor = np.maximum(0.1 * np.asarray(robust_scale, dtype=np.float64), 1e-9)
    base = np.empty_like(x)
    for t in range(n):
        base[t] = x[: min(t + 1, early)].mean(axis=0)
    rel = (x - base) / np.maximum(np.abs(base), floor)
    return np.max(rel[:, list(dims)], axis=1)


def fit_damage_shrink(train, spec=None, bins=10):
    """Fit the causal damage curve on the official train split only.

    ``train`` is a dataset with ``x``, ``y`` and ``groups`` (contiguous per
    bearing/unit). Returns a resolved damage_shrink spec.
    """
    spec = dict(spec or {})
    x = np.asarray(train.x, dtype=np.float64)
    y = np.asarray(train.y, dtype=np.float64).reshape(-1)
    groups = list(getattr(train, "groups", None) or ["all"] * len(x))
    early = int(spec.get("early", 10))
    dims = tuple(spec.get("dims") or DAMAGE_SCORE_DIMS)
    robust_scale = np.maximum(np.median(np.abs(x), axis=0), 1e-9)
    scores, labels = [], []
    for lo, hi in _group_ranges(groups):
        if hi - lo < 2:
            continue
        scores.append(_segment_scores(x[lo:hi], robust_scale, early=early, dims=dims))
        labels.append(y[lo:hi])
    if not scores:
        raise ValueError("damage_shrink fit needs at least one train sequence")
    score = np.concatenate(scores)
    label = np.concatenate(labels)
    quantiles = np.linspace(1.0 / bins, 1.0 - 1.0 / bins, bins - 1)
    edges = np.unique(np.quantile(score, quantiles))
    index = np.clip(np.digitize(score, edges), 0, len(edges))
    values = np.array([label[index == b].mean() if np.any(index == b) else label.mean()
                       for b in range(len(edges) + 1)])
    detector = dict(DAMAGE_DEFAULT_DETECTOR)
    detector.update(spec.get("detector") or {})
    resolved = {
        "type": "damage_shrink",
        "a": round(float(spec.get("a", 0.5)), 6),
        "w": round(float(spec.get("w", 0.6)), 6),
        "center": round(float(spec.get("center", 0.0)), 6),
        "center_source": "train_target_mean",
        "early": early,
        "dims": list(dims),
        "robust_scale": [float(v) for v in robust_scale],
        "curve": {"kind": "quantile_bins_step", "edges": [float(v) for v in edges],
                  "values": [float(v) for v in values]},
        "detector": detector,
        "fit_on": "train",
        "calibration_rows": 0,
        "fallback_alpha": round(float(spec.get("fallback_alpha", 0.3)), 6),
    }
    return resolved


def _damage_segments(x, spec):
    """Causal segment boundaries and per-row damage scores."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    detector = spec.get("detector") or DAMAGE_DEFAULT_DETECTOR
    dims = list(spec.get("dims") or DAMAGE_SCORE_DIMS)
    early = int(spec.get("early", 10))
    min_rows = int(detector.get("min_rows", 8))
    window = int(detector.get("window", 20))
    level_dims = dims if len(dims) >= 2 else [0, 17]
    breaks = [0]
    levels = [float(np.max(x[i, level_dims])) for i in range(n)] if n else []
    for i in range(1, n):
        if i - breaks[-1] < min_rows:
            continue
        prev = levels[i - 1]
        recent = max(levels[max(0, i - window):i])
        if prev > 0 and levels[i] < float(detector.get("rel", 0.15)) * prev and \
                levels[i] < float(detector.get("ratio", 0.10)) * recent:
            breaks.append(i)
    scores = np.zeros(n, dtype=np.float64)
    for lo, hi in zip(breaks, breaks[1:] + [n]):
        if hi > lo:
            scores[lo:hi] = _segment_scores(x[lo:hi], spec["robust_scale"],
                                            early=early, dims=dims)
    return scores, len(breaks) - 1


def apply_damage_shrink(spec, model, x):
    x = np.asarray(x, dtype=np.float64)
    raw = np.asarray(model.predict(x), dtype=np.float64).reshape(-1)
    scores, resets = _damage_segments(x, spec)
    suspicious = resets > max(2, len(x) // 200)
    if suspicious:
        alpha = float(spec.get("fallback_alpha", 0.3))
        return np.clip(alpha * raw + (1.0 - alpha) * float(spec["center"]), 0.0, None)
    edges = np.asarray(spec["curve"]["edges"], dtype=np.float64)
    values = np.asarray(spec["curve"]["values"], dtype=np.float64)
    curve = values[np.clip(np.digitize(scores, edges), 0, len(edges))]
    a = float(spec.get("a", 0.5))
    w = float(spec.get("w", 0.6))
    blended = (1.0 - w) * (a * raw + (1.0 - a) * float(spec["center"])) + w * curve
    return np.clip(blended, 0.0, None)


def resolve_postprocess(spec, model, calibration=None, train=None):
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
    if kind == "pair_threshold":
        if model.meta["task"] != "classification":
            raise ValueError("pair_threshold only applies to classification")
        classes = list(model.meta.get("classes") or [])
        pair = spec.get("pair", [0, 3])
        try:
            left, right = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("pair_threshold pair must be two class indices") from exc
        if left == right or not (0 <= left < len(classes)) or not (0 <= right < len(classes)):
            raise ValueError("pair_threshold pair must be two distinct valid class indices")
        if spec.get("delta") is None:
            raise ValueError("pair_threshold requires a fitted delta")
        delta = float(spec["delta"])
        if not np.isfinite(delta):
            raise ValueError("pair_threshold delta must be finite")
        base = spec.get("base") or {"type": "none"}
        resolved_base = resolve_postprocess(base, model, calibration=calibration, train=train)
        return {"type": "pair_threshold", "pair": [left, right], "delta": round(delta, 6),
                "base": resolved_base,
                "source": str(spec.get("source", "history_labels")),
                "history_rows": int(spec.get("history_rows", 0))}
    if kind == "damage_shrink":
        if model.meta["task"] != "regression":
            raise ValueError("damage_shrink only applies to regression")
        if "curve" not in spec or not spec["curve"].get("edges"):
            spec.setdefault("center", float(model.meta.get("target_mean", 0.0)))
            fitted = fit_damage_shrink(train, spec) if train is not None else None
            if fitted is None:
                raise ValueError("damage_shrink needs curve fields or a train dataset to fit")
            return fitted
        edges = [float(v) for v in spec["curve"].get("edges", [])]
        values = [float(v) for v in spec["curve"].get("values", [])]
        if not edges or len(values) != len(edges) + 1:
            raise ValueError("damage_shrink curve needs len(values) == len(edges) + 1")
        a = float(spec.get("a", 0.5))
        w = float(spec.get("w", 0.6))
        if not 0.0 <= a <= 1.0 or not 0.0 <= w <= 1.0:
            raise ValueError("damage_shrink a/w must be within [0, 1]")
        robust_scale = spec.get("robust_scale")
        if not robust_scale or len(robust_scale) != len(model.meta["paths"]):
            raise ValueError("damage_shrink needs robust_scale of checkpoint width")
        detector = dict(DAMAGE_DEFAULT_DETECTOR)
        detector.update(spec.get("detector") or {})
        return {"type": "damage_shrink", "a": round(a, 6), "w": round(w, 6),
                "center": round(float(spec.get("center", model.meta.get("target_mean", 0.0))), 6),
                "center_source": "train_target_mean", "early": int(spec.get("early", 10)),
                "dims": list(spec.get("dims") or DAMAGE_SCORE_DIMS),
                "robust_scale": [float(v) for v in robust_scale],
                "curve": {"kind": "quantile_bins_step", "edges": edges, "values": values},
                "detector": detector, "fit_on": spec.get("fit_on", "train"),
                "calibration_rows": 0,
                "fallback_alpha": round(float(spec.get("fallback_alpha", 0.3)), 6)}
    raise ValueError(f"unsupported postprocess: {kind}")  # pragma: no cover


def _recentered_features(model, x, mode):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=0)
    if mode == "batch_std":
        scale = np.std(x, axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        return (x - mean) / scale * model.scale + model.mean
    return x - mean + model.mean


def fit_pair_threshold(x, y, model, pair=(0, 3), min_left=30, min_right=10, grid=161):
    """Fit a decision-boundary threshold for a class pair from revealed labels.

    Only labels already returned by the platform may be passed here; the batch
    being predicted must never be included. Returns ``None`` when there are not
    enough samples of both classes to calibrate safely.
    """
    if model.meta["task"] != "classification":
        return None
    y = np.asarray(y, dtype=np.int64)
    x = np.asarray(x, dtype=np.float64)
    if len(y) == 0:
        return None
    left, right = int(pair[0]), int(pair[1])
    if int((y == left).sum()) < int(min_left) or int((y == right).sum()) < int(min_right):
        return None
    logits = model.predict_logits(x)
    score = logits[:, right] - logits[:, left]
    predicted = np.argmax(logits, axis=1)
    mask = (y == left) | (y == right)
    if not mask.any():
        return None
    baseline = float(((predicted[y == left] == left).mean() + (predicted[y == right] == right).mean()) / 2.0)
    lo, hi = np.percentile(score[mask], [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None
    best_score, best_delta = -1.0, None
    for delta in np.linspace(lo, hi, int(grid)):
        flipped = predicted.copy()
        in_pair = (predicted == left) | (predicted == right)
        flipped[in_pair & (score < delta)] = right
        flipped[in_pair & (score >= delta)] = left
        score_now = float(((flipped[y == left] == left).mean() + (flipped[y == right] == right).mean()) / 2.0)
        if score_now > best_score:
            best_score, best_delta = score_now, float(delta)
    if best_delta is None or best_score < baseline + 0.02:
        return None
    return {"delta": best_delta, "score": best_score, "baseline_score": baseline,
            "rows": int(mask.sum()), "left_rows": int((y == left).sum()),
            "right_rows": int((y == right).sum())}


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
    if kind == "pair_threshold":
        if model.meta["task"] != "classification":
            raise ValueError("pair_threshold only applies to classification")
        base = spec.get("base") or {"type": "none"}
        output = apply_postprocess(base, model, x) if base.get("type", "none") != "none" else model.predict(x)
        output = np.array(output, dtype=np.float64, copy=True)
        left, right = (int(v) for v in spec["pair"])
        logits = model.predict_logits(x)
        score = logits[:, right] - logits[:, left]
        predicted = np.argmax(output, axis=1)
        mask = (predicted == left) | (predicted == right)
        target = np.where(score < float(spec["delta"]), right, left)
        changed = mask & (predicted != target)
        if changed.any():
            output[changed] = 0.0
            output[np.arange(len(output))[changed], target[changed]] = 1.0
        return output
    if kind == "damage_shrink":
        return apply_damage_shrink(spec, model, x)
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


class DamageShrinkStream:
    """Causal streaming version of :func:`apply_damage_shrink`.

    Mirrors the batch implementation row by row: it detects segment resets,
    keeps the early baseline of the current segment and returns the calibrated
    value for each incoming raw prediction.
    """

    def __init__(self, spec):
        self.spec = dict(spec)
        self.early = int(self.spec.get("early", 10))
        self.dims = list(self.spec.get("dims") or DAMAGE_SCORE_DIMS)
        self.robust_scale = np.asarray(self.spec["robust_scale"], dtype=np.float64)
        self.floor = np.maximum(0.1 * self.robust_scale, 1e-9)
        detector = self.spec.get("detector") or DAMAGE_DEFAULT_DETECTOR
        self.min_rows = int(detector.get("min_rows", 8))
        self.window = int(detector.get("window", 20))
        self.rel = float(detector.get("rel", 0.15))
        self.ratio = float(detector.get("ratio", 0.10))
        self.edges = np.asarray(self.spec["curve"]["edges"], dtype=np.float64)
        self.values = np.asarray(self.spec["curve"]["values"], dtype=np.float64)
        self.segment_rows = []
        self.level_history = []
        self.resets = 0

    def _maybe_reset(self, row):
        level = float(np.max(row[self.dims]))
        n = len(self.level_history)
        reset = False
        if n >= self.min_rows:
            prev = self.level_history[-1]
            recent = max(self.level_history[-self.window:])
            if prev > 0 and level < self.rel * prev and level < self.ratio * recent:
                reset = True
                self.resets += 1
        self.level_history.append(level)
        return reset

    def transform(self, row, raw):
        row = np.asarray(row, dtype=np.float64)
        if self._maybe_reset(row):
            self.segment_rows = []
        self.segment_rows.append(row)
        early_rows = np.asarray(self.segment_rows[: min(len(self.segment_rows), self.early)])
        base = early_rows.mean(axis=0)
        rel = (row - base) / np.maximum(np.abs(base), self.floor)
        score = float(np.max(rel[self.dims]))
        curve = float(self.values[np.clip(np.digitize([score], self.edges), 0,
                                          len(self.edges))][0])
        a = float(self.spec.get("a", 0.5))
        w = float(self.spec.get("w", 0.6))
        center = float(self.spec["center"])
        value = (1.0 - w) * (a * float(raw) + (1.0 - a) * center) + w * curve
        return max(value, 0.0)


def predict_for_split(postprocess, model, full_dataset, subset):
    """Postprocess the full ordered stream, then return predictions for a subset.

    Segment-aware postprocesses (damage_shrink) depend on stream order and
    segment context, so evaluating a shuffled split in isolation is wrong.
    The platform feeds the hidden stream in order; we mirror that by scoring the
    full ordered validation stream once and slicing by id.
    """
    predictions = apply_postprocess(postprocess, model, full_dataset.x)
    index = {rid: i for i, rid in enumerate(full_dataset.ids)}
    take = np.asarray([index[rid] for rid in subset.ids], dtype=np.int64)
    return predictions[take]
