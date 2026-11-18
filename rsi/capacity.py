"""Capacity extensions for the contest NumPy MLP.

The official baseline checkpoint is a one-hidden-layer tanh MLP (``Model`` in
:mod:`rsi.model`).  The competition explicitly allows modifying the loss and the
hidden layers, so this module adds an *opt-in* extension that keeps the official
checkpoint format readable while adding:

* classification losses: CE, label smoothing, focal, logit adjustment,
  class-balanced weighting and an Ethanol/Acetaldehyde auxiliary BCE term;
* architectures: wider single hidden layer, two hidden layers, ReLU/tanh and
  optional inverted dropout during training.

Baseline behaviour is untouched: this model is only constructed when a caller
explicitly asks for a capacity specification, and its checkpoints use the
``contest.cpu-mlp.v2`` format tag so the plain :class:`rsi.model.Model` loader
is never asked to interpret them by accident.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from rsi.model import (
    FORMAT,
    MAX_CHECKPOINT_BYTES,
    Model,
)

CAPACITY_FORMAT = "contest.cpu-mlp.v2"
LOSS_TYPES = ("ce", "label_smoothing", "focal", "class_balanced",
              "logit_adjusted", "pair_aux")
ACTIVATIONS = ("tanh", "relu")


def normalize_loss(spec) -> dict:
    """Return a validated loss specification with defaults filled in."""
    if spec is None:
        spec = {"type": "ce"}
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict):
        raise ValueError("loss specification must be a dict or string")
    kind = str(spec.get("type", "ce")).strip().lower()
    if kind not in LOSS_TYPES:
        raise ValueError(f"unsupported loss type: {kind!r}")
    out = dict(spec)
    out["type"] = kind
    if kind == "label_smoothing":
        eps = float(out.get("eps", 0.1))
        if not 0.0 <= eps < 1.0:
            raise ValueError("label smoothing epsilon must be in [0, 1)")
        out["eps"] = eps
    elif kind == "focal":
        gamma = float(out.get("gamma", 1.0))
        if not 0.0 <= gamma <= 5.0:
            raise ValueError("focal gamma must be in [0, 5]")
        alpha = float(out.get("alpha", 1.0))
        if not 0.0 < alpha <= 1.0:
            raise ValueError("focal alpha must be in (0, 1]")
        out["gamma"], out["alpha"] = gamma, alpha
    elif kind == "class_balanced":
        beta = float(out.get("beta", 0.999))
        if not 0.0 <= beta < 1.0:
            raise ValueError("class-balanced beta must be in [0, 1)")
        out["beta"] = beta
    elif kind == "logit_adjusted":
        tau = float(out.get("tau", 1.0))
        if not 0.0 <= tau <= 3.0:
            raise ValueError("logit adjustment tau must be in [0, 3]")
        out["tau"] = tau
    elif kind == "pair_aux":
        lam = float(out.get("lambda", 0.2))
        if not 0.0 <= lam <= 2.0:
            raise ValueError("pair auxiliary lambda must be in [0, 2]")
        pair = [int(v) for v in out.get("pair", [0, 3])]
        if len(pair) != 2 or pair[0] == pair[1] or min(pair) < 0:
            raise ValueError("pair auxiliary requires two distinct class indices")
        out["lambda"], out["pair"] = lam, pair
        base = out.get("base", {"type": "ce"})
        if not isinstance(base, dict) or str(base.get("type", "ce")) != "ce":
            raise ValueError("pair_aux currently supports a CE base loss only")
        out["base"] = {"type": "ce"}
    return out


def capacity_spec(hidden=32, hidden2=0, activation="tanh", dropout=0.0,
                  loss=None) -> dict:
    """Validated, JSON-friendly capacity descriptor used by the lab/plans."""
    hidden = int(hidden)
    hidden2 = int(hidden2)
    activation = str(activation or "tanh").lower()
    if not 1 <= hidden <= 128:
        raise ValueError("hidden must be in [1, 128]")
    if not 0 <= hidden2 <= 128:
        raise ValueError("hidden2 must be in [0, 128]")
    if activation not in ACTIVATIONS:
        raise ValueError(f"unsupported activation: {activation!r}")
    dropout = float(dropout)
    if not 0.0 <= dropout < 0.5:
        raise ValueError("dropout must be in [0, 0.5)")
    return {"hidden": hidden, "hidden2": hidden2, "activation": activation,
            "dropout": dropout, "loss": normalize_loss(loss)}


def _activation(name: str, x: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(x, 0.0)
    return np.tanh(x)


def _activation_grad(name: str, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    if name == "relu":
        return (pre > 0.0).astype(np.float64)
    return 1.0 - post ** 2


def _parameter_shapes(input_size: int, hidden: int, hidden2: int,
                      output: int) -> dict:
    shapes = {"w1": (input_size, hidden), "b1": (hidden,)}
    if hidden2:
        shapes.update(w2=(hidden, hidden2), b2=(hidden2,),
                      w3=(hidden2, output), b3=(output,))
    else:
        shapes.update(w2=(hidden, output), b2=(output,))
    return shapes


class CapacityModel(Model):
    """NumPy MLP with opt-in loss/architecture capacity."""

    def __init__(self, task, paths, classes, x, y, *, hidden=32, hidden2=0,
                 activation="tanh", dropout=0.0, loss=None, seed=42,
                 learning_rate=.003, batch_size=128):
        spec = capacity_spec(hidden, hidden2, activation, dropout, loss)
        hidden, hidden2 = spec["hidden"], spec["hidden2"]
        activation, dropout = spec["activation"], spec["dropout"]
        if task not in ("classification", "regression"):
            raise ValueError("invalid model configuration")
        if x.ndim != 2 or x.shape[1] != len(paths):
            raise ValueError("feature schema mismatch")
        output = len(classes) if task == "classification" else 1
        if task == "classification" and (len(classes) < 2 or len(set(classes)) != len(classes)):
            raise ValueError("classification needs distinct classes")
        shapes = _parameter_shapes(x.shape[1], hidden, hidden2, output)
        count = int(sum(int(np.prod(shape)) for shape in shapes.values()))
        if count >= 100000:
            raise ValueError("reference baseline must have fewer than 100,000 parameters")
        rng = np.random.default_rng(seed)
        self.params = {}
        previous = x.shape[1]
        for name, shape in shapes.items():
            if name.startswith("w"):
                self.params[name] = rng.normal(0.0, math.sqrt(1.0 / previous), shape)
                previous = shape[1]
            else:
                self.params[name] = np.zeros(shape)
        self.first = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.second = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        self.meta = {
            "format": CAPACITY_FORMAT, "device": "cpu", "task": task,
            "paths": paths, "classes": list(classes), "hidden": hidden,
            "hidden2": hidden2, "activation": activation, "dropout": dropout,
            "loss": spec["loss"], "seed": seed, "epochs": 0, "steps": 0,
            "learning_rate": learning_rate, "batch_size": batch_size,
            "parameter_count": count,
            "target_mean": float(y.mean()) if task == "regression" else 0.0,
            "target_scale": max(float(y.std()), 1.0) if task == "regression" else 1.0,
        }
        self.validate()

    # -- construction from an existing checkpoint --------------------------
    @classmethod
    def from_model(cls, model, spec: dict, seed=42):
        """Create a capacity model initialised from ``model`` when shapes fit.

        Widening / adding a layer keeps the overlapping weights and re-seeds
        the new parameters, which makes "extend the parent" experiments
        comparable to the harness' continuation path.
        """
        spec = capacity_spec(**spec)
        rng = np.random.default_rng(seed)
        x_reference = np.zeros((1, len(model.meta["paths"])), dtype=np.float64)
        # Build with dummy data, then copy the overlapping parameter blocks.
        dummy = cls(model.meta["task"], model.meta["paths"], model.meta["classes"],
                    x_reference, np.zeros(1), seed=seed,
                    learning_rate=float(model.meta.get("learning_rate", 0.003)),
                    batch_size=int(model.meta.get("batch_size", 128)), **spec)
        parent = model.params
        for name, value in dummy.params.items():
            source = parent.get(name)
            if source is None:
                continue
            if source.shape == value.shape:
                value[...] = source
                continue
            # Copy the largest common leading block (works for widening and
            # for adding a second layer to a single-hidden-layer parent).
            slices = tuple(slice(0, min(a, b)) for a, b in zip(source.shape, value.shape))
            value[slices] = source[slices]
        dummy.mean = np.array(model.mean, dtype=np.float64, copy=True)
        dummy.scale = np.array(model.scale, dtype=np.float64, copy=True)
        dummy.meta["parent_seed"] = int(model.meta.get("seed", 0) or 0)
        dummy.validate()
        return dummy

    # -- validation --------------------------------------------------------
    def validate(self):
        m = self.meta
        if m.get("format") != CAPACITY_FORMAT or m.get("device") != "cpu" or m.get("task") not in ("classification", "regression"):
            raise ValueError("unsupported checkpoint")
        for name in ("seed", "epochs", "steps", "hidden", "hidden2", "batch_size", "parameter_count"):
            if type(m.get(name)) is not int or m[name] < 0:
                raise ValueError("invalid checkpoint integer")
        if not 1 <= m["hidden"] <= 128 or not 0 <= m["hidden2"] <= 128:
            raise ValueError("invalid checkpoint architecture")
        if not 1 <= m["batch_size"] <= 65536 or not 0 < m["parameter_count"] < 100000:
            raise ValueError("invalid checkpoint architecture")
        if m.get("activation") not in ACTIVATIONS:
            raise ValueError("invalid activation")
        dropout = m.get("dropout", 0.0)
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 0.5:
            raise ValueError("invalid dropout")
        m["loss"] = normalize_loss(m.get("loss"))
        if not 0 <= m["seed"] < 2**32:
            raise ValueError("seed must fit an unsigned 32-bit integer")
        if not isinstance(m.get("paths"), list) or not 1 <= len(m["paths"]) <= 4096 or any(not isinstance(p, str) for p in m["paths"]) or len(set(m["paths"])) != len(m["paths"]):
            raise ValueError("invalid checkpoint feature schema")
        classes = m.get("classes")
        if not isinstance(classes, list) or any(not isinstance(c, str) or not c for c in classes) or len(set(classes)) != len(classes):
            raise ValueError("invalid checkpoint classes")
        if (m["task"] == "classification" and len(classes) < 2) or (m["task"] == "regression" and classes):
            raise ValueError("invalid checkpoint classes")
        for name in ("learning_rate", "target_mean", "target_scale"):
            if isinstance(m.get(name), bool) or not isinstance(m.get(name), (float, int)) or not math.isfinite(m[name]):
                raise ValueError("invalid checkpoint numeric state")
        if not 0 < m["learning_rate"] <= 1 or m["target_scale"] <= 0:
            raise ValueError("invalid optimizer/scaler state")
        input_size = len(m["paths"])
        output = len(classes) if m["task"] == "classification" else 1
        shapes = _parameter_shapes(input_size, m["hidden"], m["hidden2"], output)
        for group in (self.params, self.first, self.second):
            if set(group) != set(shapes):
                raise ValueError("checkpoint tensors missing")
            for name, shape in shapes.items():
                if group[name].shape != shape or group[name].dtype != np.dtype("float64") or not np.all(np.isfinite(group[name])):
                    raise ValueError("checkpoint tensor shape or values invalid")
        if sum(value.size for value in self.params.values()) != m["parameter_count"]:
            raise ValueError("checkpoint parameter count mismatch")
        if self.mean.dtype != np.dtype("float64") or self.scale.dtype != np.dtype("float64") or self.mean.shape != (input_size,) or self.scale.shape != (input_size,) or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("checkpoint normalization invalid")
        if any(np.any(value < 0) for value in self.second.values()):
            raise ValueError("invalid Adam second moment")

    # -- forward -----------------------------------------------------------
    def _forward_cached(self, x, *, training=False, rng=None):
        layer_names = [(name, "b" + name[1:]) for name in self.params if name.startswith("w")]
        # Keep declaration order w1,w2,w3.
        layer_names.sort(key=lambda item: int(item[0][1:]))
        activation = str(self.meta.get("activation", "tanh"))
        dropout = float(self.meta.get("dropout", 0.0)) if training else 0.0
        caches = []
        h = x
        for index, (w_name, b_name) in enumerate(layer_names):
            pre = h @ self.params[w_name] + self.params[b_name]
            is_last = index == len(layer_names) - 1
            if is_last:
                return caches, pre
            post = _activation(activation, pre)
            mask = None
            if dropout > 0.0:
                keep = 1.0 - dropout
                if rng is None:
                    raise ValueError("dropout requires an rng")
                mask = (rng.random(post.shape) < keep).astype(np.float64) / keep
                post = post * mask
            caches.append((h, pre, post, mask))
            h = post
        raise ValueError("checkpoint has no output layer")

    def forward(self, x):
        caches, logits = self._forward_cached(x, training=False)
        hidden = caches[-1][2] if caches else x
        return hidden, logits

    def hidden(self, x):
        caches, _ = self._forward_cached(self.normalized(x), training=False)
        return caches[-1][2] if caches else self.normalized(x)

    def predict_logits(self, x):
        _, logits = self._forward_cached(self.normalized(x), training=False)
        return logits

    # -- loss --------------------------------------------------------------
    def _class_weights(self, y):
        counts = np.bincount(y, minlength=len(self.meta["classes"])).astype(np.float64)
        beta = float(self.meta["loss"].get("beta", 0.999))
        weights = np.where(counts > 0, (1.0 - beta) / (1.0 - np.power(beta, np.maximum(counts, 1.0))), 1.0)
        weights /= weights.mean()
        return weights

    def _classification_delta(self, logits, y, loss, class_weight):
        n = len(y)
        classes = len(self.meta["classes"])
        rows = np.arange(n)
        kind = loss["type"]
        adjusted = logits
        prior = None
        if kind == "logit_adjusted":
            counts = np.bincount(y, minlength=classes).astype(np.float64)
            prior = (counts + 1.0) / (counts.sum() + classes)
            adjusted = logits + float(loss.get("tau", 1.0)) * np.log(prior)
        shifted = adjusted - adjusted.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        onehot = np.zeros_like(probabilities)
        onehot[rows, y] = 1.0
        if class_weight is not None:
            sample_weight = np.asarray(class_weight, dtype=np.float64)[y]
        elif kind == "class_balanced":
            sample_weight = self._class_weights(y)[y]
        else:
            sample_weight = np.ones(n, dtype=np.float64)
        if kind == "label_smoothing":
            eps = float(loss.get("eps", 0.1))
            target = (1.0 - eps) * onehot + eps / classes
            delta = (probabilities - target) * sample_weight[:, None]
            delta /= max(float(sample_weight.sum()), 1e-12)
        elif kind == "focal":
            gamma = float(loss.get("gamma", 1.0))
            alpha = float(loss.get("alpha", 1.0))
            p_t = np.maximum(probabilities[rows, y], 1e-12)
            one_minus = np.maximum(1.0 - p_t, 1e-12)
            coefficient = gamma * np.power(one_minus, gamma - 1.0) * p_t * np.log(p_t) - np.power(one_minus, gamma)
            delta = coefficient[:, None] * (onehot - probabilities)
            delta *= alpha * sample_weight[:, None]
            delta /= max(float(sample_weight.sum()), 1e-12)
        else:
            delta = (probabilities - onehot) * sample_weight[:, None]
            delta /= max(float(sample_weight.sum()), 1e-12)
        loss_value = 0.0
        if kind == "focal":
            p_t = np.maximum(probabilities[rows, y], 1e-12)
            loss_value = -float((sample_weight * np.power(1.0 - p_t, float(loss.get("gamma", 1.0))) * np.log(p_t)).sum())
        else:
            loss_value = -float((sample_weight * np.log(np.maximum(probabilities[rows, y], 1e-300))).sum())
        if kind == "pair_aux":
            left, right = (int(v) for v in loss.get("pair", [0, 3]))
            lam = float(loss.get("lambda", 0.2))
            score = logits[:, right] - logits[:, left]
            target = (y == right).astype(np.float64)
            probs = 1.0 / (1.0 + np.exp(-np.clip(score, -50, 50)))
            weight = np.isin(y, [left, right]).astype(np.float64)
            grad = lam * (probs - target) * weight
            delta[rows, right] += grad
            delta[rows, left] -= grad
            bce = -(target * np.log(np.maximum(probs, 1e-12)) + (1 - target) * np.log(np.maximum(1 - probs, 1e-12)))
            loss_value += lam * float((bce * weight).sum())
            loss_value /= max(weight.sum(), 1.0)
        return delta, loss_value

    def set_learning_rate(self, value):
        value = float(value)
        if not 0 < value <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        self.meta["learning_rate"] = value
        return self

    def fit(self, x, y, epochs, class_weight=None):
        if type(epochs) is not int or not 1 <= epochs <= 10000:
            raise ValueError("epochs must be between 1 and 10000")
        x = self.normalized(x)
        if y.shape != (len(x),) or not len(x):
            raise ValueError("target shape mismatch")
        if self.meta["task"] == "regression":
            target = (y - self.meta["target_mean"]) / self.meta["target_scale"]
            if class_weight is not None:
                raise ValueError("class weights are only supported for classification")
            loss = {"type": "ce"}
        else:
            if np.any(y < 0) or np.any(y >= len(self.meta["classes"])):
                raise ValueError("unknown class")
            target = np.asarray(y, dtype=np.int64)
            loss = normalize_loss(self.meta.get("loss"))
        weights = None
        if class_weight is not None:
            weights = np.asarray(class_weight, dtype=np.float64)
            if weights.shape != (len(self.meta["classes"]),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError("invalid class weights")
        epoch_losses = []
        for _ in range(epochs):
            rng = np.random.default_rng(np.random.SeedSequence([self.meta["seed"], self.meta["epochs"]]))
            order = rng.permutation(len(x))
            loss_sum = 0.0
            for start in range(0, len(order), self.meta["batch_size"]):
                indices = order[start:start + self.meta["batch_size"]]
                batch, expected = x[indices], target[indices]
                caches, logits = self._forward_cached(batch, training=True, rng=rng)
                if self.meta["task"] == "classification":
                    delta, batch_loss = self._classification_delta(logits, expected, loss, weights)
                else:
                    error = logits[:, 0] - expected
                    batch_loss = float(np.sum(error ** 2))
                    delta = (2.0 * error / len(indices))[:, None]
                loss_sum += batch_loss
                # Backprop through the stacked layers.
                gradients = {}
                layer_names = [(name, "b" + name[1:]) for name in self.params if name.startswith("w")]
                layer_names.sort(key=lambda item: int(item[0][1:]))
                delta_z = delta
                for layer_index in range(len(layer_names) - 1, -1, -1):
                    w_name, b_name = layer_names[layer_index]
                    if layer_index == 0:
                        previous = batch
                    else:
                        previous = caches[layer_index - 1][2]
                    gradients[w_name] = previous.T @ delta_z
                    gradients[b_name] = delta_z.sum(axis=0)
                    if layer_index == 0:
                        break
                    _, pre, post, mask = caches[layer_index - 1]
                    grad = (delta_z @ self.params[w_name].T) * _activation_grad(
                        str(self.meta.get("activation", "tanh")), pre, post)
                    if mask is not None:
                        grad = grad * mask
                    delta_z = grad
                norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
                if not math.isfinite(norm):
                    raise ValueError("nonfinite gradient")
                self.meta["steps"] += 1
                for name, gradient in gradients.items():
                    gradient *= min(1.0, 5.0 / max(norm, 1e-12))
                    self.first[name] = .9 * self.first[name] + .1 * gradient
                    self.second[name] = .999 * self.second[name] + .001 * gradient ** 2
                    first = self.first[name] / (1 - .9 ** self.meta["steps"])
                    second = self.second[name] / (1 - .999 ** self.meta["steps"])
                    self.params[name] -= self.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
                if not all(np.all(np.isfinite(p)) for p in self.params.values()):
                    raise ValueError("nonfinite model state")
            self.meta["epochs"] += 1
            epoch_losses.append(loss_sum / len(x))
        self.validate()
        return epoch_losses

    # -- checkpoint I/O ----------------------------------------------------
    def save(self, path):
        self.validate()
        arrays = {**self.params,
                  **{"adam_first_" + k: v for k, v in self.first.items()},
                  **{"adam_second_" + k: v for k, v in self.second.items()},
                  "mean": self.mean, "scale": self.scale,
                  "metadata": np.array(json.dumps(self.meta, sort_keys=True, separators=(",", ":"), allow_nan=False))}
        with Path(path).open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint must be a regular file no larger than 16 MiB")
        with np.load(path, allow_pickle=False) as arrays:
            if "metadata" not in arrays:
                raise ValueError("checkpoint metadata missing")
            metadata = arrays["metadata"]
            if metadata.ndim != 0 or metadata.dtype.kind != "U":
                raise ValueError("invalid checkpoint metadata")
            meta = json.loads(str(metadata))
            if meta.get("format") != CAPACITY_FORMAT:
                raise ValueError("not a capacity checkpoint")
            input_size = len(meta.get("paths") or [])
            output = len(meta.get("classes") or []) if meta.get("task") == "classification" else 1
            shapes = _parameter_shapes(input_size, int(meta.get("hidden", 32)),
                                       int(meta.get("hidden2", 0)), output)
            expected = {*shapes, *("adam_first_" + k for k in shapes),
                        *("adam_second_" + k for k in shapes), "mean", "scale", "metadata"}
            names = set(arrays.files)
            if names != expected:
                raise ValueError("invalid checkpoint archive")
            model = cls.__new__(cls)
            model.meta = meta
            model.params = {k: arrays[k].copy() for k in shapes}
            model.first = {k: arrays["adam_first_" + k].copy() for k in shapes}
            model.second = {k: arrays["adam_second_" + k].copy() for k in shapes}
            model.mean, model.scale = arrays["mean"].copy(), arrays["scale"].copy()
        model.validate()
        return model


def build_model(parent, spec: dict, *, seed=42):
    """Build a capacity model either from a parent checkpoint or from rows.

    ``parent`` is an :class:`rsi.model.Model`; the returned model is initialized
    from it (shape-compatible blocks copied).  The caller still has to call
    ``fit`` with the training rows.
    """
    return CapacityModel.from_model(parent, capacity_spec(**spec), seed=seed)
