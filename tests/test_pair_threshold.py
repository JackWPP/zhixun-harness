import numpy as np
import pytest

from rsi.inference import apply_postprocess, fit_pair_threshold, resolve_postprocess


class DummyModel:
    def __init__(self, logits):
        self._logits = np.asarray(logits, dtype=np.float64)
        self.meta = {
            "task": "classification",
            "classes": ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
        }
        self.mean = np.zeros(4)
        self.scale = np.ones(4)

    def predict_logits(self, x):
        return self._logits

    def predict(self, x):
        z = self._logits - self._logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def test_pair_threshold_flips_low_score_rows():
    # row0: strong Ethanol (s = logit3-logit0 = -6) -> flipped to AA
    # row1: weak Ethanol (s = -1) -> stays Ethanol
    logits = np.array([
        [ 3.0, 0, 0, -3.0, 0, 0],
        [ 3.0, 0, 0,  2.0, 0, 0],
    ])
    model = DummyModel(logits)
    spec = {"type": "pair_threshold", "pair": [0, 3], "delta": -3.9}
    resolved = resolve_postprocess(spec, model)
    out = apply_postprocess(resolved, model, np.zeros((2, 4)))
    assert out[0].argmax() == 3
    assert abs(out[0].sum() - 1.0) < 1e-9
    assert out[1].argmax() == 0


def test_pair_threshold_composes_with_base():
    logits = np.array([[3.0, 0, 0, -4.5, 0, 0]])
    model = DummyModel(logits)
    spec = {"type": "pair_threshold", "pair": [0, 3], "delta": -3.9,
            "base": {"type": "recenter", "mode": "batch_mean", "weight": 0.0}}
    resolved = resolve_postprocess(spec, model)
    out = apply_postprocess(resolved, model, np.zeros((1, 4)))
    assert out[0].argmax() == 3


def test_fit_pair_threshold_recovers_inverted_direction():
    rng = np.random.default_rng(0)
    n = 200
    y = np.array([0] * 80 + [3] * 120)
    s = np.r_[rng.normal(2.0, 1.0, 80), rng.normal(-2.0, 1.0, 120)]
    logits = np.zeros((n, 6))
    logits[:, 0] = -s / 2.0
    logits[:, 3] = s / 2.0
    logits[:, 1] = 0.2
    model = DummyModel(logits)
    fit = fit_pair_threshold(np.zeros((n, 4)), y, model, min_left=30, min_right=10)
    assert fit is not None
    assert fit["delta"] < 0
    assert fit["score"] > fit["baseline_score"]


def test_fit_pair_threshold_returns_none_without_both_classes():
    y = np.array([0] * 100)
    model = DummyModel(np.zeros((100, 6)))
    assert fit_pair_threshold(np.zeros((100, 4)), y, model) is None


def test_pair_threshold_only_for_classification():
    model = DummyModel(np.zeros((2, 6)))
    model.meta["task"] = "regression"
    with pytest.raises(ValueError):
        resolve_postprocess({"type": "pair_threshold", "pair": [0, 3], "delta": -1.0}, model)
