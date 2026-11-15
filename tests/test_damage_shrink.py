"""Damage-aware RUL calibration: fitted on official train, applied on ordered stream."""
import numpy as np
import pytest

from rsi.data import load_jsonl_dataset
from rsi.inference import (DamageShrinkStream, apply_postprocess, predict_for_split,
                           resolve_postprocess)
from rsi.model import ExtendedModel
from rsi.tasks import resolve_tasks


@pytest.fixture(scope="module")
def b_context(tmp_path_factory):
    cfg = resolve_tasks(["B"])[0]
    model = ExtendedModel.from_checkpoint(cfg.checkpoint_path)
    train = load_jsonl_dataset(cfg.train_path, "regression", ())
    validation = load_jsonl_dataset(cfg.validation_path, "regression", (),
                                    model.meta["paths"])
    return cfg, model, train, validation


def test_fit_is_train_only_and_shaped(b_context):
    cfg, model, train, _ = b_context
    spec = resolve_postprocess({"type": "damage_shrink", "a": 0.5, "w": 0.6,
                                "fit": "train"}, model, train=train)
    assert spec["type"] == "damage_shrink"
    assert spec["fit_on"] == "train"
    assert spec["calibration_rows"] == 0
    assert spec["center"] == pytest.approx(float(model.meta["target_mean"]), abs=1e-6)
    assert len(spec["robust_scale"]) == len(model.meta["paths"])
    assert "target" not in spec and "labels" not in spec


def test_damage_shrink_beats_shrink_on_b(b_context, tmp_path):
    cfg, model, train, validation = b_context
    from rsi.splits import build_splits
    splits = build_splits(cfg, tmp_path / "splits", seed=42)
    spec = resolve_postprocess({"type": "damage_shrink", "a": 0.5, "w": 0.6,
                                "fit": "train"}, model, train=train)
    sel = splits["selection"]
    predicted = predict_for_split(spec, model, validation, sel)
    rmse = float(np.sqrt(np.mean((predicted - sel.y) ** 2)))
    assert rmse < 430.0, rmse


def test_stream_matches_batch_prefix(b_context):
    cfg, model, train, validation = b_context
    spec = resolve_postprocess({"type": "damage_shrink", "a": 0.5, "w": 0.6,
                                "fit": "train"}, model, train=train)
    batch = apply_postprocess(spec, model, validation.x)
    stream = DamageShrinkStream(spec)
    for i in range(len(validation.x)):
        raw = model.predict(validation.x[i:i + 1])[0]
        value = stream.transform(validation.x[i], raw)
        assert value == pytest.approx(batch[i], abs=1e-6)
