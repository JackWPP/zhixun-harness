import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rsi.inference import apply_postprocess
from rsi.model import ExtendedModel
from rsi.platform.plans import PlanContext, plan_for_proposal, retain_plan


def parent(tmp_path, task="classification"):
    x = np.r_[np.full(12, -1.), np.ones(12)][:, None]
    y = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
    classes = ["left", "right"] if task == "classification" else []
    if task == "regression":
        y = np.full(24, 100.)
    model = ExtendedModel(task, ['[["key","x"]]'], classes, x, y, hidden=1)
    model.params["w1"][:] = 1.
    model.params["b1"][:] = 0.
    model.params["w2"][:] = [-2., 2.] if classes else 1.
    model.params["b2"][:] = 0.
    checkpoint = tmp_path / "checkpoint.npz"
    model.save(checkpoint)
    model.meta["_source_path"] = str(checkpoint)
    rows = [{"id": "row-"+str(i), "inputs": {"x": float(row[0])},
             "answer": classes[int(label)] if classes else float(label)}
            for i, (row, label) in enumerate(zip(x, y))]
    return model, x, rows


def metadata(tmp_path, model, postprocess):
    checkpoint = tmp_path / "checkpoint.npz"
    payload = {"candidate_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "task": model.meta["task"], "classes": model.meta["classes"],
               "postprocess": postprocess}
    (tmp_path / "submission_meta.json").write_text(json.dumps(payload))
    return payload


def context(model, rows):
    classification = model.meta["task"] == "classification"
    return PlanContext.from_rows("A" if classification else "C", model, rows, rows,
                                 metric="macro_f1" if classification else "rmse",
                                 direction="higher" if classification else "lower", margin=.004)


def test_actual_parent_threshold_is_preserved_by_context_and_retain_routes(tmp_path):
    model, x, rows = parent(tmp_path)
    spec = {"type": "pair_threshold", "pair": [0, 1], "delta": 9.123456789,
            "base": {"type": "none"}, "source": "history_labels"}
    metadata(tmp_path, model, spec)
    ctx = context(model, rows)
    expected = apply_postprocess(spec, model, x)
    assert not np.array_equal(expected, model.predict(x))
    assert ctx.parent_postprocess == spec
    np.testing.assert_array_equal(ctx.parent_predicted, expected)
    assert ctx.parent_values["macro_f1"] == pytest.approx(1/3)
    for plan in (retain_plan(ctx), plan_for_proposal(ctx, SimpleNamespace(strategy="retain"))):
        assert plan["postprocess"] == spec
        np.testing.assert_array_equal(plan["predicted"], expected)
        assert plan["primary"] == ctx.parent_primary


def test_stored_regression_center_is_not_replaced_or_rounded(tmp_path):
    model, x, rows = parent(tmp_path, "regression")
    spec = {"type": "shrink", "alpha": .3, "center": 123.123456789}
    metadata(tmp_path, model, spec)
    ctx = context(model, rows)
    assert ctx.parent_postprocess == spec
    np.testing.assert_array_equal(ctx.parent_predicted, .3*model.predict(x)+.7*spec["center"])


@pytest.mark.parametrize("spec", [None, {}, {"type": "none"}])
def test_digest_bound_none_metadata_remains_compatible(tmp_path, spec):
    model, x, rows = parent(tmp_path)
    metadata(tmp_path, model, spec)
    ctx = context(model, rows)
    assert ctx.parent_postprocess == {"type": "none"}
    np.testing.assert_array_equal(ctx.parent_predicted, model.predict(x))


def test_official_initial_checkpoint_without_metadata_defaults_to_none(tmp_path):
    model, x, rows = parent(tmp_path)
    ctx = context(model, rows)
    assert ctx.parent_postprocess == {"type": "none"}
    np.testing.assert_array_equal(ctx.parent_predicted, model.predict(x))


def test_wrong_checkpoint_digest_is_rejected(tmp_path):
    model, _, rows = parent(tmp_path)
    payload = metadata(tmp_path, model, {"type": "none"})
    payload["candidate_checkpoint_sha256"] = "0"*64
    (tmp_path / "submission_meta.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        context(model, rows)


@pytest.mark.parametrize("spec", [[], "none", {"type": "unknown"},
    {"type": "pair_threshold", "pair": [0, 1, 1], "delta": 1.},
    {"type": "pair_threshold", "pair": [0, 2], "delta": 1.},
    {"type": "pair_threshold", "pair": [0, 0], "delta": 1.},
    {"type": "pair_threshold", "pair": [0, 1], "delta": float("nan")},
    {"type": "recenter", "weight": [1.]},
    {"type": "recenter", "weight": 2.},
    {"type": "shrink", "alpha": .5, "center": 100.},
])
def test_illegal_parent_postprocess_is_rejected(tmp_path, spec):
    model, _, rows = parent(tmp_path)
    metadata(tmp_path, model, spec)
    with pytest.raises(ValueError):
        context(model, rows)


@pytest.mark.parametrize("payload", ["null", "[]", "{broken", '{"postprocess":null}'])
def test_invalid_or_unsigned_metadata_is_not_silently_treated_as_none(tmp_path, payload):
    model, _, rows = parent(tmp_path)
    (tmp_path / "submission_meta.json").write_text(payload)
    with pytest.raises(ValueError):
        context(model, rows)
