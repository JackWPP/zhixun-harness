"""Small deterministic checks for cache identity, isolation and parent lineage."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rsi.model import ExtendedModel, Model
from rsi.platform import plans
from rsi.platform.experiment_cache import ExperimentCache, file_sha256


def context(tmp_path: Path, *, classification=False, cache=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    x = np.arange(24, dtype=np.float64).reshape(12, 2) / 10
    y = (np.arange(12) % 2 if classification else 10 + np.arange(12, dtype=float))
    task = "classification" if classification else "regression"
    classes = ["a", "b"] if classification else []
    paths = [json.dumps([["key", key]]) for key in ("x", "z")]
    model = ExtendedModel(task, paths, classes, x, y, hidden=4, seed=13)
    model.fit(x, y, 1)
    source = tmp_path / "parent.npz"
    model.save(source)
    model.meta["_source_path"] = str(source)
    train = [{"id": f"train-{i}", "inputs": {"x": float(row[0]), "z": float(row[1])},
              "answer": classes[int(y[i])] if classification else float(y[i])}
             for i, row in enumerate(x)]
    validation = [{**row, "id": f"validation-{i}"} for i, row in enumerate(train[:5])]
    return plans.PlanContext.from_rows(
        "A" if classification else "B", model, train, validation,
        metric="macro_f1" if classification else "rmse",
        direction="higher" if classification else "lower", margin=0,
        classes=classes, experiment_cache=cache)


def test_same_training_is_reused_across_postprocessing_and_revalidated(tmp_path, monkeypatch):
    ctx = context(tmp_path, classification=True)
    calls = []
    original = Model.fit

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Model, "fit", counted)
    parameters = {"epochs": 2, "learning_rate": .001}
    first = plans.plan_for_proposal(ctx, SimpleNamespace(
        strategy="continue_train", params=parameters, postprocess={"type": "none"}))
    second = plans.plan_for_proposal(ctx, SimpleNamespace(
        strategy="continue_train", params=parameters,
        postprocess={"type": "recenter", "mode": "batch_mean", "weight": .2}))
    assert first["ok"] and second["ok"]
    assert len(calls) == 1
    assert not first["experiment_cache"]["cache_hit"]
    assert second["experiment_cache"]["cache_hit"]
    assert first["model"] is not second["model"]
    for key in first["model"].params:
        np.testing.assert_array_equal(first["model"].params[key], second["model"].params[key])
    expected, _, predictions = plans.evaluate(ctx, second["model"], second["postprocess"])
    assert second["values"] == expected
    np.testing.assert_array_equal(second["predicted"], predictions)
    # A changed validation projection still requires a new evaluation of the
    # cached training result, without rerunning the identical fit.
    shorter = plans.PlanContext.from_rows(
        "A", ctx.model, ctx.train_rows, ctx.val_rows[:2], metric="macro_f1",
        direction="higher", margin=0, classes=ctx.classes,
        experiment_cache=ctx.experiment_cache)
    third = plans.plan_for_proposal(shorter, SimpleNamespace(
        strategy="continue_train", params=parameters, postprocess={"type": "none"}))
    assert third["experiment_cache"]["cache_hit"] and len(calls) == 1
    assert len(third["predicted"]) == 2


def test_cache_does_not_mutate_parent_or_share_candidate_mutations(tmp_path):
    ctx = context(tmp_path)
    source = Path(ctx.model.meta["_source_path"])
    before = source.read_bytes()
    first, _ = plans.cached_continuation(ctx, {"epochs": 2})
    expected = first.params["w1"].copy()
    first.params["w1"][:] = 900
    second, metadata = plans.cached_continuation(ctx, {"epochs": 2})
    assert metadata["cache_hit"]
    np.testing.assert_array_equal(second.params["w1"], expected)
    assert source.read_bytes() == before
    np.testing.assert_array_equal(second.mean, ctx.model.mean)
    np.testing.assert_array_equal(second.scale, ctx.model.scale)
    assert second.meta["epochs"] == ctx.model.meta["epochs"] + 2
    assert second.meta["steps"] == ctx.model.meta["steps"] + 2
    assert second.meta["seed"] == ctx.model.meta["seed"]


@pytest.mark.parametrize("change", ["features", "labels", "order", "parent", "parameters", "seed", "code"])
def test_actual_training_identity_changes_invalidate_cache(tmp_path, change):
    ctx = context(tmp_path)
    cache = ExperimentCache()
    calls = []

    def fit():
        calls.append(1)
        return copy.deepcopy(ctx.model)

    arguments = dict(parent_path=Path(ctx.model.meta["_source_path"]), task_key="B",
                     x_train=ctx.x_train.copy(), y_train=ctx.y_train.copy(),
                     configuration={"epochs": 2, "seed": 13}, fit=fit, code_revision="code-a")
    _, first = cache.get_or_fit(**arguments)
    if change == "features":
        arguments["x_train"][0, 0] += .25
    elif change == "labels":
        arguments["y_train"][0] += .25
    elif change == "order":
        arguments["x_train"] = arguments["x_train"][::-1]
        arguments["y_train"] = arguments["y_train"][::-1]
    elif change == "parent":
        other = copy.deepcopy(ctx.model)
        other.params["b2"] += .25
        path = tmp_path / "different-parent.npz"
        other.save(path)
        arguments["parent_path"] = path
    elif change == "parameters":
        arguments["configuration"] = {"epochs": 3, "seed": 13}
    elif change == "seed":
        arguments["configuration"] = {"epochs": 2, "seed": 14}
    else:
        arguments["code_revision"] = "code-b"
    _, second = cache.get_or_fit(**arguments)
    assert len(calls) == 2 and not second["cache_hit"]
    assert first["cache_key"] != second["cache_key"]


def test_state_cache_survives_new_context_and_matches_uncached_fit(tmp_path):
    directory = tmp_path / "state" / "experiment-cache"
    ctx = context(tmp_path / "parent", cache=ExperimentCache(directory))
    params = {"epochs": 2, "learning_rate": .001}
    first, first_meta = plans.cached_continuation(ctx, params)
    ctx.experiment_cache = ExperimentCache(directory)
    second, second_meta = plans.cached_continuation(ctx, params)
    ctx.experiment_cache = ExperimentCache(enabled=False)
    uncached, uncached_meta = plans.cached_continuation(ctx, params)
    assert not first_meta["cache_hit"] and not uncached_meta["cache_hit"]
    assert second_meta["cache_hit"] and second_meta["cache_source"] == "state"
    for key in first.params:
        for member in ("params", "first", "second"):
            np.testing.assert_array_equal(getattr(first, member)[key], getattr(second, member)[key])
            np.testing.assert_array_equal(getattr(first, member)[key], getattr(uncached, member)[key])
    np.testing.assert_array_equal(first.predict(ctx.x_val), second.predict(ctx.x_val))
    np.testing.assert_array_equal(first.predict(ctx.x_val), uncached.predict(ctx.x_val))


@pytest.mark.parametrize("damage", ["json", "checksum", "checkpoint", "symlink"])
def test_corrupt_cache_is_safe_miss(tmp_path, damage):
    directory = tmp_path / "cache"
    ctx = context(tmp_path / "parent", cache=ExperimentCache(directory))
    _, metadata = plans.cached_continuation(ctx, {"epochs": 1})
    entry = directory / metadata["cache_key"]
    manifest = entry / "manifest.json"
    checkpoint = entry / "checkpoint.npz"
    if damage == "json":
        manifest.write_text("[not json")
    elif damage == "checksum":
        payload = json.loads(manifest.read_text())
        payload["checkpoint_sha256"] = "wrong"
        manifest.write_text(json.dumps(payload))
    elif damage == "checkpoint":
        checkpoint.write_bytes(b"not a checkpoint")
        payload = json.loads(manifest.read_text())
        payload["checkpoint_sha256"] = file_sha256(checkpoint)
        manifest.write_text(json.dumps(payload))
    else:
        checkpoint.unlink()
        checkpoint.symlink_to(ctx.model.meta["_source_path"])
    ctx.experiment_cache = ExperimentCache(directory)
    model, retry = plans.cached_continuation(ctx, {"epochs": 1})
    assert not retry["cache_hit"]
    assert model.meta["epochs"] == ctx.model.meta["epochs"] + 1


def test_cache_toggle_disables_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_V2_EXPERIMENT_CACHE", "0")
    ctx = context(tmp_path)
    _, first = plans.cached_continuation(ctx, {"epochs": 1})
    _, second = plans.cached_continuation(ctx, {"epochs": 1})
    assert first["cache_key"] == second["cache_key"]
    assert not second["cache_enabled"] and not second["cache_hit"]


def test_b_default_continues_parent_once_without_scratch(tmp_path, monkeypatch):
    monkeypatch.delenv("RSI_V2_SCRATCH_ARM", raising=False)
    monkeypatch.delenv("RSI_V2_B_EPOCHS", raising=False)
    monkeypatch.delenv("RSI_V2_B_LR", raising=False)
    ctx = context(tmp_path)
    calls = []
    original = Model.fit

    def counted(self, x, y, epochs, **kwargs):
        calls.append({"epochs": epochs, "initial_epochs": self.meta["epochs"],
                      "hidden": self.meta["hidden"], "seed": self.meta["seed"]})
        return original(self, x, y, epochs, **kwargs)

    monkeypatch.setattr(Model, "fit", counted)
    catalogue = plans.build_plans(ctx)
    assert [plan["kind"] for plan in catalogue] == ["retain", "continue_train"]
    assert calls == [{"epochs": 15, "initial_epochs": 1, "hidden": 4, "seed": 13}]
    assert plans.scratch_plans(ctx) == []
    chosen = plans.b_continuation_plan(ctx)
    assert chosen["ok"] and chosen["experiment_cache"]["cache_hit"]
    assert len(calls) == 1
    monkeypatch.setenv("RSI_V2_SCRATCH_ARM", "B")
    assert plans._scratch_enabled("B") and not plans._scratch_enabled("C")


def test_retain_uses_context_parent_prediction_once(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    calls = []
    original = ExtendedModel.predict

    def counted(self, x):
        calls.append(len(x))
        return original(self, x)

    monkeypatch.setattr(ExtendedModel, "predict", counted)
    fresh = plans.PlanContext.from_rows(
        "B", ctx.model, ctx.train_rows, ctx.val_rows, metric="rmse", direction="lower", margin=0)
    retained = plans.retain_plan(fresh)
    assert calls == [len(ctx.val_rows)]
    assert retained["values"] == fresh.parent_values
    assert retained["primary"] == fresh.parent_primary
    np.testing.assert_array_equal(retained["predicted"], fresh.parent_predicted)
