"""Exercise final promotion guards through the real train adapter, without fit."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rsi.metrics import from_predictions
from rsi.model import ExtendedModel
from rsi.platform import submission as sub
from rsi.platform.c_safety import group_guard, interpolate_checkpoint


class Round:
    round_index = 1
    max_output_tokens = 256

    def __init__(self, directory, key, model, x, y, unknown_ids=False):
        self.dataset_key = key
        self.task = dict(sub.TASK_DEFAULTS[key])
        checkpoint = directory / "parent.npz"
        model.save(checkpoint)
        self.parent = SimpleNamespace(
            path=lambda _: checkpoint,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        rows = []
        for i, (features, label) in enumerate(zip(x, y)):
            group = "fd001" if i < len(y)//2 else "fd002"
            identifier = f"{group}_train_u{(i%8)//4+1:04d}_c{i:04d}"
            if key == "B":
                identifier = f"xjtu_Bearing1_4_s{i:06d}"
            rows.append({"id": str(i) if unknown_ids else identifier,
                         "inputs": {"x": float(features[0])},
                         "answer": str(label) if key == "A" else float(label)})
        train_rows = [{**row, "id": "train_"+row["id"]} for row in rows]
        self.data = SimpleNamespace(training_rows=lambda: iter(train_rows),
                                    validation_rows=lambda: iter(rows))
        self.calls = []
        self.submitted = self.finished = None

    def history(self):
        return []

    def state_workspace(self, destination):
        Path(destination).mkdir(parents=True, exist_ok=True)
        return Path(destination)

    def model(self, messages, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(text='{"choice":1,"hypothesis":"candidate","reason":"test"}',
                               call_id="call-1")

    def submit_candidate(self, directory, **kwargs):
        self.submitted = {"directory": Path(directory), **kwargs}
        return {"artifact_id": "selected"}

    def save_state(self, directory):
        self.saved_state = Path(directory)

    def finish(self, **kwargs):
        self.finished = kwargs


@pytest.fixture(autouse=True)
def isolated_pipeline(tmp_path, monkeypatch):
    for key, value in {"CONTEST_WORK": str(tmp_path/"work"), "RSI_V2_CONTROLLER": "pipeline",
                       "RSI_V2_PROBE_RETAIN": "off", "RSI_V2_WARMUP_ROUNDS": "0",
                       "RSI_V2_C_EFFICIENT": "1", "RSI_V2_GATE": "off"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sub, "_apply_drift_policy", lambda ctx, chosen, plans: (chosen, ""))


def regression(left, right, *, spike=False):
    x = np.r_[np.zeros(8), np.ones(8)][:, None]
    if spike:
        x[:] = 0
        x[0] = 1
    y = np.full(16, 100.)
    model = ExtendedModel("regression", ['[["key","x"]]'], [], x, y, hidden=1)
    model.mean[:] = 0
    model.scale[:] = 1
    model.params["w1"][:] = 1
    model.params["b1"][:] = 0
    model.params["w2"][:] = (right-left)/np.tanh(1.)
    model.params["b2"][:] = left
    model.validate()
    return model, x, y


def candidate_plan(ctx, model, evidence_pred=None):
    values, primary, predicted = sub._plans.evaluate(ctx, model, {"type": "none"})
    if evidence_pred is not None:
        predicted = evidence_pred
        values = from_predictions(ctx.task, predicted, ctx.y_val, ctx.classes, profile=ctx.task_key)
        primary = values["rmse"]
    return {"kind": "continue_train_test", "label": "test candidate", "model": model,
            "postprocess": {"type": "none"}, "checkpoint_changed": True,
            "values": values, "primary": primary, "predicted": predicted,
            "group_guard": {"passed": True}}


def install_candidates(monkeypatch, model, evidence_pred=None):
    def build(key, parent, train, validation, **kwargs):
        ctx = kwargs["ctx"]
        return [sub._plans.retain_plan(ctx), candidate_plan(ctx, model, evidence_pred)]
    monkeypatch.setattr(sub, "_candidate_plans", build)


def assert_retained(rnd):
    assert rnd.finished["configuration"]["chosen"] == "retain"
    assert len(rnd.calls) == 1
    assert (rnd.submitted["directory"] / "checkpoint.npz").read_bytes() == rnd.parent.path("checkpoint.npz").read_bytes()


def test_c_llm_choice_cannot_bypass_actual_fd_regression(tmp_path, monkeypatch):
    parent, x, y = regression(1., 20.)
    changed, _, _ = regression(2., 10.)
    install_candidates(monkeypatch, changed)
    rnd = Round(tmp_path, "C", parent, x, y)
    sub.train(rnd)
    assert_retained(rnd)
    assert rnd.finished["configuration"]["c_metric_gate"]["passed"] is False


def test_c_final_guard_rechecks_model_instead_of_trusting_stale_predictions(tmp_path, monkeypatch):
    parent, x, y = regression(1., 20.)
    changed, _, _ = regression(2., 10.)
    install_candidates(monkeypatch, changed, evidence_pred=y+.5)
    rnd = Round(tmp_path, "C", parent, x, y)
    sub.train(rnd)
    assert_retained(rnd)


def test_c_unknown_ids_can_promote_with_explicit_rul_fallback(tmp_path, monkeypatch):
    parent, x, y = regression(20., 20.)
    changed, _, _ = regression(10., 10.)
    install_candidates(monkeypatch, changed)
    rnd = Round(tmp_path, "C", parent, x, y, unknown_ids=True)
    sub.train(rnd)
    configuration = rnd.finished["configuration"]
    assert configuration["chosen"] == "continue_train_test"
    assert configuration["c_metric_gate"]["group_guard"]["protection"] == "rul_fallback"
    assert configuration["c_metric_gate"]["group_guard"]["coverage"]["fd_complete"] is False


def test_c_selected_checkpoint_preserves_explicit_adam_restart(tmp_path, monkeypatch):
    parent, x, y = regression(20., 20.)
    trained, _, _ = regression(0., 0.)
    trained.meta.update(epochs=10, steps=12)
    for key in trained.params:
        trained.first[key][:] = 1
        trained.second[key][:] = 1
    selected = interpolate_checkpoint(parent, trained, .25)
    install_candidates(monkeypatch, selected)
    rnd = Round(tmp_path, "C", parent, x, y)
    sub.train(rnd)
    saved = ExtendedModel.load(rnd.submitted["directory"] / "checkpoint.npz")
    assert saved.meta["epochs"] == 10 and saved.meta["steps"] == 0
    assert saved.meta["c_conservative"]["optimizer_policy"] == "restart_adam_preserve_epoch_history"
    for key in saved.params:
        np.testing.assert_array_equal(saved.params[key], selected.params[key])
        assert not saved.first[key].any() and not saved.second[key].any()


def test_a_bad_drift_policy_override_is_vetoed_at_final_guard(tmp_path, monkeypatch):
    x = np.r_[np.full(20, -1.), np.ones(20)][:, None]
    labels = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    model = ExtendedModel("classification", ['[["key","x"]]'], ["a", "b"], x, labels, hidden=1)
    model.params["w1"][:] = 1
    model.params["b1"][:] = 0
    model.params["w2"][:] = [-2., 2.]
    model.params["b2"][:] = 0
    changed = model.clone()
    changed.params["w2"] *= -1
    monkeypatch.setattr(sub, "_candidate_plans", lambda *args, **kwargs: [sub._plans.retain_plan(kwargs["ctx"])])
    monkeypatch.setattr(sub, "_apply_drift_policy",
                        lambda ctx, chosen, plans: (candidate_plan(ctx, changed), "high_drift_override"))
    rnd = Round(tmp_path, "A", model, x, np.where(labels == 0, "a", "b"))
    sub.train(rnd)
    assert_retained(rnd)
    assert rnd.finished["configuration"]["a_final_guard"]["passed"] is False


def test_b_final_guard_rechecks_mae_after_policy_override(tmp_path, monkeypatch):
    parent, x, y = regression(0., 100., spike=True)
    changed, _, _ = regression(10., 10., spike=True)
    monkeypatch.setattr(sub, "_candidate_plans", lambda *args, **kwargs: [sub._plans.retain_plan(kwargs["ctx"])])
    monkeypatch.setattr(sub, "_apply_drift_policy",
                        lambda ctx, chosen, plans: (candidate_plan(ctx, changed, evidence_pred=y+1), "policy_override"))
    rnd = Round(tmp_path, "B", parent, x, y)
    sub.train(rnd)
    assert_retained(rnd)
    assert rnd.finished["configuration"]["b_final_guard"]["passed"] is False


def test_changed_checkpoint_is_saved_even_for_non_continuation_kind(tmp_path, monkeypatch):
    parent, x, y = regression(20., 20.)
    changed, _, _ = regression(10., 10.)
    def build(*args, **kwargs):
        ctx = kwargs["ctx"]
        plan = candidate_plan(ctx, changed)
        plan["kind"] = "custom_changed_model"
        assert "checkpoint" not in plan
        return [sub._plans.retain_plan(ctx), plan]
    monkeypatch.setattr(sub, "_candidate_plans", build)
    rnd = Round(tmp_path, "B", parent, x, y)
    sub.train(rnd)
    assert rnd.finished["configuration"]["chosen"] == "custom_changed_model"
    saved = ExtendedModel.load(rnd.submitted["directory"] / "checkpoint.npz")
    np.testing.assert_array_equal(saved.predict(x), changed.predict(x))
    assert not np.array_equal(saved.predict(x), parent.predict(x))


def test_c_kernel_selection_also_reaches_final_group_guard(tmp_path, monkeypatch):
    parent, x, y = regression(1., 20.)
    changed, _, _ = regression(2., 10.)
    monkeypatch.setenv("RSI_V2_C_EFFICIENT", "0")
    monkeypatch.setenv("RSI_V2_CONTROLLER", "kernel")
    state_dir = tmp_path / "kernel_state"
    state_dir.mkdir()
    monkeypatch.setattr(sub, "_open_agent_state", lambda *args: SimpleNamespace(dir=state_dir))
    monkeypatch.setattr(sub, "_kernel_brief", lambda *args: {})
    def run_kernel(rnd, *args, **kwargs):
        receipt = rnd.model([{"role": "user", "content": "kernel diagnostic"}])
        return SimpleNamespace(calls=1, call_ids=[receipt.call_id],
                               decision={"decision": "promote"}, host=SimpleNamespace(results=[]),
                               error=None, usage={}, trace=[], to_dict=lambda: {})
    monkeypatch.setattr(sub._agent_host, "run_kernel_round", run_kernel)
    monkeypatch.setattr(sub, "_choose_kernel_plan",
                        lambda result, ctx, direction: candidate_plan(ctx, changed, evidence_pred=y+.5))
    monkeypatch.setattr(sub, "_safety_net_plans", lambda ctx: [sub._plans.retain_plan(ctx)])
    rnd = Round(tmp_path, "C", parent, x, y)
    sub.train(rnd)
    assert_retained(rnd)
    assert rnd.finished["configuration"]["controller"] == "kernel"
    assert rnd.finished["configuration"]["c_metric_gate"]["passed"] is False


def test_b_global_improvement_cannot_hide_small_bearing_regression_in_train(tmp_path, monkeypatch):
    parent, _, _ = regression(1., 20.)
    changed, _, _ = regression(2., 10.)
    x = np.r_[np.zeros(10), np.ones(90)][:, None]
    y = np.full(100, 100.)
    install_candidates(monkeypatch, changed)
    rnd = Round(tmp_path, "B", parent, x, y)
    validation = list(rnd.data.validation_rows())
    for index, row in enumerate(validation):
        group = "Bearing1_4" if index < 10 else "Bearing3_4"
        row["id"] = f"xjtu_{group}_s{index:06d}"
    training = [{**row, "id": "train_"+row["id"]} for row in validation]
    rnd.data = SimpleNamespace(training_rows=lambda: iter(training),
                               validation_rows=lambda: iter(validation))
    sub.train(rnd)
    assert_retained(rnd)
    audit = rnd.finished["configuration"]["b_final_guard"]["group_guard"]
    assert audit["global"]["passed"]
    assert audit["groups"]["Bearing1_4"]["unique_snapshots"] == 10
    assert audit["groups"]["Bearing3_4"]["unique_snapshots"] == 90
    assert audit["groups"]["Bearing1_4"]["failed_metrics"] == ["rmse", "mae"]
    assert audit["coverage"]["complete"] and not audit["passed"]


def test_a_final_remeasurement_synchronizes_selected_metrics_and_journal(tmp_path, monkeypatch):
    x = np.r_[np.full(20, -1.), np.ones(20)][:, None]
    labels = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    parent = ExtendedModel("classification", ['[["key","x"]]'], ["a", "b"], x, labels, hidden=1)
    parent.params["w1"][:] = 1
    parent.params["b1"][:] = 0
    parent.params["w2"][:] = [-2., 2.]
    parent.params["b2"][:] = 0
    changed = parent.clone()
    changed.params["w2"] *= 2
    chosen_plans = []
    monkeypatch.setattr(sub, "_candidate_plans", lambda *args, **kwargs: [sub._plans.retain_plan(kwargs["ctx"])])
    def policy(ctx, chosen, plans):
        plan = candidate_plan(ctx, changed)
        plan.update(values={"macro_f1": .123}, primary=.123, predicted=np.zeros((40, 2)))
        chosen_plans.append(plan)
        return plan, "recent_holdout_candidate"
    monkeypatch.setattr(sub, "_apply_drift_policy", policy)
    rnd = Round(tmp_path, "A", parent, x, np.where(labels == 0, "a", "b"))
    sub.train(rnd)
    selected = chosen_plans[0]
    assert rnd.finished["configuration"]["a_final_guard"]["passed"]
    assert selected["values"]["macro_f1"] == 1.
    assert selected["primary"] == 1.
    np.testing.assert_array_equal(selected["predicted"], changed.predict(x))
    journal = json.loads((rnd.saved_state/"round_records.jsonl").read_text().splitlines()[-1])
    assert journal["selected_metrics"]["macro_f1"] == 1.


def test_rejected_c_candidate_keeps_cache_alpha_and_group_audit_in_configuration(tmp_path, monkeypatch):
    parent, x, y = regression(1., 20.)
    changed, _, _ = regression(2., 10.)
    cache_note = {"cache_hit": True, "key": "deterministic-test-key", "saved_fit_cpu_seconds": 3.1}
    def build(key, model, train, validation, **kwargs):
        ctx = kwargs["ctx"]
        plan = candidate_plan(ctx, changed)
        plan.update(alpha=.25, experiment_cache=cache_note,
                    group_guard=group_guard(ctx.parent_predicted, plan["predicted"], ctx.y_val,
                                            [row["id"] for row in validation], enforce_rul=True))
        assert not plan["group_guard"]["passed"]
        return [sub._plans.retain_plan(ctx), plan]
    monkeypatch.setattr(sub, "_candidate_plans", build)
    rnd = Round(tmp_path, "C", parent, x, y)
    sub.train(rnd)
    assert_retained(rnd)
    candidate = next(item for item in rnd.finished["configuration"]["candidates"]
                     if item["kind"] == "continue_train_test")
    assert candidate["experiment_cache"] == cache_note
    assert candidate["alpha"] == .25
    assert candidate["training_action"] == "cache_reused"
    assert not candidate["group_guard"]["passed"]
    assert candidate["group_guard"]["groups"]["fd001"]["failed_metrics"] == ["rmse", "mae", "nasa_score"]
