"""Tests for the host-gateway agent route (kernel controller on workspace v2)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
TASK_A = REPO / "datasets" / "cpu-a-guide-20261110-v1"


def read_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def load_task_a(limit_train: int = 2000, limit_val: int = 200):
    from rsi.model import ExtendedModel

    model = ExtendedModel.from_checkpoint(TASK_A / "checkpoint.npz")
    model.meta["_source_path"] = str(TASK_A / "checkpoint.npz")
    train = read_rows(TASK_A / "public" / "train.jsonl", limit_train)
    validation = read_rows(TASK_A / "public" / "validation.jsonl", limit_val)
    return model, train, validation


def plan_context():
    from rsi.platform.plans import PlanContext

    model, train, validation = load_task_a()
    return PlanContext.from_rows("A", model, train, validation, metric="macro_f1",
                                 direction="higher", margin=0.004,
                                 classes=list(model.meta.get("classes") or []))


def test_plans_retain_matches_parent():
    from rsi.platform.plans import retain_plan

    ctx = plan_context()
    plan = retain_plan(ctx)
    assert plan["ok"] is True
    assert plan["kind"] == "retain"
    assert abs(plan["primary"] - ctx.parent_primary) < 1e-12


def test_plans_recenter_and_pair_fit_without_touching_validation():
    from rsi.executor import ExperimentProposal
    from rsi.platform.plans import plan_for_proposal

    ctx = plan_context()
    recenter = plan_for_proposal(ctx, ExperimentProposal(
        strategy="retain", params={},
        postprocess={"type": "recenter", "mode": "batch_mean", "weight": 0.4},
        hypothesis="recenter helps drift"))
    assert recenter["ok"] and recenter["postprocess"]["type"] == "recenter"

    # The official A train projection has too few Acetaldehyde rows for the
    # pair fit to clear its own improvement bar, so an unfitted pair spec must
    # degrade to "none" rather than error.  An explicit (history-fitted) delta
    # must be preserved.
    pair = plan_for_proposal(ctx, ExperimentProposal(
        strategy="retain", params={},
        postprocess={"type": "pair_threshold", "pair": [0, 3]},
        hypothesis="delayed labels"))
    assert pair["ok"]
    assert pair["postprocess"]["type"] in ("pair_threshold", "none")
    explicit = plan_for_proposal(ctx, ExperimentProposal(
        strategy="retain", params={},
        postprocess={"type": "pair_threshold", "pair": [0, 3], "delta": -4.0},
        hypothesis="history-fitted delta"))
    assert explicit["postprocess"]["type"] == "pair_threshold"
    assert explicit["postprocess"]["delta"] == -4.0


def test_plans_continue_train_keeps_checkpoint_compatible():
    from rsi.executor import ExperimentProposal
    from rsi.platform.plans import plan_for_proposal

    ctx = plan_context()
    plan = plan_for_proposal(ctx, ExperimentProposal(
        strategy="continue_train", params={"learning_rate": 1e-3, "epochs": 1},
        postprocess={"type": "none"}, hypothesis="short continuation"))
    assert plan["ok"], plan.get("error")
    assert plan["checkpoint_changed"] is True
    assert plan["model"].meta["task"] == "classification"


class FakeMessage(SimpleNamespace):
    pass


class FakeReceipt:
    def __init__(self, content: str, call_id: str):
        message = FakeMessage(content=content, tool_calls=None)
        self.choice = SimpleNamespace(message=message)
        self.response = SimpleNamespace(choices=[self.choice], usage={
            "prompt_tokens": 100, "completion_tokens": 20})
        self.call_id = call_id
        self.text = content


class ScriptedRnd:
    """Minimal workspace-v2 round double that replays scripted agent actions."""

    def __init__(self, actions: list[dict]):
        self.actions = list(actions)
        self.calls = 0
        self.round_index = 1
        self.dataset_key = "A"
        self.task = {"task": "classification", "metric": "macro_f1", "direction": "higher"}
        self.max_output_tokens = 256

    def model(self, messages, **options):
        action = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return FakeReceipt(json.dumps(action, ensure_ascii=False), "receipt-%d" % self.calls)


def test_kernel_round_text_protocol_end_to_end():
    from rsi.platform.agent_host import AgentState, run_kernel_round
    from rsi.platform.plans import PlanContext
    from rsi.model import ExtendedModel

    model = ExtendedModel.from_checkpoint(TASK_A / "checkpoint.npz")
    train = read_rows(TASK_A / "public" / "train.jsonl", 300)
    validation = read_rows(TASK_A / "public" / "validation.jsonl", 200)
    ctx = PlanContext.from_rows("A", model, train, validation, metric="macro_f1",
                                direction="higher", margin=0.004,
                                classes=list(model.meta.get("classes") or []))
    brief = {"task": {"key": "A"}, "round_index": 1, "rounds_total": 10,
             "history": [], "catalog": ""}
    actions = [
        {"tool": "get_run_state", "args": {}},
        {"tool": "propose_experiment", "args": {
            "strategy": "retain", "params_json": "{}",
            "postprocess_json": "{\"type\": \"recenter\", \"mode\": \"batch_mean\", \"weight\": 0.4}",
            "hypothesis": "recenter should help under drift"}},
        {"tool": "run_experiment", "args": {"plan_id": "plan-01"}},
        {"tool": "decide", "args": {"decision": "retain", "reason": "margin not met"}},
        {"tool": "finish_round", "args": {"summary": "recenter evaluated", "lesson": "keep margin"}},
    ]
    rnd = ScriptedRnd(actions)
    result = run_kernel_round(rnd, ctx, brief, AgentState(), run_id="A-r1",
                              round_index=1, experiments_total=0, max_turns=6)
    assert result.calls == 5
    assert len(result.call_ids) == 5
    assert result.decision is not None
    assert result.decision["decision"] == "retain"
    assert result.host is not None and len(result.host.results) == 1
    assert result.host.results[0]["ok"] is True


def test_gateway_runtime_counts_retries_and_usage():
    from rsi.platform.agent_host import GatewayRuntime

    class FlakyRnd:
        def __init__(self):
            self.calls = 0

        def model(self, messages, **options):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return FakeReceipt("{}", "call-ok")

    runtime = GatewayRuntime(FlakyRnd(), max_turns=2)
    receipt = runtime.model([{"role": "user", "content": "hi"}], max_tokens=8)
    assert receipt.call_id == "call-ok"
    assert runtime.call_ids == ["call-ok"]
    assert runtime.budget.used_input == 100
    assert runtime.budget.used_output == 20


def plan_context_full():
    from rsi.platform.plans import PlanContext

    model, train, _ = load_task_a(limit_train=100000, limit_val=0)
    validation = read_rows(TASK_A / "public" / "validation.jsonl", 100000)
    return PlanContext.from_rows("A", model, train, validation, metric="macro_f1",
                                 direction="higher", margin=0.004,
                                 classes=list(model.meta.get("classes") or []))


def test_drift_profile_levels_and_pair_history_arm():
    from rsi.platform.plans import PAIR_HISTORY_DELTA, drift_profile, pair_history_plan

    ctx = plan_context_full()
    profile = drift_profile(ctx)
    assert profile["level"] == "high"
    arm = pair_history_plan(ctx)
    assert arm["ok"] is True
    assert arm["kind"] == "pair_history"
    assert abs(float(arm["postprocess"]["delta"]) - PAIR_HISTORY_DELTA) < 1e-12
    # The historical delta is a prior: it is *not* re-fitted on this validation.
    assert arm["postprocess"]["type"] == "pair_threshold"


def test_apply_drift_policy_prefers_history_arm_on_high_drift():
    from rsi.platform import submission
    from rsi.platform.plans import retain_plan

    ctx = plan_context_full()
    chosen, note = submission._apply_drift_policy(ctx, retain_plan(ctx))
    assert chosen["kind"] == "pair_history"
    assert note.startswith("drift_policy:")
    # A recenter plan is also overridden while the historical arm passes the
    # "not materially worse than parent" guard.
    from rsi.executor import ExperimentProposal
    from rsi.platform.plans import plan_for_proposal

    recenter = plan_for_proposal(ctx, ExperimentProposal(
        strategy="retain", params={},
        postprocess={"type": "recenter", "mode": "batch_std", "weight": 0.2}))
    chosen2, note2 = submission._apply_drift_policy(ctx, recenter)
    assert chosen2["kind"] == "pair_history"
    assert note2.startswith("drift_policy:")


def test_drift_policy_env_switch_disables_override(monkeypatch):
    from rsi.platform import submission
    from rsi.platform.plans import retain_plan

    ctx = plan_context_full()
    parent = retain_plan(ctx)
    monkeypatch.setenv("RSI_V2_DRIFT_POLICY", "0")
    chosen, note = submission._apply_drift_policy(ctx, parent)
    assert chosen["kind"] == "retain"
    assert note == ""
    monkeypatch.setenv("RSI_V2_DRIFT_POLICY", "1")
    chosen2, note2 = submission._apply_drift_policy(ctx, parent)
    assert chosen2["kind"] == "pair_history"
    assert note2.startswith("drift_policy:")


def test_drift_policy_env_switch_reads_pair_delta(monkeypatch):
    import importlib

    from rsi.platform import plans

    monkeypatch.setenv("RSI_V2_PAIR_DELTA", "-2.5")
    reloaded = importlib.reload(plans)
    try:
        assert reloaded.PAIR_HISTORY_DELTA == pytest.approx(-2.5)
    finally:
        monkeypatch.delenv("RSI_V2_PAIR_DELTA", raising=False)
        importlib.reload(plans)


def test_gateway_runtime_max_output_reads_env(monkeypatch):
    from rsi.platform.agent_host import GatewayRuntime

    captured = []

    class Rnd:
        round_index = 1

        def model(self, messages, **options):
            captured.append(int(options.get("max_tokens") or 0))
            return FakeReceipt("{\"tool\": \"finish_round\", \"args\": {}}", "call-max-output")

    monkeypatch.setenv("RSI_V2_MAX_OUTPUT", "1536")
    runtime = GatewayRuntime(Rnd(), max_turns=2, token_cap=0)
    runtime.model([{"role": "user", "content": "state"}])
    assert captured == [1536]
