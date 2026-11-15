"""Offline tests for the autonomous kernel layer (no network, no LLM)."""
from __future__ import annotations

import asyncio

import pytest

from rsi.agent.actions import KernelEnv, build_kernel_tools
from rsi.agent.goals import GoalLoop
from rsi.agent.guard import Guard, GuardViolation
from rsi.agent.kernel import Kernel
from rsi.agent.runtime import RoleResult
from rsi.executor import ExperimentProposal
from rsi.tasks import resolve_tasks


class FakeHost:
    def __init__(self, parent: float = 0.714):
        self.parent = parent
        self.experiments = 0
        self.panel = []
        self.parent_sha = "sha256:" + "a" * 64

    def context(self) -> dict:
        return {"task": {"key": "A"}, "drift": {"severity": "moderate"},
                "history": [], "metrics": {"primary": self.parent}, "budget": {}}

    def parent_primary(self) -> float:
        return self.parent

    def run_experiment(self, proposal: ExperimentProposal) -> dict:
        self.experiments += 1
        eid = f"exp-{self.experiments:04d}"
        primary = 0.75 if proposal.postprocess.get("type") == "recenter" else self.parent
        return {"experiment_id": eid, "ok": True,
                "proposal": {"strategy": proposal.strategy, "params": proposal.params,
                             "postprocess": proposal.postprocess},
                "postprocess": proposal.postprocess,
                "parent_checkpoint": "/tmp/parent.npz",
                "candidate_checkpoint": f"/tmp/{eid}.npz",
                "checkpoint_sha256": "sha256:" + "b" * 64,
                "code_sha256": "sha256:" + "c" * 64,
                "metrics": {"selection": {"primary": primary, "macro_f1": primary},
                            "report": {"primary": primary}},
                "cpu_seconds": 0.1, "wall_seconds": 0.2, "error": None,
                "notes": {}, "log_path": None}

    def record_panel(self, entries: list[dict]) -> None:
        self.panel.append(entries)


class DummyRuntime:
    backend = "dummy"

    def __init__(self):
        from rsi.agent.config import TokenBudget

        self.budget = TokenBudget(cap=1_000_000)

    def available(self) -> bool:
        return False


def _env(host: FakeHost, task_key: str = "A") -> KernelEnv:
    config = resolve_tasks([task_key])[0]
    return KernelEnv(task=config, host=host, guard=Guard(config),
                     runtime=DummyRuntime(), brief={})


def _tool(env: KernelEnv, name: str):
    tools = build_kernel_tools(env)
    return next(tool for tool in tools if tool.name == name)


def _call(env: KernelEnv, name: str, **params):
    tool = _tool(env, name)
    model = tool.parameters(**params)
    return asyncio.run(tool.execute("call-1", model))


# -- Guard -----------------------------------------------------------------
def test_guard_rejects_third_party_training_data():
    config = resolve_tasks(["A"])[0]
    guard = Guard(config)
    with pytest.raises(GuardViolation):
        guard.validate_catalog(ExperimentProposal("continue_train", {"data": "train+all-public"}))


def test_guard_enforces_caps_and_gate():
    config = resolve_tasks(["A"])[0]
    guard = Guard(config, max_experiments_per_round=2, max_experiments_per_task=3)
    guard.validate_experiment(ExperimentProposal("retain", {}), 1, 1)
    with pytest.raises(GuardViolation):
        guard.validate_experiment(ExperimentProposal("retain", {}), 2, 2)
    weak = {"exp-1": {"ok": True, "metrics": {"selection": {"primary": config.baseline_primary}}}}
    with pytest.raises(GuardViolation):
        guard.check_decision("promote", "exp-1", weak, config.baseline_primary, 1)
    strong = {"exp-1": {"ok": True, "metrics": {"selection": {"primary": config.baseline_primary + 0.05}}}}
    decision = guard.check_decision("promote", "exp-1", strong, config.baseline_primary, 1)
    assert decision.decision == "promote"


# -- kernel tools ----------------------------------------------------------
def test_kernel_tools_execute_a_full_round():
    host = FakeHost()
    env = _env(host)
    # an illegal plan is rejected but does not crash the agent
    rejected = _call(env, "propose_experiment", strategy="continue_train",
                     params_json='{"data": "train+all-public"}',
                     postprocess_json="{}", hypothesis="cheat", expected_effect="x", risk="y")
    assert rejected.details["ok"] is False and env.violations

    proposed = _call(env, "propose_experiment", strategy="retain",
                     params_json="{}",
                     postprocess_json='{"type": "recenter", "mode": "batch_mean", "weight": 0.4}',
                     hypothesis="drift is a mean shift", expected_effect="+0.02", risk="low")
    plan_id = proposed.content[0]["text"]
    assert "plan-01" in plan_id

    executed = _call(env, "run_experiment", plan_id="plan-01")
    assert executed.details["ok"] is True
    assert env.results["exp-0001"]["metrics"]["selection"]["primary"] == 0.75

    decided = _call(env, "decide", decision="promote", experiment_id="exp-0001",
                    reason="recenter beats margin", confidence=0.9)
    assert decided.details["ok"] is True
    assert env.decision["decision"] == "promote"

    finished = _call(env, "finish_round", summary="promoted recenter", lesson="recenter works")
    assert finished.details["ok"] is True


def test_kernel_tool_rejects_promote_below_gate():
    host = FakeHost()
    env = _env(host)
    _call(env, "propose_experiment", strategy="retain", params_json="{}",
          postprocess_json="{}", hypothesis="noop", expected_effect="0", risk="low")
    _ = _call(env, "run_experiment", plan_id="plan-01")  # primary == parent -> retain only
    result = _call(env, "decide", decision="promote", experiment_id="exp-0001",
                   reason="force it", confidence=1.0)
    assert result.details["ok"] is False
    assert env.decision is None


def test_kernel_auto_decision_closes_round():
    host = FakeHost(parent=0.714)
    env = _env(host)
    env.results["exp-0001"] = host.run_experiment(ExperimentProposal(
        "retain", {}, {"type": "recenter", "mode": "batch_mean", "weight": 0.4}))
    kernel = Kernel(resolve_tasks(["A"])[0], DummyRuntime(), Guard(resolve_tasks(["A"])[0]))
    decision = kernel._auto_decision(env, host)
    assert decision is not None and decision["decision"] == "promote"


# -- goal loop -------------------------------------------------------------
def test_goal_loop_persists_and_closes(tmp_path):
    from rsi.memory import StateDB

    db = StateDB(tmp_path / "state.db")
    loop = GoalLoop(task_key="A", run_id="run-1", db=db, rounds_target=2)
    state = loop.start()
    assert state.status == "active"
    done, _ = loop.judge(decision="retain")
    assert not done
    done, reason = loop.judge(decision="promote")
    assert done and "2 rounds" in reason
    loaded = db.load_goal("A:run-1")
    assert loaded["rounds_done"] == 2 and loaded["status"] == "done"
    db.close()


def test_goal_loop_fail_open_on_budget():
    loop = GoalLoop(task_key="B", rounds_target=10)
    loop.start()
    done, reason = loop.judge(budget_remaining=0)
    assert done and "budget" in reason


def test_role_result_shape():
    result = RoleResult(role="kernel", ok=True, text="ok")
    assert result.to_dict()["role"] == "kernel"


def test_planner_ranks_by_win_rate():
    from rsi.planner import fallback_proposals, rank_proposals

    plans = fallback_proposals("B", 3)
    assert len(plans) == 2
    stats = [{"strategy": plans[1].strategy,
              "postprocess_type": (plans[1].postprocess or {}).get("type", "none"),
              "win_rate": 0.9}]
    ranked = rank_proposals(plans, stats)
    assert ranked[0].strategy == plans[1].strategy
    # no stats -> order untouched
    assert rank_proposals(plans, None) == plans
    # the round's seeded plan is never dropped, even if untried
    assert plans[0] in ranked
