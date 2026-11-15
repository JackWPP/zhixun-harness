"""Non-blocking soft-goal semantics (user m03315).

The goal inside the delivered harness is advisory only: missing a target must
never block a run from producing valid predictions / records.  These tests pin
that behaviour so a future refactor cannot turn the soft target into a hard
gate.
"""

from __future__ import annotations

from types import SimpleNamespace

from rsi.agent.goals import GoalLoop
from rsi.runner import Runner


class _FakeDB:
    def __init__(self) -> None:
        self._goals: dict[str, dict] = {}

    def save_goal(self, session: str, state: dict) -> None:
        self._goals[session] = dict(state)

    def load_goal(self, session: str):
        return self._goals.get(session)


def _runner():
    return object.__new__(Runner)


def test_goal_soft_missed_is_advisory():
    r = _runner()
    state = SimpleNamespace(
        config=SimpleNamespace(key="A", direction="higher"),
        selection_metrics={"primary": 0.70},
        degraded=False,
    )
    soft = r._goal_soft(state)
    assert soft["target"] == 0.7346
    assert soft["verdict"] == "missed"
    assert soft["note"] == "soft target, non-blocking"
    # the gap is negative but nothing raises / blocks
    assert soft["gap"] < 0


def test_goal_soft_met_and_regression_direction():
    r = _runner()
    hit = r._goal_soft(SimpleNamespace(
        config=SimpleNamespace(key="A", direction="higher"),
        selection_metrics={"primary": 0.75}, degraded=False))
    assert hit["verdict"] == "met" and hit["gap"] > 0

    b_hit = r._goal_soft(SimpleNamespace(
        config=SimpleNamespace(key="B", direction="lower"),
        selection_metrics={"primary": 400.0}, degraded=False))
    assert b_hit["verdict"] == "met" and b_hit["gap"] > 0

    unknown = r._goal_soft(SimpleNamespace(
        config=SimpleNamespace(key="D", direction="higher"),
        selection_metrics={"primary": 0.9}, degraded=False))
    assert unknown["verdict"] == "unknown"


def test_goal_loop_does_not_finish_on_missed_soft_target():
    loop = GoalLoop(task_key="A", run_id="run-x", db=_FakeDB(), rounds_target=10)
    done, reason = loop.judge(
        decision="retain",
        round_record={"index": 0, "decision": "retain", "metrics": {"primary": 0.10}},
        budget_remaining=1_000_000,
        wall_remaining=10_000,
        degraded=False,
    )
    assert done is False
    assert "round" in reason.lower() or "continue" in reason.lower()


def test_goal_loop_still_honours_hard_budget():
    loop = GoalLoop(task_key="A", run_id="run-y", db=_FakeDB(), rounds_target=10)
    done, _ = loop.judge(decision="retain", round_record={"index": 1},
                         budget_remaining=0, wall_remaining=10, degraded=False)
    assert done is True
