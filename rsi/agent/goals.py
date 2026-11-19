"""Goal loop for the autonomous kernel.

The supervisor still owns the hard limits (wall clock, experiments, token
budget); inside those limits a *goal* gives the PI agent a persistent objective
and a continuation signal, so a single round is not a fixed six-step script but
a loop the agent can reason about:

    observe -> hypothesis -> experiment -> evaluate -> decide -> remember

The judge is deterministic by default (no extra LLM call, keeps prompt-cache and
token budget intact), fail-open on any error, and the continuation prompt is a
plain user message so the system prompt (and its prefix cache) never changes.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field

from rsi.records import utcnow_iso

MAX_GOAL_TURNS = 20
MAX_PARSE_FAILURES = 3


@dataclass
class GoalState:
    goal: str
    task_key: str
    run_id: str = ""
    goal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "active"          # active | paused | done | cleared
    turns_used: int = 0
    max_turns: int = MAX_GOAL_TURNS
    rounds_target: int = 10
    rounds_done: int = 0
    consecutive_parse_failures: int = 0
    subgoals: list[str] = field(default_factory=list)
    last_verdict: str = ""
    last_reason: str = ""
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GoalState":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


def default_subgoals(rounds_target: int) -> list[str]:
    return [
        f"在 {rounds_target} 轮内完成全部轮次，且每一轮都有可核验的记录（实验、指标、决策）。",
        "每一轮至少执行 1 个有效实验，或给出明确的 retain/rollback 理由并写明证据。",
        "只使用 train 拟合；selection 仅用于公开候选选择，不读取隐藏测试标签。",
        "在预算内完成；若预算不足，优先保留当前最优模型并写清降级决策。",
    ]


class GoalLoop:
    """Thin persistence + judge wrapper around :class:`GoalState`."""

    def __init__(self, *, task_key: str, run_id: str = "", db=None,
                 rounds_target: int = 10, max_turns: int = MAX_GOAL_TURNS):
        self.task_key = task_key
        self.run_id = run_id
        self.db = db
        self.rounds_target = int(rounds_target)
        self.max_turns = int(max_turns)
        self.state: GoalState | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self, goal: str | None = None) -> GoalState:
        goal = goal or (
            f"在不触碰隐藏标签、只用官方允许数据的前提下，最大化任务 {self.task_key} "
            f"的公开验证主指标，并留下完整可复现的十轮实验记录。")
        self.state = GoalState(goal=goal, task_key=self.task_key, run_id=self.run_id,
                               rounds_target=self.rounds_target,
                               max_turns=self.max_turns,
                               subgoals=default_subgoals(self.rounds_target))
        self._save()
        return self.state

    @property
    def session_key(self) -> str:
        return f"{self.task_key}:{self.run_id}"

    def load(self) -> GoalState | None:
        if self.db is not None:
            try:
                data = self.db.load_goal(self.session_key)
                if data:
                    self.state = GoalState.from_dict(data)
                    return self.state
            except Exception:
                pass
        return self.state

    def reset(self) -> GoalState:
        return self.start()

    # -- judge -------------------------------------------------------------
    def judge(self, *, decision: str | None = None, round_record: dict | None = None,
              budget_remaining: int | None = None, wall_remaining: float | None = None,
              degraded: bool = False) -> tuple[bool, str]:
        """Return ``(done, reason)``. Deterministic and fail-open."""
        state = self.state or self.start()
        state.rounds_done += 1
        if decision:
            state.last_verdict = decision
        if round_record:
            state.last_reason = str(round_record.get("decision_reason") or "")[:400]
        if state.rounds_done >= state.rounds_target:
            done, reason = True, f"all {state.rounds_target} rounds completed"
        elif degraded and state.rounds_done >= max(1, state.rounds_target - 2):
            done, reason = True, "time budget degraded, closing early with complete rounds"
        elif budget_remaining is not None and budget_remaining <= 0:
            done, reason = True, "token budget exhausted"
        elif wall_remaining is not None and wall_remaining <= 0:
            done, reason = True, "wall clock budget exhausted"
        else:
            done, reason = False, "continue"
        self.record_verdict(done, reason)
        return done, reason

    def record_verdict(self, done: bool, reason: str) -> None:
        if self.state is None:
            return
        self.state.last_verdict = "done" if done else "continue"
        self.state.last_reason = str(reason)[:400]
        self.state.turns_used += 1
        if done:
            self.state.status = "done"
        elif self.state.turns_used >= self.state.max_turns:
            self.state.status = "paused"
        self._save()

    def note_parse_failure(self, detail: str = "") -> None:
        if self.state is None:
            return
        self.state.consecutive_parse_failures += 1
        if self.state.consecutive_parse_failures >= MAX_PARSE_FAILURES:
            self.pause(f"judge/agent output unparsable {MAX_PARSE_FAILURES} times: {detail}"[:300])
        self._save()

    def note_parse_success(self) -> None:
        if self.state is not None:
            self.state.consecutive_parse_failures = 0

    def pause(self, reason: str) -> None:
        if self.state is None:
            return
        self.state.status = "paused"
        self.state.last_reason = reason[:400]
        self._save()

    def should_continue(self) -> bool:
        return bool(self.state and self.state.status == "active"
                    and self.state.turns_used < self.state.max_turns)

    def completion_message(self) -> str:
        """Plain user message used for the next loop turn (keeps system prompt frozen)."""
        state = self.state or self.start()
        return (
            f"目标循环第 {state.turns_used} 轮结束。目标：{state.goal}\n"
            f"进度：rounds_done={state.rounds_done}/{state.rounds_target}，"
            f"turns_used={state.turns_used}/{state.max_turns}，status={state.status}。\n"
            f"子目标：{json.dumps(state.subgoals, ensure_ascii=False)}\n"
            "若仍有剩余轮次和预算，请继续下一轮：读取最新状态与记忆，提出假设，执行实验，"
            "评估候选，调用 decide 做决策，并在有人类可读结论文本时调用 finish_round。"
        )

    # -- persistence -------------------------------------------------------
    def _save(self) -> None:
        if self.db is None or self.state is None:
            return
        self.state.updated_at = utcnow_iso()
        try:
            self.db.save_goal(self.session_key, self.state.to_dict())
        except Exception:
            pass
