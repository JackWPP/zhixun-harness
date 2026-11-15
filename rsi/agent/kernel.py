"""Autonomous kernel: one PI agent loop per round.

The kernel is the "brain" of the harness. It holds a persistent goal, a frozen
memory snapshot and a tool belt with real actions (run experiments, decide,
delegate, write memory). The supervisor ({mod}`rsi.runner`) only enforces budgets
and records the outcome; when the LLM is unavailable or the kernel fails, the
deterministic pipeline controller takes over so the harness always completes the
ten rounds required by the rules.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from rsi.records import utcnow_iso
from rsi.selection import best_candidate, gate
from rsi.tasks import TaskConfig

from .actions import KernelEnv, build_kernel_tools
from .guard import Guard
from .prompts_kernel import KERNEL_SYSTEM, kernel_round_prompt
from .runtime import PIRuntime


@dataclass
class RoundOutcome:
    ok: bool
    fallback: bool = False
    error: str | None = None
    proposals: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    decision: dict | None = None
    finished: dict | None = None
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency: float = 0.0
    tool_calls: int = 0
    env: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {"ok": self.ok, "fallback": self.fallback, "error": self.error,
                "decision": self.decision, "finished": self.finished,
                "usage": self.usage, "latency": round(self.latency, 2),
                "tool_calls": self.tool_calls, "trace": self.trace[-30:],
                "env": self.env}


class Kernel:
    def __init__(self, task: TaskConfig, runtime: PIRuntime, guard: Guard, *,
                 memory=None, skills=None, recall=None, recorder=None,
                 run_id: str = "", max_turns: int = 20,
                 round_timeout: float = 1800.0, experiment_timeout: float = 900.0,
                 verbose: bool = True):
        self.task = task
        self.runtime = runtime
        self.guard = guard
        self.memory = memory
        self.skills = skills
        self.recall = recall
        self.recorder = recorder
        self.run_id = run_id
        self.max_turns = int(max_turns)
        self.round_timeout = float(round_timeout)
        self.experiment_timeout = float(experiment_timeout)
        self.verbose = verbose
        self.rounds = 0

    # -- public ------------------------------------------------------------
    def available(self) -> bool:
        return self.runtime.available() and not self.runtime.budget.exhausted()

    def run_round(self, host, brief: dict, round_index: int,
                  experiments_total: int = 0) -> RoundOutcome:
        start = time.time()
        if not self.available():
            return RoundOutcome(ok=False, fallback=True,
                                error="LLM runtime unavailable or token budget exhausted",
                                latency=time.time() - start)
        env = KernelEnv(task=self.task, host=host, guard=self.guard,
                        runtime=self.runtime, memory=self.memory,
                        skills=self.skills, recall=self.recall, brief=brief)
        env.experiments_total = int(experiments_total)
        env.max_experiments_round = self.guard.max_experiments_per_round
        tools = build_kernel_tools(env)
        user_prompt = kernel_round_prompt(brief)

        async def _run_agent():
            return await self.runtime.arun_agent(
                KERNEL_SYSTEM, user_prompt, tools,
                max_turns=self.max_turns,
                tool_timeout=max(self.experiment_timeout + 240.0, 600.0),
                timeout_seconds=self.round_timeout)

        error: str | None = None
        result = None
        try:
            agent = self.runtime.run_coro(_run_agent(), timeout=self.round_timeout + 120.0)
            result = self.runtime.collect_agent(agent, role="kernel", start=start)
            error = result.error
        except Exception as exc:  # pragma: no cover - network/runtime failure
            error = f"{exc.__class__.__name__}: {exc}"
            self.runtime.budget.errors += 1

        usage = result.usage if result is not None else {}
        if self.recorder is not None and result is not None:
            try:
                self.recorder.llm_call(self.task.key, "kernel", usage,
                                       ok=result.ok, latency=result.latency,
                                       tool_calls=result.tool_calls, controller="kernel")
            except Exception:
                pass

        outcome = RoundOutcome(
            ok=bool(result and result.ok), error=error,
            proposals=env.plan_list(), results=list(env.results.values()),
            decision=env.decision, finished=env.finished, trace=env.trace,
            usage=usage, latency=time.time() - start,
            tool_calls=env.delegates + env.panels + len(env.trace),
            env=env.summary())

        # Deterministic closure: the kernel must still end the round with a
        # decision even if it forgot to call decide / finish_round.
        if env.decision is None:
            auto = self._auto_decision(env, host)
            if auto is not None:
                outcome.decision = auto
                outcome.ok = True
                outcome.trace.append({"tool": "auto_decision", "ok": True,
                                      "detail": auto.get("reason", ""),
                                      "at": utcnow_iso()})
            else:
                outcome.ok = False
        if outcome.decision is not None and outcome.finished is None:
            outcome.finished = {"summary": self._auto_summary(outcome),
                                "lesson": "", "source": "kernel_autoclose"}
        if outcome.decision is not None:
            outcome.decision.setdefault("source", "kernel")
        self.rounds += 1
        if self.verbose:
            print(f"[kernel] round {round_index}/{brief.get('rounds_total')} "
                  f"ok={outcome.ok} experiments={len(outcome.results)} "
                  f"decision={(outcome.decision or {}).get('decision')} "
                  f"tokens={usage.get('input', 0) + usage.get('output', 0)}", flush=True)
        return outcome

    # -- deterministic closure --------------------------------------------
    def _auto_decision(self, env: KernelEnv, host) -> dict | None:
        if not env.results:
            return None
        direction = self.task.direction
        best = best_candidate(list(env.results.values()), direction)
        if best is None:
            return None
        primary = ((best.get("metrics") or {}).get("selection") or {}).get("primary")
        if primary is None:
            return None
        try:
            deterministic = gate(self.task.task, float(host.parent_primary()),
                                 float(primary), self.task.margin)
        except Exception:
            deterministic = None
        if deterministic is None:
            return None
        return {"requested": deterministic.decision,
                "decision": deterministic.decision,
                "experiment_id": best.get("experiment_id"),
                "reason": f"kernel did not decide; deterministic gate closed the round: "
                          f"{deterministic.reason}",
                "delta": deterministic.delta,
                "gate_reason": deterministic.reason,
                "source": "kernel_autoclose"}

    @staticmethod
    def _auto_summary(outcome: RoundOutcome) -> str:
        decision = (outcome.decision or {}).get("decision", "retain")
        return (f"kernel autoclose: {len(outcome.results)} experiment(s) executed, "
                f"decision={decision}, reason={(outcome.decision or {}).get('reason', '')}"[:600])
