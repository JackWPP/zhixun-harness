"""Kernel tool surface: the autonomous agent's hands.

Unlike the read-only coordinator role tools, the kernel can *act*: execute
experiments, make a gated decision, delegate to sub-agents, convene a panel and
write memory/skills. Every mutating tool passes through :class:`Guard` first,
and every call is appended to ``env.trace`` for the experiment record.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from rsi.executor import ExperimentProposal
from rsi.records import utcnow_iso
from rsi.tasks import TaskConfig

from . import delegate as delegate_mod
from .guard import Guard, GuardViolation
from .prompts_kernel import KERNEL_SYSTEM, kernel_round_prompt
from .runtime import PIRuntime
from .views import agent_view, search_selection_records


def _result(payload, details: dict | None = None):
    from pi_agent_core.types import AgentToolResult
    payload = agent_view(payload)
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > 7000:
        text = text[:7000] + "...(truncated)"
    return AgentToolResult(content=[{"type": "text", "text": text}],
                           details=agent_view(details) if details else {"ok": bool(payload.get("ok", True))})


@dataclass
class KernelEnv:
    task: TaskConfig
    host: object                       # supervisor callback object, see runner.TaskHost
    guard: Guard
    runtime: PIRuntime
    memory: object = None
    skills: object = None
    recall: object = None
    brief: dict = field(default_factory=dict)
    proposals: dict[str, ExperimentProposal] = field(default_factory=dict)
    results: dict[str, dict] = field(default_factory=dict)
    decision: dict | None = None
    finished: dict | None = None
    trace: list[dict] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    used_skills: list[str] = field(default_factory=list)
    memory_writes: int = 0
    skill_writes: int = 0
    delegates: int = 0
    panels: int = 0
    experiments_round: int = 0
    experiments_total: int = 0
    max_experiments_round: int = 3
    plan_seq: int = 0
    started_at: float = field(default_factory=time.time)

    # -- helpers -----------------------------------------------------------
    def note(self, tool: str, args: dict | None = None, ok: bool = True,
             detail: str = "") -> None:
        self.trace.append({"tool": tool, "ok": bool(ok), "detail": str(detail)[:400],
                           "args": _shrink(args or {}, 600), "at": utcnow_iso()})

    def plan_list(self) -> list[dict]:
        return [{"plan_id": pid, "strategy": p.strategy, "params": p.params,
                 "postprocess": p.postprocess, "hypothesis": p.hypothesis,
                 "expected_effect": p.expected_effect, "risk": p.risk}
                for pid, p in self.proposals.items()]

    def summary(self) -> dict:
        return {
            "experiments_round": self.experiments_round,
            "experiments_total": self.experiments_total,
            "proposals": len(self.proposals),
            "executed": len(self.results),
            "decision": self.decision,
            "finished": self.finished,
            "memory_writes": self.memory_writes,
            "skill_writes": self.skill_writes,
            "delegates": self.delegates,
            "panels": self.panels,
            "used_skills": list(dict.fromkeys(self.used_skills)),
            "violations": self.violations[-6:],
            "trace": self.trace[-40:],
            "elapsed_seconds": round(time.time() - self.started_at, 2),
        }


def _shrink(value, limit: int):
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + "...(truncated)"


# ---------------------------------------------------------------------------
# parameter models
# ---------------------------------------------------------------------------
class EmptyParams(BaseModel):
    pass


class HistoryParams(BaseModel):
    limit: int = Field(default=5, ge=1, le=10)
    task: str = Field(default="", description="optional task key")


class LessonsParams(BaseModel):
    task: str = Field(default="", description="optional task key")
    strategy: str = Field(default="", description="optional strategy filter")
    limit: int = Field(default=4, ge=1, le=8)


class SearchParams(BaseModel):
    query: str = Field(description="keyword(s) to search experiment records")
    limit: int = Field(default=8, ge=1, le=20)


class ContextParams(BaseModel):
    section: Literal["task", "drift", "history", "metrics", "catalog", "budget",
                     "memory", "skills"] = Field(description="context section")


class ReadSkillParams(BaseModel):
    name: str = Field(description="skill name from list_skills")


class ProposeParams(BaseModel):
    strategy: str = Field(description="strategy name from the catalog")
    params_json: str = Field(default="{}", description="strategy params as JSON object")
    postprocess_json: str = Field(default="{}",
                                  description='postprocess as JSON object, e.g. {"type":"recenter","mode":"batch_mean","weight":0.4}')
    hypothesis: str = Field(description="falsifiable hypothesis (what should improve and why)")
    expected_effect: str = Field(default="", description="expected metric change, e.g. +0.02 macro_f1")
    risk: str = Field(default="", description="main risk of this experiment")


class RunParams(BaseModel):
    plan_id: str = Field(description="plan id returned by propose_experiment")


class CompareParams(BaseModel):
    experiment_ids: list[str] = Field(default_factory=list,
                                      description="ids to compare; empty means all")


class DecideParams(BaseModel):
    decision: Literal["promote", "retain", "rollback"] = Field(
        description="promote only if the public selection metric crosses the margin")
    experiment_id: str = Field(default="", description="candidate to promote (required for promote)")
    reason: str = Field(description="evidence-based reason, cite metrics/sha")


class PanelParams(BaseModel):
    issue: str = Field(description="what the team should resolve; be specific")


class DelegateParams(BaseModel):
    role: Literal["experimenter", "diagnostician", "critic", "arbiter", "reporter"]
    brief: str = Field(description="self-contained brief for the sub-agent")


class MemoryAddParams(BaseModel):
    text: str = Field(description="one concrete lesson/observation, <= 400 chars")
    scope: Literal["task", "global"] = Field(default="task")


class MemoryReplaceParams(BaseModel):
    old_text: str = Field(description="unique substring of the entry to replace")
    text: str = Field(description="new entry text")
    scope: Literal["task", "global"] = Field(default="task")


class MemoryRemoveParams(BaseModel):
    old_text: str = Field(description="unique substring of the entry to remove")
    scope: Literal["task", "global"] = Field(default="task")


class SkillManageParams(BaseModel):
    action: Literal["create", "patch", "delete"] = Field(description="skill management action")
    name: str = Field(description="skill name (lowercase, 3-40 chars)")
    text: str = Field(default="", description="full SKILL.md text for create")
    old_string: str = Field(default="", description="unique substring to replace for patch")
    new_string: str = Field(default="", description="replacement text for patch")


class FinishRoundParams(BaseModel):
    summary: str = Field(description="what you did this round and why, <= 600 chars")
    lesson: str = Field(default="", description="optional one-line lesson to remember")


# ---------------------------------------------------------------------------
# tool builder
# ---------------------------------------------------------------------------
def build_kernel_tools(env: KernelEnv):
    from pi_agent_core.tools import SimpleTool

    def tool(name: str, description: str, parameters, handler):
        return SimpleTool(name=name, label=name, description=description,
                          parameters=parameters, execute_fn=handler)

    def mem_target(scope: str) -> tuple[str, str | None]:
        return ("memory", None) if scope == "global" else ("task", env.task.key)

    # -- read-only ---------------------------------------------------------
    async def h_state(tool_call_id, params, signal, on_update):
        payload = {k: env.brief.get(k) for k in
                   ("task", "goal", "metrics", "budget", "round_index", "rounds_total",
                    "drift", "memory", "lessons", "stats", "skills", "history")}
        payload["experiments"] = env.summary()
        payload["plans"] = env.plan_list()
        env.note("get_run_state")
        return _result(payload)

    async def h_history(tool_call_id, params, signal, on_update):
        items = env.brief.get("history", []) or []
        env.note("get_history", {"limit": params.limit})
        return _result({"ok": True, "history": items[-params.limit:]})

    async def h_lessons(tool_call_id, params, signal, on_update):
        task = params.task or env.task.key
        text = env.recall.lessons_text(task, params.strategy or None, limit=params.limit)
        env.note("get_lessons", {"task": task, "strategy": params.strategy})
        return _result({"ok": True, "lessons": text})

    async def h_search(tool_call_id, params, signal, on_update):
        rows = search_selection_records(env.recall, params.query,
                                        task=env.task.key, limit=params.limit)
        env.note("search_records", {"query": params.query})
        return _result({"ok": True, "results": rows})

    async def h_context(tool_call_id, params, signal, on_update):
        context = env.host.context()
        env.note("get_context", {"section": params.section})
        return _result({"ok": True, "section": params.section,
                        "content": context.get(params.section, {})})

    async def h_skills(tool_call_id, params, signal, on_update):
        env.note("list_skills")
        return _result({"ok": True, "skills": env.skills.index_text()})

    async def h_read_skill(tool_call_id, params, signal, on_update):
        try:
            text = env.skills.read_text(params.name)
        except Exception as exc:
            return _result({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
        env.used_skills.append(params.name)
        env.note("read_skill", {"name": params.name})
        return _result({"ok": True, "name": params.name, "skill": text})

    # -- experiment loop ---------------------------------------------------
    async def h_propose(tool_call_id, params, signal, on_update):
        try:
            params_obj = json.loads(params.params_json or "{}")
            postprocess = json.loads(params.postprocess_json or "{}")
            if not isinstance(params_obj, dict) or not isinstance(postprocess, dict):
                raise ValueError("params_json/postprocess_json must be JSON objects")
        except Exception as exc:
            return _result({"ok": False, "error": f"invalid JSON: {exc}"})
        proposal = ExperimentProposal(strategy=params.strategy, params=params_obj,
                                      postprocess=postprocess,
                                      hypothesis=params.hypothesis,
                                      expected_effect=params.expected_effect,
                                      risk=params.risk)
        try:
            env.guard.validate_catalog(proposal)
        except GuardViolation as exc:
            env.violations.append({"tool": "propose_experiment", "reason": str(exc)})
            return _result({"ok": False, "error": str(exc)})
        env.plan_seq += 1
        plan_id = f"plan-{env.plan_seq:02d}"
        env.proposals[plan_id] = proposal
        env.note("propose_experiment", {"plan_id": plan_id, "strategy": params.strategy})
        return _result({"ok": True, "plan_id": plan_id, "plans": env.plan_list()})

    async def h_run(tool_call_id, params, signal, on_update):
        if env.decision is not None:
            return _result({"ok": False,
                            "error": "decision already recorded this round; no more experiments"})
        proposal = env.proposals.get(params.plan_id)
        if proposal is None:
            return _result({"ok": False, "error": f"unknown plan_id {params.plan_id!r}"})
        try:
            clean = env.guard.validate_experiment(proposal, env.experiments_round,
                                                  env.experiments_total)
        except GuardViolation as exc:
            env.violations.append({"tool": "run_experiment", "reason": str(exc)})
            return _result({"ok": False, "error": str(exc)})
        env.experiments_round += 1
        env.experiments_total += 1
        result = env.host.run_experiment(clean)
        env.results[result["experiment_id"]] = result
        env.note("run_experiment", {"plan_id": params.plan_id,
                                    "experiment_id": result["experiment_id"]},
                 ok=bool(result.get("ok")), detail=str(result.get("error") or ""))
        return _result({"ok": bool(result.get("ok")), "experiment": result})

    async def h_compare(tool_call_id, params, signal, on_update):
        ids = params.experiment_ids or list(env.results)
        rows = [env.results[i] for i in ids if i in env.results]
        direction = -1 if env.task.direction == "higher" else 1
        rows.sort(key=lambda r: ((r.get("metrics") or {}).get("selection") or {}).get("primary", 0) * direction)
        env.note("compare_candidates", {"ids": ids})
        return _result({"ok": True, "direction": env.task.direction, "rows": rows})

    async def h_decide(tool_call_id, params, signal, on_update):
        if env.decision is not None:
            return _result({"ok": False, "error": "decision already recorded this round"})
        try:
            gate_decision = env.guard.check_decision(
                params.decision, params.experiment_id or None, env.results,
                env.host.parent_primary(), env.experiments_round)
        except GuardViolation as exc:
            env.violations.append({"tool": "decide", "decision": params.decision,
                                   "reason": str(exc)})
            return _result({"ok": False, "error": f"guard rejected decision: {exc}"})
        env.decision = {"requested": params.decision, "decision": gate_decision.decision,
                        "experiment_id": params.experiment_id, "reason": params.reason,
                        "delta": gate_decision.delta, "gate_reason": gate_decision.reason,
                        "source": "kernel"}
        env.note("decide", {"decision": gate_decision.decision,
                            "experiment_id": params.experiment_id})
        return _result({"ok": True, "decision": env.decision})

    async def h_finish(tool_call_id, params, signal, on_update):
        if env.decision is None:
            return _result({"ok": False, "error": "call decide before finish_round"})
        env.finished = {"summary": params.summary, "lesson": params.lesson}
        env.note("finish_round")
        return _result({"ok": True, "round": env.summary()})

    # -- collaboration -----------------------------------------------------
    async def h_delegate(tool_call_id, params, signal, on_update):
        try:
            result = await delegate_mod.delegate(
                env.runtime, params.role, params.brief, env.host.context(),
                max_turns=6, timeout_seconds=300)
        except Exception as exc:  # pragma: no cover - network failure
            return _result({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
        env.delegates += 1
        env.note("delegate", {"role": params.role}, ok=bool(result.get("ok")))
        return _result(result)

    async def h_panel(tool_call_id, params, signal, on_update):
        if env.panels >= 2:
            return _result({"ok": False, "error": "panel already used twice this task"})
        try:
            entries = await delegate_mod.run_panel(env.runtime, env.host.context(),
                                                   params.issue)
        except Exception as exc:  # pragma: no cover - network failure
            return _result({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
        env.panels += 1
        env.note("convene_panel", {"issue": params.issue[:120]})
        try:
            env.host.record_panel(entries)
        except Exception:
            pass
        return _result({"ok": True, "panel": entries})

    # -- memory / skills ---------------------------------------------------
    async def h_memory_add(tool_call_id, params, signal, on_update):
        try:
            text = env.guard.scan_text(params.text, "memory")
        except GuardViolation as exc:
            return _result({"ok": False, "error": str(exc)})
        target, task_key = mem_target(params.scope)
        result = env.memory.add(text, target=target, task_key=task_key).to_dict()
        env.memory_writes += 1
        env.note("memory_add", {"scope": params.scope}, ok=result["ok"])
        return _result(result)

    async def h_memory_replace(tool_call_id, params, signal, on_update):
        try:
            text = env.guard.scan_text(params.text, "memory")
        except GuardViolation as exc:
            return _result({"ok": False, "error": str(exc)})
        target, task_key = mem_target(params.scope)
        result = env.memory.replace(params.old_text, text, target=target,
                                    task_key=task_key).to_dict()
        env.memory_writes += 1
        env.note("memory_replace", {"scope": params.scope}, ok=result["ok"])
        return _result(result)

    async def h_memory_remove(tool_call_id, params, signal, on_update):
        target, task_key = mem_target(params.scope)
        result = env.memory.remove(params.old_text, target=target,
                                   task_key=task_key).to_dict()
        env.memory_writes += 1
        env.note("memory_remove", {"scope": params.scope}, ok=result["ok"])
        return _result(result)

    async def h_skill(tool_call_id, params, signal, on_update):
        try:
            if params.action == "create":
                env.guard.check_skill_text(params.text)
                result = env.skills.create(params.name, params.text)
            elif params.action == "patch":
                env.guard.check_skill_text(params.new_string)
                result = env.skills.patch(params.name, params.old_string, params.new_string)
            else:
                result = env.skills.delete(params.name)
        except GuardViolation as exc:
            return _result({"ok": False, "error": str(exc)})
        except Exception as exc:
            return _result({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
        env.skill_writes += 1
        env.note("skill_manage", {"action": params.action, "name": params.name},
                 ok=bool(result.get("ok", True)))
        return _result(result)

    return [
        tool("get_run_state", "读取当前运行状态、目标进度、父模型指标与预算（无隐藏数据）。",
             EmptyParams, h_state),
        tool("get_history", "读取最近几轮的精简记录（决策、指标、策略、哈希）。",
             HistoryParams, h_history),
        tool("get_lessons", "读取跨轮次沉淀的经验教训（可按策略过滤）。",
             LessonsParams, h_lessons),
        tool("search_records", "检索本任务的 selection 实验记录；不检索保留评测结果。",
             SearchParams, h_search),
        tool("get_context", "读取某一节上下文包（drift/history/catalog/memory 等）。",
             ContextParams, h_context),
        tool("list_skills", "列出可用的过程性技能索引（渐进披露第 0 层）。",
             EmptyParams, h_skills),
        tool("read_skill", "读取某个技能的完整说明（渐进披露第 1 层）。",
             ReadSkillParams, h_read_skill),
        tool("propose_experiment", "登记一个实验方案（不执行）。参数必须来自策略目录。",
             ProposeParams, h_propose),
        tool("run_experiment", "执行已登记的实验并返回公开 selection 指标。",
             RunParams, h_run),
        tool("compare_candidates", "按主指标方向比较已执行候选。",
             CompareParams, h_compare),
        tool("decide", "记录本轮最终决策；promote 必须通过确定性门控。",
             DecideParams, h_decide),
        tool("finish_round", "提交本轮摘要与可选经验，结束本轮。",
             FinishRoundParams, h_finish),
        tool("delegate", "把子任务交给隔离上下文的子 Agent（诊断/实验/评审/仲裁/记录）。",
             DelegateParams, h_delegate),
        tool("convene_panel", "召四角色专家组会诊卡点（每任务最多 2 次）。",
             PanelParams, h_panel),
        tool("memory_add", "追加一条有界记忆（跨轮/跨 run 复用）。",
             MemoryAddParams, h_memory_add),
        tool("memory_replace", "替换一条记忆（用唯一子串定位）。",
             MemoryReplaceParams, h_memory_replace),
        tool("memory_remove", "删除一条记忆（用唯一子串定位）。",
             MemoryRemoveParams, h_memory_remove),
        tool("skill_manage", "创建/修改/删除过程性技能（SKILL.md，有大小与安全限制）。",
             SkillManageParams, h_skill),
    ]


KERNEL_SYSTEM_PROMPT = KERNEL_SYSTEM
__all__ = ["KernelEnv", "build_kernel_tools", "KERNEL_SYSTEM_PROMPT",
           "kernel_round_prompt"]
