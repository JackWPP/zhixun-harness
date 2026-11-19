"""Delegate tool: isolated sub-agents for experiment design / diagnosis / review.

Each sub-agent gets its own message list and a read-only + submit_* toolset, so
it cannot touch the filesystem, the budget ledger or the parent state. Results
come back as small structured payloads that count toward the same token budget.
"""
from __future__ import annotations

import asyncio
import time

from . import prompts
from .runtime import PIRuntime, RoleResult
from .tools import ToolSink, build_toolset
from .views import agent_view

ROLE_TOOLSETS = {
    "experimenter": "plan",
    "diagnostician": "diagnosis",
    "critic": "review",
    "arbiter": "verdict",
    "reporter": "reporter",
}

ROLE_BRIEFS = {
    "experimenter": "从策略目录中提出 1-3 个可执行实验，每个实验单独调用 submit_plan。",
    "diagnostician": "分析当前漂移，调用 submit_diagnosis 给出诊断与下一步重点。",
    "critic": "审查候选实验，调用 submit_review 批准或否决。",
    "arbiter": "在指标接近时给最终建议，调用 submit_verdict。",
    "reporter": "用 150 字以内总结本轮事实，不要调用工具。",
}


async def call_role_agent(runtime: PIRuntime, role: str, brief: str, context: dict, *,
                          max_turns: int = 6, tool_timeout: float = 90.0,
                          timeout_seconds: float = 300.0) -> RoleResult:
    start = time.time()
    if role not in ROLE_TOOLSETS:
        return RoleResult(role=role, error=f"unknown role {role!r}", latency=0.0)
    if not runtime.available():
        return runtime._failure(role, runtime._error or "LLM runtime unavailable", start)
    if runtime.budget.exhausted():
        return runtime._failure(role, "token budget exhausted", start)
    sink = ToolSink()
    tools = build_toolset(ROLE_TOOLSETS[role], sink, context)
    system_prompt = prompts.SYSTEM_PROMPTS[role]
    user_prompt = f"{agent_view(brief)}\n{ROLE_BRIEFS.get(role, '')}"
    try:
        agent = await runtime.arun_agent(
            system_prompt, user_prompt, tools,
            max_turns=max_turns, tool_timeout=tool_timeout,
            timeout_seconds=timeout_seconds)
        return runtime.collect_agent(agent, role=role, sink=sink, start=start)
    except asyncio.TimeoutError:
        return runtime._failure(role, f"timeout after {timeout_seconds:.0f}s", start)
    except Exception as exc:  # pragma: no cover - network/runtime failure
        return runtime._failure(role, f"{exc.__class__.__name__}: {exc}", start)


async def delegate(runtime: PIRuntime, role: str, brief: str, context: dict, *,
                   max_turns: int = 6, timeout_seconds: float = 300.0) -> dict:
    result = await call_role_agent(runtime, role, brief, context,
                                   max_turns=max_turns,
                                   timeout_seconds=timeout_seconds)
    return {"role": role, **result.to_dict(), "submitted": result.submitted[:4]}


async def run_panel(runtime: PIRuntime, context: dict, issue: str, *,
                    timeout_seconds: float = 480.0) -> list[dict]:
    """Run four role standpoints concurrently (Hermes-style panel)."""
    roles = ["diagnostician", "experimenter", "critic", "reporter"]

    async def one(role: str) -> dict:
        start = time.time()
        if not runtime.available():
            return {"role": role, "ok": False, "standpoint": "", "evidence": "",
                    "confidence": 0.0, "error": "LLM runtime unavailable"}
        sink = ToolSink()
        tools = build_toolset("standpoint", sink, context)
        system_prompt = prompts.SYSTEM_PROMPTS["panel"].format(role=role)
        user_prompt = prompts.standpoint_prompt(context, role, issue)
        try:
            agent = await runtime.arun_agent(
                system_prompt, user_prompt, tools, max_turns=4, tool_timeout=60,
                timeout_seconds=timeout_seconds)
            result = runtime.collect_agent(agent, role=role, sink=sink, start=start)
            payload = result.submitted[0] if result.submitted else {}
            return {"role": role, "ok": result.ok,
                    "standpoint": payload.get("standpoint", ""),
                    "evidence": payload.get("evidence", ""),
                    "confidence": payload.get("confidence", 0.0),
                    "usage": result.usage, "error": result.error}
        except Exception as exc:  # pragma: no cover - network failure
            return {"role": role, "ok": False, "standpoint": "", "evidence": "",
                    "confidence": 0.0, "error": f"{exc.__class__.__name__}: {exc}"}

    results = await asyncio.gather(*(one(role) for role in roles), return_exceptions=True)
    out: list[dict] = []
    for role, item in zip(roles, results):
        if isinstance(item, BaseException):
            out.append({"role": role, "ok": False, "standpoint": "", "evidence": "",
                        "confidence": 0.0, "error": str(item)})
        else:
            out.append(item)
    return out
