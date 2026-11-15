"""Structured tool surface exposed to the PI agents.

Agents never get shell/file write access. They can only (a) read a compact
context packet through ``get_context`` and (b) submit structured decisions
through a ``submit_*`` tool whose arguments are validated by pydantic. The tool
sink is the single source of truth for the coordinator; free-form text is only
a fallback if the model ignores the tool contract.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field


class ContextParams(BaseModel):
    section: Literal["task", "drift", "history", "metrics", "catalog", "budget"] = Field(
        description="which context packet to read")


class DiagnosisParams(BaseModel):
    summary: str = Field(description="one-paragraph drift diagnosis")
    severity: Literal["stable", "mild", "moderate", "severe"] = Field(
        description="overall drift severity")
    causes: list[str] = Field(description="ranked likely causes, 1-4 items")
    focus: list[str] = Field(description="what the next experiment should target, 1-3 items")
    evidence: str = Field(description="quantitative evidence copied from the context")
    confidence: float = Field(ge=0.0, le=1.0, description="confidence 0-1")


class PlanParams(BaseModel):
    strategy: str = Field(description="strategy name from the catalog")
    params_json: str = Field(default="{}", description="JSON object with strategy parameters")
    postprocess_json: str = Field(default="{}", description="JSON object with postprocess config")
    hypothesis: str = Field(description="falsifiable hypothesis for this experiment")
    expected_effect: str = Field(default="", description="expected metric change")
    risk: str = Field(default="", description="main risk of this experiment")


class ReviewParams(BaseModel):
    approvals: list[int] = Field(default_factory=list,
                                 description="0-based plan indices approved for execution")
    vetoes: list[str] = Field(default_factory=list,
                              description='veto reasons as "<index>: <reason>"')
    note: str = Field(default="", description="short reviewer note")


class VerdictParams(BaseModel):
    decision: Literal["promote", "retain", "rollback"] = Field(
        description="final decision suggested to the deterministic gate")
    chosen: int = Field(ge=0, description="0-based index of the candidate to keep")
    reason: str = Field(description="why this candidate")
    confidence: float = Field(ge=0.0, le=1.0)


class StandpointParams(BaseModel):
    standpoint: str = Field(description="your position in one paragraph")
    evidence: str = Field(description="evidence from the context")
    confidence: float = Field(ge=0.0, le=1.0)


class ToolSink:
    def __init__(self):
        self.payloads: list[dict] = []
        self.calls: list[str] = []

    def add(self, tool: str, payload: dict) -> None:
        self.calls.append(tool)
        self.payloads.append(payload)


def _result(text: str, details=None):
    from pi_agent_core.types import AgentToolResult
    return AgentToolResult(content=[{"type": "text", "text": text}],
                           details=details or {"ok": True})


def _submit_handler(kind: str, sink: ToolSink, store_key: str | None = None):
    async def handler(tool_call_id, params, signal, on_update):
        payload = params.model_dump() if hasattr(params, "model_dump") else dict(params)
        record = {"kind": kind, **payload}
        if store_key:
            record["store_key"] = store_key
        sink.add(kind, record)
        return _result(json.dumps({"accepted": True, "kind": kind}, ensure_ascii=False))
    return handler


def context_handler(context: dict):
    async def handler(tool_call_id, params, signal, on_update):
        section = getattr(params, "section", "task")
        payload = context.get(section, {"note": "section unavailable"})
        text = json.dumps(payload, ensure_ascii=False)
        if len(text) > 6000:
            text = text[:6000] + "...(truncated)"
        return _result(text)
    return handler


def build_toolset(kind: str, sink: ToolSink, context: dict):
    """Return the SimpleTool list for a role.

    kind is one of: diagnosis, plan, review, verdict, standpoint.
    """
    from pi_agent_core.tools import SimpleTool

    tools = [SimpleTool(name="get_context", label="get_context",
                        description="Read a compact context packet (task/drift/history/metrics/catalog/budget).",
                        parameters=ContextParams, execute_fn=context_handler(context))]
    if kind == "diagnosis":
        tools.append(SimpleTool(name="submit_diagnosis", label="submit_diagnosis",
                                description="Submit the drift diagnosis. Call exactly once.",
                                parameters=DiagnosisParams,
                                execute_fn=_submit_handler("diagnosis", sink)))
    elif kind == "plan":
        tools.append(SimpleTool(name="submit_plan", label="submit_plan",
                                description="Submit one experiment plan. Call 1-3 times, each call is one plan.",
                                parameters=PlanParams,
                                execute_fn=_submit_handler("plan", sink)))
    elif kind == "review":
        tools.append(SimpleTool(name="submit_review", label="submit_review",
                                description="Veto or approve the proposed plans. Call exactly once.",
                                parameters=ReviewParams,
                                execute_fn=_submit_handler("review", sink)))
    elif kind == "verdict":
        tools.append(SimpleTool(name="submit_verdict", label="submit_verdict",
                                description="Submit the arbitration verdict. Call exactly once.",
                                parameters=VerdictParams,
                                execute_fn=_submit_handler("verdict", sink)))
    elif kind == "reporter":
        pass  # read-only role: only get_context is exposed
    elif kind == "standpoint":
        tools.append(SimpleTool(name="submit_standpoint", label="submit_standpoint",
                                description="Submit your panel standpoint. Call exactly once.",
                                parameters=StandpointParams,
                                execute_fn=_submit_handler("standpoint", sink)))
    else:
        raise ValueError(f"unknown toolset kind: {kind}")
    return tools
