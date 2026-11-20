"""Host-gateway agent driver for workspace-v2 submissions.

The competition exposes the language model as ``rnd.model(...)`` (a host RPC),
not as an HTTP endpoint.  This module turns that RPC into the small runtime
surface the PI-agent kernel expects (``arun_agent`` / ``collect_agent`` /
``run_coro`` + ``TokenBudget``), so the *same* kernel code that runs against
pi-agent-core can drive a platform round.

Two transports are implemented:

* native tool calls — ``rnd.model(messages, tools=..., tool_choice="auto")``;
  used only when the host supports it (``RSI_V2_NATIVE_TOOLS=1``);
* a strict text-action protocol — one JSON object per assistant turn; this is
  the default because the workspace-v2 gateway adapter is not guaranteed to
  forward ``tools``.

All model calls are real host calls, every ``call_id`` is collected for
``rnd.finish(model_call_ids=...)``, and nothing here writes to stdout.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rsi.agent.config import TokenBudget
from rsi.agent.guard import Guard
from rsi.agent.kernel import Kernel
from rsi.agent.prompts_kernel import KERNEL_SYSTEM, kernel_round_prompt
from rsi.agent.runtime import RoleResult
from rsi.platform import plans
from rsi.tasks import TaskConfig

HARNESS_ROOT = Path(__file__).resolve().parents[2]

TOOL_MANUAL = """工具协议（必须严格遵守）：
每次回复只输出一个 JSON 对象，不要输出任何解释或 Markdown 代码块：
{"tool": "<工具名>", "args": {<参数对象>}}
可用工具：
%s
规则：
- 一轮最多 3 个实验；run_experiment 会真实执行候选（可能耗时数十秒）。
- 只能使用官方 train 数据与公开 validation；禁止猜测隐藏标签。
- 决定后调用 decide，然后调用 finish_round 总结本轮并结束。
- 若没有合适动作，直接调用 finish_round。"""


def _tool_manual(tools) -> str:
    lines = []
    for tool in tools:
        description = str(getattr(tool, "description", "") or "").strip().replace("\n", " ")
        lines.append("- %s: %s" % (tool.name, description[:240]))
    return TOOL_MANUAL % ("\n".join(lines))


def _schema(tool) -> dict:
    params = getattr(tool, "parameters", None)
    if isinstance(params, type):
        for method in ("model_json_schema", "schema"):
            fn = getattr(params, method, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:  # pragma: no cover - pydantic version drift
                    pass
    elif isinstance(params, dict):
        return params
    return {"type": "object", "properties": {}}


def _tool_spec(tool) -> dict:
    description = str(getattr(tool, "description", "") or "").strip()[:600]
    return {"type": "function",
            "function": {"name": tool.name, "description": description,
                         "parameters": _schema(tool)}}


def _params_for(tool, args: dict):
    cls = getattr(tool, "parameters", None)
    if isinstance(cls, type):
        try:
            return cls(**args)
        except Exception:
            return None
    return args


def _content_text(result) -> str:
    blocks = getattr(result, "content", None)
    if blocks is None and isinstance(result, dict):
        blocks = result.get("content")
    if isinstance(blocks, str):
        return blocks
    parts = []
    for block in blocks or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        else:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
    if not parts:
        parts.append(str(result)[:500])
    return "\n".join(parts)


def _clip(text: str, limit: int = 4000) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _balanced_json(text: str):
    """Return the first JSON object found in *text* (tolerates prose/code fences)."""
    text = str(text or "")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_action(text: str):
    obj = _balanced_json(text)
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool") or obj.get("action") or obj.get("name")
    if not name:
        return None
    args = obj.get("args", obj.get("arguments", obj.get("params", obj.get("input", {}))))
    if not isinstance(args, dict):
        args = {}
    return str(name), args


def _response_message(response):
    """Extract ``choices[0].message`` from a dict or object host response."""
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if not choices:
        return None
    first = choices[0]
    return first.get("message") if isinstance(first, dict) else getattr(first, "message", None)


def _message_text(message) -> str:
    if isinstance(message, dict):
        content = message.get("content")
        # OpenAI-style content may be a list of parts.
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content
                           if isinstance(part, dict))
        return str(content or "")
    return str(getattr(message, "content", "") or "")


class GatewayRuntime:
    """Minimal PI-runtime-compatible driver on top of ``rnd.model``."""

    backend = "host-gateway"

    def __init__(self, rnd, *, max_turns: int = 6, token_cap: int = 0,
                 timeout_seconds: float = 900.0):
        self.rnd = rnd
        self.max_turns = max(1, int(max_turns))
        self.budget = TokenBudget(cap=max(0, int(token_cap)))
        self.timeout_seconds = float(timeout_seconds)
        self.calls = 0
        self.call_ids: list[str] = []
        self.tool_calls = 0
        self._last_usage: dict = {}
        # The platform host advertises the OpenAI tool-calling options in the
        # SDK whitelist; PyTRiO's Qwen3.8 gateway defaults to *thinking on*, and
        # with the platform's 256-token output cap the reasoning trace can eat
        # the whole budget (content=""), which breaks the text JSON protocol.
        # Native tool_calls still arrive in that case, so prefer them and try to
        # disable thinking explicitly; both degrade gracefully when unsupported.
        flag = os.environ.get("RSI_V2_NATIVE_TOOLS", "1").strip().lower()
        self.native_tools = flag not in {"0", "off", "false", "no"}
        mode = os.environ.get("RSI_V2_KERNEL_THINKING", "off").strip().lower()
        self.thinking = mode in {"on", "true", "1", "yes"}
        self.thinking_supported = True
        self.native_supported = self.native_tools

    # -- PI runtime surface ------------------------------------------------
    def available(self) -> bool:
        return True

    def run_coro(self, coro, timeout: float | None = None):
        limit = float(timeout or self.timeout_seconds or 0) or None

        async def _run():
            if limit:
                return await asyncio.wait_for(coro, timeout=limit)
            return await coro

        return asyncio.run(_run())

    async def arun_agent(self, system, user, tools, max_turns: int = 6,
                         tool_timeout: float = 60.0, timeout_seconds: float | None = None):
        limit = float(timeout_seconds or self.timeout_seconds or 0) or None

        async def _run():
            return await self._loop(system, user, tools,
                                    turns=min(int(max_turns or self.max_turns), self.max_turns))

        if limit:
            return await asyncio.wait_for(_run(), timeout=limit)
        return await _run()

    def collect_agent(self, agent, role: str = "kernel", sink=None, start=None):
        latency = time.time() - float(start or time.time())
        result = RoleResult(role=role, ok=agent.get("ok", False),
                            text=str(agent.get("text", ""))[:800],
                            usage=self.budget.to_dict(), error=agent.get("error"),
                            latency=latency, tool_calls=int(agent.get("tool_calls", 0)))
        if sink is not None:
            try:
                sink(result)
            except Exception:  # pragma: no cover - optional listener
                pass
        return result

    def call(self, system, user, max_tokens: int = 512):
        receipt = self.model([{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                             max_tokens=max_tokens)
        message = _response_message(getattr(receipt, "response", None))
        return RoleResult(role="kernel", ok=receipt is not None,
                          text=_message_text(message) if message is not None else "",
                          usage=self.budget.to_dict())

    # -- host model --------------------------------------------------------
    def _max_output(self) -> int:
        try:
            value = int(os.environ.get("RSI_V2_MAX_OUTPUT", "256") or 256)
        except (TypeError, ValueError):
            value = 256
        return max(64, min(4096, value))

    def _model_options(self, *, tools=None) -> dict:
        options: dict = {"max_tokens": None, "temperature": 0}
        if tools and self.native_supported:
            options["tools"] = [_tool_spec(tool) for tool in tools]
            options["tool_choice"] = "auto"
        if self.thinking_supported:
            options["enable_thinking"] = bool(self.thinking)
        return options

    def model(self, messages, *, max_tokens: int | None = None, tools=None):
        if max_tokens is None:
            max_tokens = self._max_output()
        last: Exception | None = None
        for attempt in range(3):
            options = self._model_options(tools=tools)
            options["max_tokens"] = max_tokens
            try:
                receipt = self.rnd.model(messages, **options)
                self.calls += 1
                call_id = str(getattr(receipt, "call_id", "") or "")
                if call_id:
                    self.call_ids.append(call_id)
                usage = _receipt_usage(receipt)
                if usage:
                    self._last_usage = usage
                    self.budget.add(usage)
                return receipt
            except Exception as error:  # host RPC is occasionally transient
                last = error
                # Some hosts reject options they do not implement.  Downgrade
                # capability one step at a time instead of failing the round.
                if self.thinking_supported:
                    self.thinking_supported = False
                    continue
                if self.native_supported:
                    self.native_supported = False
                    continue
                time.sleep(0.5 * (attempt + 1))
        raise last if last is not None else RuntimeError("host model call failed")

    async def _loop(self, system, user, tools, *, turns: int):
        tool_map = {tool.name: tool for tool in tools}
        manual = _tool_manual(tools)
        messages = [{"role": "system", "content": str(system)},
                    {"role": "user", "content": str(user) + "\n\n" + manual}]
        last_text = ""
        idle = 0
        error = None
        for turn in range(turns):
            if self.budget.exhausted():
                error = "token budget exhausted"
                break
            try:
                receipt = self.model(messages, tools=tools)
            except Exception as exc:
                error = "%s: %s" % (type(exc).__name__, exc)
                break
            message = _response_message(getattr(receipt, "response", None))
            if message is None:
                error = "malformed host gateway response"
                break
            last_text = _message_text(message)
            if os.environ.get("SIM_AGENT_DEBUG"):
                print("[agent] turn=%d text=%r" % (turn, last_text[:900]), file=sys.stderr, flush=True)
            native_calls = _native_tool_calls(message)
            if native_calls:
                tool_name = native_calls[0][0]
                tool_args = native_calls[0][1]
                action = (tool_name, tool_args)
            else:
                action = extract_action(last_text)
            if action is None:
                idle += 1
                if idle >= 2:
                    error = "model did not emit a tool action"
                    break
                messages.append({"role": "assistant", "content": last_text or "(empty reply)"})
                messages.append({"role": "user", "content":
                                 "未检测到工具调用。请只输出一个 JSON 对象："
                                 '{"tool": "...", "args": {...}}，或调用 finish_round。'})
                continue
            idle = 0
            name, args = action
            tool = tool_map.get(name)
            if tool is None:
                messages.append({"role": "assistant", "content": last_text or "(unknown tool)"})
                messages.append({"role": "user", "content":
                                 "未知工具 %r。可用工具：%s" % (name, ", ".join(tool_map))})
                continue
            params = _params_for(tool, args)
            if params is None:
                messages.append({"role": "assistant", "content": last_text or "(invalid args)"})
                messages.append({"role": "user", "content":
                                 "工具 %s 参数无效：%s。请检查参数后重试。" % (name, json.dumps(args, ensure_ascii=False)[:600])})
                continue
            observation = None
            try:
                result = await tool.execute("call-%d" % (turn + 1), params)
                observation = _clip(_content_text(result), 4000)
                self.tool_calls += 1
            except Exception as exc:  # tool failures become observations, not crashes
                observation = _clip("工具执行失败：%s: %s" % (type(exc).__name__, exc), 2000)
            messages.append({"role": "assistant", "content": last_text or "(tool call)"})
            if name == "finish_round":
                break
            messages.append({"role": "user", "content":
                             "工具 %s 返回：\n%s\n请继续下一步。" % (name, observation)})
            if sum(len(str(item.get("content", ""))) for item in messages) > 60_000:
                error = "context budget exhausted before the model finished"
                break
        return {"ok": error is None, "text": last_text, "error": error,
                "tool_calls": self.tool_calls}


def _receipt_usage(receipt) -> dict:
    response = getattr(receipt, "response", None)
    usage = None
    if isinstance(response, dict):
        usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if usage is None:
        usage = getattr(receipt, "usage", None)
    if isinstance(usage, dict):
        return {"input": int(usage.get("prompt_tokens", usage.get("input_tokens", usage.get("input", 0))) or 0),
                "output": int(usage.get("completion_tokens", usage.get("output_tokens", usage.get("output", 0))) or 0),
                "reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0)}
    if usage is None:
        return {}
    return {"input": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output": int(getattr(usage, "completion_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(usage, "reasoning_tokens", 0) or 0)}


def _native_tool_calls(message):
    calls = getattr(message, "tool_calls", None)
    if calls is None and isinstance(message, dict):
        calls = message.get("tool_calls")
    if not calls:
        return None
    name, raw = None, {}
    first = calls[0]
    if isinstance(first, dict):
        function = first.get("function") or {}
        name = first.get("name") or function.get("name")
        raw = function.get("arguments") or first.get("arguments") or {}
    else:
        function = getattr(first, "function", None)
        name = getattr(first, "name", None) or getattr(function, "name", None)
        raw = getattr(function, "arguments", None) or getattr(first, "arguments", None) or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    return [(str(name), raw)] if name else None


# ---------------------------------------------------------------------------
# state bundle
# ---------------------------------------------------------------------------
@dataclass
class AgentState:
    dir: Path | None = None
    memory: object | None = None
    skills: object | None = None
    recall: object | None = None
    knowledge: object | None = None
    db: object | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"dir": str(self.dir or ""), "memory": self.memory is not None,
                "skills": self.skills is not None, "recall": self.recall is not None,
                "knowledge": self.knowledge is not None, "db": self.db is not None,
                "errors": self.errors}


def _copy_tree(source: Path, destination: Path):
    if not source.is_dir():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def open_state(work: Path, rnd) -> AgentState:
    """Restore the agent's cross-round memory/skills/knowledge from ``/state``."""
    state_dir = plans.new_state_dir(Path(work), rnd)
    state = AgentState(dir=state_dir)
    try:
        from rsi.memory.store import MemoryStore
        state.memory = MemoryStore(state_dir / "memory")
    except Exception as exc:
        state.errors.append("memory: %s" % exc)
    try:
        from rsi.memory.db import StateDB
        state.db = StateDB(state_dir / "rsi_state.db")
    except Exception as exc:
        state.errors.append("db: %s" % exc)
    try:
        from rsi.memory.skills import SkillStore
        skills_root = state_dir / "skills"
        if not skills_root.is_dir():
            _copy_tree(HARNESS_ROOT / "skills", skills_root)
        state.skills = SkillStore(skills_root,
                                  on_use=(state.db.skill_used if state.db else None))
    except Exception as exc:
        state.errors.append("skills: %s" % exc)
    try:
        from rsi.memory.recall import Recall
        state.recall = Recall(state.db, state.memory, state.skills)
    except Exception as exc:
        state.errors.append("recall: %s" % exc)
    try:
        from rsi.knowledge import default_knowledge
        knowledge_root = state_dir / "knowledge"
        if not knowledge_root.is_dir():
            _copy_tree(HARNESS_ROOT / "knowledge", knowledge_root)
        state.knowledge = default_knowledge(root=knowledge_root)
    except Exception as exc:
        state.errors.append("knowledge: %s" % exc)
    return state


# ---------------------------------------------------------------------------
# kernel host
# ---------------------------------------------------------------------------
class PlatformKernelHost:
    """KernelHost backed by the platform round (data/previous/model only)."""

    def __init__(self, ctx: plans.PlanContext, brief: dict, *, run_id: str = "",
                 history: list[dict] | None = None):
        self.ctx = ctx
        self.brief = dict(brief or {})
        self.run_id = run_id
        self.history = list(history or [])
        self.results: list[dict] = []
        self.plans: dict[str, dict] = {}
        self.panels: list[dict] = []
        self.seq = 0

    def context(self) -> dict:
        payload = dict(self.brief)
        payload.setdefault("task", self.ctx.task_key)
        payload.setdefault("primary_metric", self.ctx.metric)
        payload.setdefault("direction", self.ctx.direction)
        payload.setdefault("parent_primary", self.ctx.parent_primary)
        payload.setdefault("history", self.history[-10:])
        payload.setdefault("catalog", plans.catalog_text())
        return payload

    def parent_primary(self) -> float:
        return float(self.ctx.parent_primary)

    def run_experiment(self, proposal) -> dict:
        self.seq += 1
        experiment_id = "exp-%04d" % self.seq
        plan = plans.plan_for_proposal(self.ctx, proposal)
        plan["experiment_id"] = experiment_id
        if plan.get("ok", True):
            self.plans[experiment_id] = plan
        record = {
            "ok": bool(plan.get("ok", True)),
            "experiment_id": experiment_id,
            "strategy": plan.get("strategy"),
            "postprocess": (plan.get("postprocess") or {}).get("type"),
            "label": plan.get("label"),
            "metrics": plan.get("metrics") or {},
            "primary": plan.get("primary"),
            "hypothesis": plan.get("hypothesis"),
            "error": plan.get("error"),
        }
        self.results.append(record)
        return record

    def plan_for(self, experiment_id: str | None):
        if not experiment_id:
            return None
        return self.plans.get(str(experiment_id))

    def record_panel(self, entries) -> dict:
        panel_id = "panel-%04d" % (len(self.panels) + 1)
        self.panels.append({"panel_id": panel_id, "entries": list(entries or [])})
        return {"ok": True, "panel_id": panel_id}


# ---------------------------------------------------------------------------
# one round
# ---------------------------------------------------------------------------
@dataclass
class KernelRound:
    ok: bool = False
    call_ids: list[str] = field(default_factory=list)
    calls: int = 0
    decision: dict | None = None
    finished: dict | None = None
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    error: str | None = None
    host: PlatformKernelHost | None = None
    outcome: object | None = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "calls": self.calls, "call_ids": self.call_ids,
                "decision": self.decision, "finished": self.finished,
                "usage": self.usage, "error": self.error,
                "trace": self.trace[-20:]}


def _task_config(ctx: plans.PlanContext) -> TaskConfig:
    return TaskConfig(
        key=ctx.task_key,
        name=ctx.task_key,
        name_en=ctx.task_key,
        task=ctx.task,
        package="official",
        classes=tuple(ctx.classes),
        primary_metric=ctx.metric,
        direction=ctx.direction,
        baseline_primary=float(ctx.parent_primary),
        margin=float(ctx.margin),
    )


def run_kernel_round(rnd, ctx: plans.PlanContext, brief: dict, state: AgentState, *,
                     run_id: str = "", round_index: int = 1,
                     experiments_total: int = 0, max_turns: int = 6) -> KernelRound:
    """Run one PI-agent kernel round against the host gateway."""
    try:
        cap = int(os.environ.get("RSI_V2_TOKEN_CAP", "0") or 0)
    except (TypeError, ValueError):
        cap = 0
    runtime = GatewayRuntime(rnd, max_turns=max_turns, token_cap=cap)
    task_cfg = _task_config(ctx)
    try:
        per_round = int(os.environ.get("RSI_V2_MAX_EXPERIMENTS_ROUND", "3") or 3)
    except (TypeError, ValueError):
        per_round = 3
    try:
        per_task = int(os.environ.get("RSI_V2_MAX_EXPERIMENTS_TASK", "40") or 40)
    except (TypeError, ValueError):
        per_task = 40
    guard = Guard(task_cfg, max_experiments_per_round=max(1, per_round),
                  max_experiments_per_task=max(1, per_task))
    host = PlatformKernelHost(ctx, brief, run_id=run_id,
                              history=list(brief.get("history") or []))
    kernel = Kernel(task_cfg, runtime, guard,
                    memory=state.memory, skills=state.skills,
                    recall=state.recall, knowledge=state.knowledge,
                    run_id=run_id, max_turns=max_turns, verbose=False)
    try:
        outcome = kernel.run_round(host, brief, int(round_index), int(experiments_total))
    except Exception as exc:  # the round must never crash the submission
        return KernelRound(ok=False, call_ids=runtime.call_ids, calls=runtime.calls,
                           usage=runtime.budget.to_dict(),
                           error="%s: %s" % (type(exc).__name__, exc), host=host)
    decision = getattr(outcome, "decision", None)
    return KernelRound(
        ok=bool(getattr(outcome, "ok", False) and decision),
        call_ids=list(runtime.call_ids),
        calls=runtime.calls,
        decision=decision,
        finished=getattr(outcome, "finished", None),
        trace=list(getattr(outcome, "trace", []) or []),
        usage=runtime.budget.to_dict(),
        error=getattr(outcome, "error", None),
        host=host,
        outcome=outcome,
    )
