"""PI (pi-agent-core-lc) runtime adapter.

Only this module imports the PI package. If it is missing or the endpoint is
unreachable, ``available()`` is False and the coordinator silently falls back to
the deterministic planner; the harness therefore still completes a full run in
the offline evaluation image.

Two call styles are supported:

* ``call()``     — synchronous single-role call (coordinator / deterministic pipeline);
* ``arun_agent()`` — async agent builder used by the autonomous kernel, which runs
  inside one long-lived event loop and may call nested agents (delegate/panel).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from .config import LLMConfig, TokenBudget
from .tools import ToolSink, build_toolset


@dataclass
class RoleResult:
    role: str
    ok: bool = False
    text: str = ""
    submitted: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    error: str | None = None
    latency: float = 0.0
    tool_calls: int = 0

    def to_dict(self) -> dict:
        return {"role": self.role, "ok": self.ok, "latency": round(self.latency, 2),
                "tool_calls": self.tool_calls, "usage": self.usage,
                "error": self.error, "text": self.text[:800]}


def _block_value(block, key, default=None):
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


class PIRuntime:
    backend = "pi-agent-core-lc"

    def __init__(self, config: LLMConfig, budget: TokenBudget | None = None):
        self.config = config
        self.budget = budget or TokenBudget(cap=config.token_cap)
        self._model = None
        self._checked = False
        self._error: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.calls = 0

    # -- availability -----------------------------------------------------
    def available(self) -> bool:
        if not self.config.enabled:
            return False
        if self._checked:
            return self._error is None
        try:
            from pi_agent_core import Model  # noqa: F401
            from pi_agent_core.tools import SimpleTool  # noqa: F401

            self._model = Model(provider=self.config.provider, model_id=self.config.model,
                                base_url=self.config.base_url,
                                context_window=self.config.context_window)
            self._checked = True
        except Exception as exc:  # pragma: no cover - depends on optional package
            self._error = f"{exc.__class__.__name__}: {exc}"
            self._checked = True
        return self._error is None

    @property
    def model(self):
        if not self.available():
            raise RuntimeError(self._error or "LLM runtime unavailable")
        return self._model

    def _event_loop(self) -> asyncio.AbstractEventLoop:
        """One long-lived loop per runtime.

        langchain-openai/httpx cache async clients bound to the loop that first
        used them; creating a fresh ``asyncio.run`` loop per call makes every
        call after the first fail with ``Event loop is closed``.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def run_coro(self, coro, timeout: float | None = None):
        """Run a coroutine on the persistent loop (kernel entry point)."""
        loop = self._event_loop()
        if timeout:
            coro = asyncio.wait_for(coro, timeout=timeout)
        return loop.run_until_complete(coro)

    def close(self) -> None:
        loop, self._loop = self._loop, None
        if loop is not None and not loop.is_closed():
            try:
                loop.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    def status(self) -> dict:
        return {"backend": self.backend, "available": self.available(),
                "error": self._error, "model": self.config.model,
                "base_url": self.config.base_url, "calls": self.calls,
                "budget": self.budget.to_dict()}

    # -- hook -------------------------------------------------------------
    def _before_llm_call(self, current_context, last_budget):
        if self.budget.exhausted():
            raise RuntimeError("token budget exhausted")
        fraction = getattr(last_budget, "fraction", 0.0) if last_budget is not None else 0.0
        if fraction > 0.92:
            raise RuntimeError(f"context window nearly full ({fraction:.0%})")
        return None

    # -- async primitives (kernel) ---------------------------------------
    async def arun_agent(self, system_prompt: str, user_prompt: str, tools: list,
                         *, max_turns: int | None = None,
                         tool_timeout: float | None = None,
                         timeout_seconds: float | None = None):
        """Build and run one PI agent inside the current event loop.

        Must be awaited from inside :meth:`run_coro`; nested agents (delegate,
        panel) can be launched from tool handlers with ``asyncio.gather``.
        """
        from pi_agent_core import Agent, langchain_stream

        # The PI hook runs before every provider request, including later turns
        # and delegated agents. Reserve the serialized prompt plus a bounded
        # completion before making the request; after_llm_call replaces that
        # reservation with actual provider usage. A failed call keeps no
        # reservation once this agent exits.
        reservations: dict[asyncio.Task, tuple[str, int]] = {}

        def before_llm_call(current_context, last_budget):
            if self.budget.exhausted():
                raise RuntimeError("token budget exhausted")
            fraction = getattr(last_budget, "fraction", 0.0) if last_budget is not None else 0.0
            if fraction > 0.92:
                raise RuntimeError(f"context window nearly full ({fraction:.0%})")
            task = asyncio.current_task()
            if task is None:
                raise RuntimeError("cannot reserve token budget outside an asyncio task")
            estimate = _estimate_context_tokens(current_context)
            response_limit = max(1, int(self.config.max_response_tokens))
            reservation = self.budget.reserve_request(estimate, response_limit)
            if reservation is None:
                raise RuntimeError("token budget cannot reserve prompt and one response token")
            reservations[task] = reservation
            return None

        async def after_llm_call(current_context, message):
            del current_context
            task = asyncio.current_task()
            reservation = reservations.pop(task, None) if task is not None else None
            if reservation is not None:
                self.budget.settle_request(reservation[0], _usage_dict(_block_value(message, "usage")))

        async def bounded_stream(model, context, options=None):
            from dataclasses import replace
            from pi_agent_core.types import StreamOptions

            task = asyncio.current_task()
            reservation = reservations.get(task) if task is not None else None
            if reservation is None:
                raise RuntimeError("model request has no token-budget reservation")
            current = options or StreamOptions()
            current_limit = getattr(current, "max_tokens", None)
            cap = reservation[1]
            if current_limit is not None:
                cap = min(cap, int(current_limit))
            return await langchain_stream(model, context, replace(current, max_tokens=cap))

        agent = Agent(
            initial_state={"system_prompt": system_prompt,
                           "model": self.model,
                           "tools": tools},
            stream_fn=bounded_stream,
            max_turns=int(max_turns or self.config.max_turns),
            tool_timeout=float(tool_timeout or self.config.tool_timeout),
            max_retries=1,
            before_llm_call=before_llm_call,
            after_llm_call=after_llm_call,
        )
        prompt_coro = agent.prompt(user_prompt)
        if timeout_seconds:
            prompt_coro = asyncio.wait_for(prompt_coro, timeout=timeout_seconds)
        try:
            await prompt_coro
            await agent.wait_for_idle()
        finally:
            for reservation_id, _ in list(reservations.values()):
                # A provider may have generated billable tokens before a
                # transport error prevented usage metadata. Charge the full
                # reservation in that ambiguous case instead of making the
                # failed request free.
                self.budget.settle_request(reservation_id, {})
            reservations.clear()
        return agent

    def collect_agent(self, agent, *, role: str, sink: ToolSink | None = None,
                      start: float | None = None) -> RoleResult:
        result = RoleResult(role=role)
        start = start if start is not None else time.time()
        result.ok = getattr(agent, "error_message", None) is None
        result.error = getattr(agent, "error_message", None)
        usage = {"input": 0, "output": 0, "reasoning_tokens": 0}
        texts: list[str] = []
        for message in getattr(agent, "messages", []) or []:
            if getattr(message, "role", None) != "assistant":
                continue
            message_usage = getattr(message, "usage", None)
            if message_usage is not None:
                usage["input"] += int(getattr(message_usage, "input", 0) or 0)
                usage["output"] += int(getattr(message_usage, "output", 0) or 0)
                usage["reasoning_tokens"] += int(
                    getattr(message_usage, "reasoningTokens", 0) or 0)
            for block in getattr(message, "content", []) or []:
                if _block_value(block, "type") == "text":
                    texts.append(_block_value(block, "text", "") or "")
                elif _block_value(block, "type") == "toolCall":
                    result.tool_calls += 1
        result.usage = usage
        result.text = "\n".join(t for t in texts if t).strip()[-3000:]
        if sink is not None:
            result.submitted = list(sink.payloads)
        # Usage is charged in after_llm_call for each provider response, before
        # the next tool or model turn can start. Do not double-count it here.
        self.calls += 1
        result.latency = time.time() - start
        if not result.submitted and result.text and result.ok:
            parsed = _parse_json_object(result.text)
            if parsed is not None:
                parsed.setdefault("kind", "freeform")
                result.submitted = [parsed]
        return result

    def _failure(self, role: str, error: str, start: float) -> RoleResult:
        result = RoleResult(role=role, ok=False, error=error,
                            latency=time.time() - start)
        self.budget.errors += 1
        self.budget.calls += 1
        self.calls += 1
        return result

    # -- single role call (sync) ------------------------------------------
    def call(self, role: str, system_prompt: str, user_prompt: str,
             toolset_kind: str, context: dict) -> RoleResult:
        start = time.time()
        if not self.available():
            return self._failure(role, self._error or "LLM runtime unavailable", start)
        if self.budget.exhausted():
            return self._failure(role, "token budget exhausted", start)
        sink = ToolSink()
        tools = build_toolset(toolset_kind, sink, context)

        async def _run():
            return await self.arun_agent(
                system_prompt, user_prompt, tools,
                max_turns=self.config.max_turns,
                tool_timeout=self.config.tool_timeout,
                timeout_seconds=self.config.timeout_seconds)

        try:
            agent = self.run_coro(_run(), timeout=self.config.timeout_seconds + 30)
            result = self.collect_agent(agent, role=role, sink=sink, start=start)
        except asyncio.TimeoutError:
            return self._failure(role, f"timeout after {self.config.timeout_seconds:.0f}s", start)
        except Exception as exc:  # pragma: no cover - network/runtime failure
            return self._failure(role, f"{exc.__class__.__name__}: {exc}", start)
        return result


def _parse_json_object(text: str) -> dict | None:
    """Best-effort extraction of a single JSON object from free-form text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start:end + 1])
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _usage_dict(usage) -> dict:
    """Normalize PI/LangChain usage objects without depending on one version."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        get = usage.get
    else:
        get = lambda key, default=None: getattr(usage, key, default)
    return {
        "input": get("input", 0) or 0,
        "output": get("output", 0) or 0,
        "reasoning_tokens": get("reasoningTokens", get("reasoning_tokens", 0)) or 0,
    }


def _estimate_context_tokens(context) -> int:
    """Conservative input estimate for a hard per-request budget reservation.

    The PI context includes messages and tool schemas. Counting serialized UTF-8
    bytes is deliberately more conservative than a character-count heuristic;
    the provider's returned usage remains the audit value.
    """
    if hasattr(context, "model_dump"):
        value = context.model_dump(mode="json")
    elif hasattr(context, "__dict__"):
        value = vars(context)
    else:
        value = context
    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        serialized = repr(value)
    return len(serialized.encode("utf-8")) + 128
