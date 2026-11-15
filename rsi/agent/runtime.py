"""PI (pi-agent-core-lc) runtime adapter.

Only this module imports the PI package. If it is missing or the endpoint is
unreachable, ``available()`` is False and the coordinator silently falls back to
the deterministic planner; the harness therefore still completes a full run in
the offline evaluation image.
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

    # -- single role call -------------------------------------------------
    def call(self, role: str, system_prompt: str, user_prompt: str,
             toolset_kind: str, context: dict) -> RoleResult:
        result = RoleResult(role=role)
        start = time.time()
        sink = ToolSink()
        if not self.available():
            result.error = self._error or "LLM runtime unavailable"
            result.latency = time.time() - start
            return result
        if self.budget.exhausted():
            result.error = "token budget exhausted"
            result.latency = time.time() - start
            return result

        tools = build_toolset(toolset_kind, sink, context)

        async def _run():
            from pi_agent_core import Agent, langchain_stream

            agent = Agent(
                initial_state={"system_prompt": system_prompt,
                               "model": self._model,
                               "tools": tools},
                stream_fn=langchain_stream,
                max_turns=self.config.max_turns,
                tool_timeout=self.config.tool_timeout,
                max_retries=1,
                before_llm_call=self._before_llm_call,
            )
            await agent.prompt(user_prompt)
            await agent.wait_for_idle()
            return agent

        try:
            loop = self._event_loop()
            agent = loop.run_until_complete(
                asyncio.wait_for(_run(), timeout=self.config.timeout_seconds))
            result.ok = getattr(agent, "error_message", None) is None
            result.error = getattr(agent, "error_message", None)
            usage = {"input": 0, "output": 0, "reasoning_tokens": 0}
            texts: list[str] = []
            for message in agent.messages:
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
            result.text = "\n".join(t for t in texts if t).strip()[-1500:]
            self.budget.add(usage)
        except asyncio.TimeoutError:
            result.error = f"timeout after {self.config.timeout_seconds:.0f}s"
            self.budget.errors += 1
        except Exception as exc:  # pragma: no cover - network/runtime failure
            result.error = f"{exc.__class__.__name__}: {exc}"
            self.budget.errors += 1
        finally:
            self.calls += 1
            result.submitted = list(sink.payloads)
            result.latency = time.time() - start
            self.budget.calls += 1

        if not result.submitted and result.text and result.ok:
            parsed = _parse_json_object(result.text)
            if parsed is not None:
                parsed.setdefault("kind", "freeform")
                result.submitted = [parsed]
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
