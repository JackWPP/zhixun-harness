"""LLM runtime configuration, loaded from environment variables only.

The endpoint/key never appear in code, logs or the image: the evaluation image
is started with ``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL`` set (PyTRiO's
OpenAI-compatible endpoint works unchanged). ``RSI_LLM_MODE=off`` forces the
deterministic fallback policy, which is also the default when no key is set.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class LLMConfig:
    enabled: bool
    provider: str = "openai"
    base_url: str = ""
    api_key: str = ""
    model: str = "qwen/qwen-3.8-27b"
    context_window: int = 64000
    timeout_seconds: float = 240.0
    max_turns: int = 6
    tool_timeout: float = 60.0
    token_cap: int = 2_000_000
    temperature: float | None = None

    def masked(self) -> dict:
        return {"provider": self.provider, "base_url": self.base_url, "model": self.model,
                "context_window": self.context_window, "enabled": self.enabled,
                "token_cap": self.token_cap}


def load_llm_config(mode: str = "auto") -> LLMConfig:
    mode = (mode or "auto").lower()
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL", "qwen/qwen-3.8-27b").strip()
    if mode == "off":
        enabled = False
    elif mode == "on":
        if not base_url or not api_key:
            raise SystemExit("LLM mode 'on' requires LLM_BASE_URL and LLM_API_KEY")
        enabled = True
    else:
        enabled = bool(base_url and api_key)
    config = LLMConfig(
        enabled=enabled,
        provider=os.environ.get("LLM_PROVIDER", "openai").strip() or "openai",
        base_url=base_url,
        api_key=api_key,
        model=model,
        context_window=_int("LLM_CONTEXT_WINDOW", 64000),
        timeout_seconds=_float("LLM_TIMEOUT_SECONDS", 240.0),
        max_turns=_int("LLM_MAX_TURNS", 6),
        tool_timeout=_float("LLM_TOOL_TIMEOUT", 60.0),
        token_cap=_int("RSI_TOKEN_CAP", 2_000_000),
    )
    if enabled:
        # langchain-openai (the PI StreamFn adapter) reads this variable.
        os.environ.setdefault("OPENAI_API_KEY", config.api_key)
        os.environ.setdefault("OPENAI_BASE_URL", config.base_url)
    return config


@dataclass
class TokenBudget:
    cap: int = 2_000_000
    used_input: int = 0
    used_output: int = 0
    used_reasoning: int = 0
    calls: int = 0
    errors: int = 0

    def add(self, usage: dict) -> None:
        self.calls += 1
        self.used_input += int(usage.get("input", 0))
        self.used_output += int(usage.get("output", 0))
        self.used_reasoning += int(usage.get("reasoning_tokens", 0))

    @property
    def used(self) -> int:
        return self.used_input + self.used_output

    def remaining(self) -> int:
        # cap <= 0 means "unlimited" (development / rehearsal mode); the
        # competition default keeps the platform 8M budget.
        if self.cap <= 0:
            return 10 ** 18
        return max(0, self.cap - self.used)

    def exhausted(self) -> bool:
        return self.cap > 0 and self.used >= self.cap

    def to_dict(self) -> dict:
        return {"tokens_used": self.used, "tokens_in": self.used_input,
                "tokens_out": self.used_output, "reasoning_tokens": self.used_reasoning,
                "calls": self.calls, "errors": self.errors, "tokens_cap": self.cap,
                "remaining": self.remaining()}
