"""LLM runtime configuration, loaded from environment variables only.

The endpoint/key never appear in code, logs or the image: the evaluation image
is started with ``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL`` set (PyTRiO's
OpenAI-compatible endpoint works unchanged). ``RSI_LLM_MODE=off`` forces the
deterministic fallback policy, which is also the default when no key is set.
"""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field


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
    max_response_tokens: int = 8192
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
        max_response_tokens=max(1, _int("RSI_MAX_RESPONSE_TOKENS", 8192)),
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
    reserved: int = 0
    _reservations: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, usage: dict) -> None:
        with self._lock:
            self.calls += 1
            self.used_input += int(usage.get("input", 0))
            self.used_output += int(usage.get("output", 0))
            self.used_reasoning += int(usage.get("reasoning_tokens", 0))

    @property
    def used(self) -> int:
        return self.used_input + self.used_output

    def remaining(self) -> int:
        if self.cap <= 0:
            return 2**63 - 1
        return max(0, self.cap - self.used - self.reserved)

    def exhausted(self) -> bool:
        return self.cap > 0 and self.used + self.reserved >= self.cap

    def reserve_request(self, estimated_input: int, max_output: int) -> tuple[str, int] | None:
        """Reserve a bounded model response before sending it.

        The input estimate is intentionally conservative (serialized UTF-8
        bytes, including tool schemas and prior messages). ``max_output`` is
        reduced to the remaining allowance, so concurrent delegate calls share
        one atomic run-level budget instead of each observing the same balance.
        A cap of zero means unmetered development mode, but responses are still
        capped by ``max_output``.
        """
        estimate = max(0, int(estimated_input))
        requested_output = max(1, int(max_output))
        with self._lock:
            remaining = (2**63 - 1 if self.cap <= 0
                         else max(0, self.cap - self.used - self.reserved))
            output = min(requested_output, max(0, remaining - estimate))
            if output < 1:
                return None
            reservation = estimate + output
            key = uuid.uuid4().hex
            self._reservations[key] = reservation
            self.reserved += reservation
            return key, output

    def settle_request(self, reservation_id: str, usage: dict) -> None:
        """Replace a reservation with provider-reported usage."""
        with self._lock:
            reserved = self._reservations.pop(reservation_id, 0)
            self.reserved = max(0, self.reserved - reserved)
            input_tokens = max(0, int(usage.get("input", 0) or 0))
            output_tokens = max(0, int(usage.get("output", 0) or 0))
            # Missing usage must not silently make a call free. Consume the
            # conservative reservation and flag it in the report.
            if input_tokens + output_tokens == 0 and reserved:
                input_tokens = reserved
                self.errors += 1
            self.calls += 1
            self.used_input += input_tokens
            self.used_output += output_tokens
            self.used_reasoning += max(0, int(usage.get("reasoning_tokens", 0) or 0))

    def release_request(self, reservation_id: str) -> None:
        """Release a request that failed before a billable response arrived."""
        with self._lock:
            reserved = self._reservations.pop(reservation_id, 0)
            self.reserved = max(0, self.reserved - reserved)

    def to_dict(self) -> dict:
        return {"tokens_used": self.used, "tokens_in": self.used_input,
                "tokens_out": self.used_output, "reasoning_tokens": self.used_reasoning,
                "calls": self.calls, "errors": self.errors, "tokens_cap": self.cap,
                "remaining": self.remaining(), "tokens_reserved": self.reserved}
