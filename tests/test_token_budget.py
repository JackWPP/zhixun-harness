from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pi_agent_core.types import StreamOptions

from rsi.agent.config import LLMConfig, TokenBudget
from rsi.agent.runtime import PIRuntime, _estimate_context_tokens


def test_request_budget_reservation_bounds_completion_and_settles_usage():
    budget = TokenBudget(cap=1000)
    first = budget.reserve_request(700, 500)
    assert first is not None
    assert first[1] == 300
    assert budget.remaining() == 0
    assert budget.reserve_request(1, 1) is None

    budget.settle_request(first[0], {"input": 700, "output": 200,
                                     "reasoning_tokens": 25})
    assert budget.used == 900
    assert budget.remaining() == 100
    assert budget.used_reasoning == 25


def test_concurrent_reservations_share_one_cap():
    budget = TokenBudget(cap=100)
    results = []

    def reserve():
        results.append(budget.reserve_request(20, 40))

    threads = [__import__("threading").Thread(target=reserve) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(1 for row in results if row is not None) == 2
    assert budget.reserved == 100
    assert budget.reserve_request(1, 1) is None


def test_runtime_request_hook_reserves_before_provider_and_caps_output():
    runtime = PIRuntime(LLMConfig(enabled=True, token_cap=1000, max_response_tokens=500))
    runtime._checked = True
    runtime._model = object()
    observed = {}

    class FakeAgent:
        error_message = None
        messages = []

        async def prompt(self, _):
            context = SimpleNamespace(messages=[{"role": "user", "content": "x" * 100}],
                                      tools=[])
            observed["agent"].before_llm_call(context, None)
            request = await observed["stream"](
                None, None, StreamOptions())
            message = SimpleNamespace(usage=SimpleNamespace(input=500, output=300,
                                                              reasoningTokens=10))
            await observed["agent"].after_llm_call(context, message)

        async def wait_for_idle(self):
            return None

    async def fake_stream(model, context, options=None):
        observed["options"] = options
        return "ok"

    def fake_agent_ctor(**kwargs):
        agent = FakeAgent()
        agent.before_llm_call = kwargs["before_llm_call"]
        agent.after_llm_call = kwargs["after_llm_call"]
        observed["agent"] = agent
        observed["stream"] = kwargs["stream_fn"]
        # Patch the imported PI Agent constructor locally for this test.
        return agent

    # Exercise the hook logic directly to avoid importing provider code/network.
    import pi_agent_core
    original_agent = pi_agent_core.Agent
    original_stream = pi_agent_core.langchain_stream
    pi_agent_core.Agent = fake_agent_ctor
    pi_agent_core.langchain_stream = fake_stream
    try:
        async def execute():
            return await runtime.arun_agent("sys", "user", [])
        agent = runtime.run_coro(execute())
    finally:
        pi_agent_core.Agent = original_agent
        pi_agent_core.langchain_stream = original_stream
    assert agent.error_message is None
    assert observed["options"].max_tokens <= 500
    assert runtime.budget.used == 800
    assert runtime.budget.remaining() == 200


def test_context_estimate_counts_tool_and_message_payloads():
    context = SimpleNamespace(messages=[{"content": "中文" * 100}], tools=[{"name": "x" * 50}])
    assert _estimate_context_tokens(context) >= len(("中文" * 100).encode("utf-8"))
