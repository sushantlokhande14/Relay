"""Failover behaviour. Runs standalone: python tests/test_failover.py"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relay.providers.base import ProviderError
from relay.providers.failover import FailoverChain
from relay.providers.mock_provider import MockProvider
from relay.schemas import ChatCompletionRequest


def _req() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="mock", messages=[{"role": "user", "content": "hi"}])


async def collect(chain: FailoverChain):
    text = ""
    async for chunk in chain.stream(_req()):
        text += chunk.delta
    return text


async def test_falls_through_to_healthy_provider():
    chain = FailoverChain([MockProvider("flaky", fail_rate=1.0), MockProvider("mock")])
    text = await collect(chain)
    assert text, "expected the healthy provider to answer"
    print("PASS: failover reaches the healthy provider")


async def test_all_failing_raises():
    chain = FailoverChain([MockProvider("a", fail_rate=1.0), MockProvider("b", fail_rate=1.0)])
    try:
        await collect(chain)
    except ProviderError as e:
        assert "all providers failed" in str(e)
        print("PASS: chain raises when every provider fails")
        return
    raise AssertionError("expected ProviderError when all providers fail")


async def test_timeout_triggers_failover():
    # A provider that stalls past the timeout should be abandoned for the next.
    slow = MockProvider("slow", ttft_ms=5000)
    slow.timeout_s = 0.2
    chain = FailoverChain([slow, MockProvider("fast")])
    text = await collect(chain)
    assert text, "expected the fast provider to answer after the slow one timed out"
    print("PASS: a stalled provider times out and failover continues")


async def main():
    await test_falls_through_to_healthy_provider()
    await test_all_failing_raises()
    await test_timeout_triggers_failover()
    print("all failover tests passed")


if __name__ == "__main__":
    asyncio.run(main())
