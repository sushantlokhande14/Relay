from __future__ import annotations

import asyncio
import random
from typing import AsyncIterator

from ..schemas import ChatCompletionRequest
from .base import Provider, ProviderError, StreamChunk, Usage


def estimate_tokens(text: str) -> int:
    # Rough count, ~4 chars per token. Fine for a demo and for the mock's own
    # bookkeeping; the real providers report exact usage.
    return max(1, len(text) // 4)


class MockProvider(Provider):
    """A local provider that needs no keys. It streams a deterministic reply
    built from the prompt, with configurable latency so the load test sees
    something realistic."""

    def __init__(
        self,
        name: str = "mock",
        ttft_ms: float = 120,
        per_token_ms: float = 8,
        reply_tokens: int = 40,
        fail_rate: float = 0.0,
    ):
        self.name = name
        self.ttft_ms = ttft_ms
        self.per_token_ms = per_token_ms
        self.reply_tokens = reply_tokens
        self.fail_rate = fail_rate

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        if self.fail_rate and random.random() < self.fail_rate:
            raise ProviderError(f"{self.name}: injected failure")

        prompt = req.messages[-1].content if req.messages else ""
        prompt_tokens = sum(estimate_tokens(m.content) for m in req.messages)

        await asyncio.sleep(self.ttft_ms / 1000)
        completion_tokens = 0
        for word in self._reply_words(prompt):
            await asyncio.sleep(self.per_token_ms / 1000)
            yield StreamChunk(delta=word)
            completion_tokens += 1

        yield StreamChunk(usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))

    def _reply_words(self, prompt: str) -> list[str]:
        head = prompt.strip().replace("\n", " ")
        if len(head) > 60:
            head = head[:60] + "..."
        opener = f'You asked: "{head}". ' if head else ""
        filler = "This is a mock reply for local testing. It runs without API keys. "
        text = opener + filler * 4
        words = text.split()[: self.reply_tokens]
        return [w + " " for w in words]
