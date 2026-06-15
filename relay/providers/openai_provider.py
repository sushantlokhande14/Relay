from __future__ import annotations

from typing import AsyncIterator

from ..schemas import ChatCompletionRequest
from .base import Provider, ProviderError, StreamChunk, Usage


class OpenAIProvider(Provider):
    """Talks to OpenAI's chat completions API and streams deltas back in our
    own StreamChunk shape, so the gateway never sees provider-specific types."""

    def __init__(self, name: str, api_key: str, model: str, timeout_s: float = 30):
        from openai import AsyncOpenAI

        self.name = name
        self.model = model
        self.timeout_s = timeout_s
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_s)

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for event in stream:
                if event.usage:
                    yield StreamChunk(
                        usage=Usage(event.usage.prompt_tokens, event.usage.completion_tokens)
                    )
                if event.choices:
                    delta = event.choices[0].delta.content
                    if delta:
                        yield StreamChunk(delta=delta)
        except Exception as e:  # network, auth, rate limit, timeout
            raise ProviderError(f"{self.name}: {e}") from e
