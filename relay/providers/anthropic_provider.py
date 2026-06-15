from __future__ import annotations

from typing import AsyncIterator

from ..schemas import ChatCompletionRequest
from .base import Provider, ProviderError, StreamChunk, Usage


class AnthropicProvider(Provider):
    """Same interface as the others, over Anthropic's Messages API. Anthropic
    keeps the system prompt separate and requires max_tokens, so we adapt the
    OpenAI-style request to fit."""

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        timeout_s: float = 30,
        default_max_tokens: int = 1024,
    ):
        from anthropic import AsyncAnthropic

        self.name = name
        self.model = model
        self.timeout_s = timeout_s
        self.default_max_tokens = default_max_tokens
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_s)

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        system = "\n".join(m.content for m in req.messages if m.role == "system")
        messages = [
            {"role": m.role, "content": m.content}
            for m in req.messages
            if m.role in ("user", "assistant")
        ]
        kwargs = {
            "model": self.model,
            "max_tokens": req.max_tokens or self.default_max_tokens,
            "messages": messages,
            "temperature": req.temperature,
        }
        if system:
            kwargs["system"] = system

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(delta=text)
                final = await stream.get_final_message()
                yield StreamChunk(
                    usage=Usage(final.usage.input_tokens, final.usage.output_tokens)
                )
        except Exception as e:
            raise ProviderError(f"{self.name}: {e}") from e
