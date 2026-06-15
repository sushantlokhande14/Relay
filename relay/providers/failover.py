from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..schemas import ChatCompletionRequest
from .base import Provider, ProviderError, StreamChunk


async def _iter_with_timeout(agen: AsyncIterator[StreamChunk], timeout: float):
    # Apply an idle timeout to each step, which also covers time to first token.
    # On timeout we close the underlying generator before raising.
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(agen.__anext__(), timeout)
            except StopAsyncIteration:
                return
            yield chunk
    finally:
        await agen.aclose()


class FailoverChain:
    """Tries providers in order. If one errors or stalls before sending any
    output, move to the next. Once a provider has streamed a token to the
    client there's no safe way to switch, so a mid-stream failure stops there."""

    def __init__(self, providers: list[Provider], default_timeout_s: float = 30):
        self.providers = providers
        self.default_timeout_s = default_timeout_s

    async def stream(self, req: ChatCompletionRequest):
        errors: list[str] = []
        for provider in self.providers:
            timeout = getattr(provider, "timeout_s", self.default_timeout_s)
            started = False
            agen = provider.stream(req)
            try:
                async for chunk in _iter_with_timeout(agen, timeout):
                    started = True
                    chunk.provider = provider.name
                    yield chunk
                return
            except Exception as e:
                errors.append(f"{provider.name}: {type(e).__name__}: {e}")
                if started:
                    raise ProviderError(
                        f"{provider.name} failed after streaming had started: {e}"
                    ) from e
                # nothing sent yet, fall through to the next provider
        raise ProviderError("all providers failed -> " + " | ".join(errors))
