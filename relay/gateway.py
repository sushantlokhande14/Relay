from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from .cache.keys import (
    conversation_text,
    normalize_request,
    params_json,
    prompt_hash,
)
from .cache.store import Entry, Store
from .config import Settings
from .providers.base import Provider, StreamChunk, Usage
from .providers.failover import FailoverChain
from .schemas import ChatCompletionRequest


@dataclass
class RequestContext:
    """Filled in as a request flows through the pipeline. The app reads it after
    streaming to set headers and (later) record metrics."""

    route: str = ""
    cache: str = "miss"  # "exact" | "semantic" | "miss"
    provider: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    similarity: float | None = None


class Gateway:
    def __init__(self, settings: Settings, providers: dict[str, Provider], store: Store):
        self.settings = settings
        self.providers = providers
        self.store = store
        self._chains: dict[str, FailoverChain] = {}
        for name, route in settings.routes.items():
            members = [providers[p] for p in route.providers if p in providers]
            self._chains[name] = FailoverChain(members)

    def route_for(self, req: ChatCompletionRequest) -> str:
        return req.route or self.settings.default_route

    async def stream(
        self, req: ChatCompletionRequest, ctx: RequestContext
    ) -> AsyncIterator[StreamChunk]:
        route_name = self.route_for(req)
        ctx.route = route_name
        normalized = normalize_request(req, route_name)
        h = prompt_hash(normalized)

        # 1. exact cache
        row = self.store.get_by_hash(h, route_name)
        if row is not None:
            self.store.touch(row["id"])
            ctx.cache = "exact"
            ctx.prompt_tokens = row["prompt_tokens"]
            ctx.completion_tokens = row["completion_tokens"]
            yield StreamChunk(delta=row["response_text"])
            yield StreamChunk(usage=Usage(row["prompt_tokens"], row["completion_tokens"]))
            return

        # 2. semantic cache goes here (next milestone)

        # 3. miss -> provider chain, then store the result
        ctx.cache = "miss"
        chain = self._chains[route_name]
        content = ""
        usage: Usage | None = None
        async for chunk in chain.stream(req):
            if chunk.provider:
                ctx.provider = chunk.provider
            if chunk.delta:
                content += chunk.delta
            if chunk.usage:
                usage = chunk.usage
            yield chunk

        if usage:
            ctx.prompt_tokens = usage.prompt_tokens
            ctx.completion_tokens = usage.completion_tokens

        self.store.insert(
            Entry(
                prompt_hash=h,
                route=route_name,
                prompt_text=conversation_text(req),
                response_text=content,
                model=req.model,
                params_json=params_json(req),
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                embedding=None,
            )
        )
