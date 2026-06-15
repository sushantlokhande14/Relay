from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

from .cache.embed import Embedder
from .cache.keys import (
    conversation_text,
    normalize_request,
    params_json,
    prompt_hash,
)
from .cache.semantic import SemanticCache
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
    model: str = ""
    cache: str = "miss"  # "exact" | "semantic" | "miss"
    provider: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    similarity: float | None = None


class Gateway:
    def __init__(
        self,
        settings: Settings,
        providers: dict[str, Provider],
        store: Store,
        embedder: Embedder | None = None,
        semantic: SemanticCache | None = None,
    ):
        self.settings = settings
        self.providers = providers
        self.store = store
        self.embedder = embedder
        self.semantic = semantic
        self.cache_enabled = settings.cache.enabled
        self._chains: dict[str, FailoverChain] = {}
        for name, route in settings.routes.items():
            members = [providers[p] for p in route.providers if p in providers]
            self._chains[name] = FailoverChain(members)

    def route_for(self, req: ChatCompletionRequest) -> str:
        return req.route or self.settings.default_route

    def _threshold(self, route_name: str) -> float:
        return self.settings.routes[route_name].similarity_threshold

    async def stream(
        self, req: ChatCompletionRequest, ctx: RequestContext
    ) -> AsyncIterator[StreamChunk]:
        route_name = self.route_for(req)
        ctx.route = route_name
        ctx.model = req.model
        normalized = normalize_request(req, route_name)
        h = prompt_hash(normalized)

        embedding = None
        if self.cache_enabled:
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

            # 2. semantic cache: embed the prompt, ask proxima for the nearest one
            if self.embedder is not None and self.semantic is not None:
                embedding = self.embedder.embed_one(conversation_text(req))
                row, similarity = self.semantic.search(
                    embedding, route_name, self._threshold(route_name)
                )
                if row is not None:
                    self.store.touch(row["id"])
                    ctx.cache = "semantic"
                    ctx.similarity = similarity
                    ctx.prompt_tokens = row["prompt_tokens"]
                    ctx.completion_tokens = row["completion_tokens"]
                    yield StreamChunk(delta=row["response_text"])
                    yield StreamChunk(usage=Usage(row["prompt_tokens"], row["completion_tokens"]))
                    return

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

        if not self.cache_enabled:
            return

        # On a miss the embedding may not exist yet (e.g. no message text changed
        # the route). Compute it here so the entry is searchable next time.
        if self.embedder is not None and embedding is None:
            embedding = self.embedder.embed_one(conversation_text(req))

        row_id = self.store.insert(
            Entry(
                prompt_hash=h,
                route=route_name,
                prompt_text=conversation_text(req),
                response_text=content,
                model=req.model,
                params_json=params_json(req),
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                embedding=embedding,
            )
        )
        if self.semantic is not None and embedding is not None:
            self.semantic.add(row_id, embedding)
