from __future__ import annotations

from typing import AsyncIterator

from .config import Settings
from .providers.base import Provider, StreamChunk
from .providers.failover import FailoverChain
from .schemas import ChatCompletionRequest


class Gateway:
    """The request pipeline. Picks a route and runs its provider chain with
    failover. Caching, rate limiting and metrics get layered in here over the
    next milestones."""

    def __init__(self, settings: Settings, providers: dict[str, Provider]):
        self.settings = settings
        self.providers = providers
        self._chains: dict[str, FailoverChain] = {}
        for name, route in settings.routes.items():
            members = [providers[p] for p in route.providers if p in providers]
            self._chains[name] = FailoverChain(members)

    def route_for(self, req: ChatCompletionRequest) -> str:
        return req.route or self.settings.default_route

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        chain = self._chains[self.route_for(req)]
        async for chunk in chain.stream(req):
            yield chunk
