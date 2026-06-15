from __future__ import annotations

from typing import AsyncIterator

from .config import Settings
from .providers.base import Provider, StreamChunk
from .schemas import ChatCompletionRequest


class Gateway:
    """The request pipeline. Right now it just picks a route and calls the
    first provider in its chain. Caching, failover, rate limiting and metrics
    get layered in here over the next milestones."""

    def __init__(self, settings: Settings, providers: dict[str, Provider]):
        self.settings = settings
        self.providers = providers

    def route_for(self, req: ChatCompletionRequest) -> str:
        return req.route or self.settings.default_route

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        route = self.settings.routes[self.route_for(req)]
        provider = self.providers[route.providers[0]]
        async for chunk in provider.stream(req):
            yield chunk
