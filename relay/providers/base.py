from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from ..schemas import ChatCompletionRequest


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class StreamChunk:
    # a text delta, plus usage that providers attach to their final chunk.
    # provider is stamped by the failover chain so the gateway knows who answered.
    delta: str = ""
    usage: Usage | None = None
    provider: str | None = None


class ProviderError(Exception):
    """Raised when a provider call fails. The failover chain catches this."""


class Provider(ABC):
    name: str

    @abstractmethod
    def stream(self, req: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        ...
