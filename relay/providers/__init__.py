from __future__ import annotations

from ..config import Settings
from .base import Provider
from .mock_provider import MockProvider


def build_providers(settings: Settings) -> dict[str, Provider]:
    """Build the provider instances named in config. Unknown or unbuildable
    providers (no SDK, no key) are skipped so the gateway still boots on mock."""
    out: dict[str, Provider] = {}
    for name, pc in settings.providers.items():
        if pc.type == "mock":
            out[name] = MockProvider(
                name=name,
                ttft_ms=pc.ttft_ms,
                per_token_ms=pc.per_token_ms,
                fail_rate=pc.fail_rate,
            )
        else:
            # Real providers (openai, anthropic) are wired up in the next
            # milestone. Skip quietly for now.
            continue
    return out
