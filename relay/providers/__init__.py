from __future__ import annotations

from ..config import Settings
from .base import Provider
from .mock_provider import MockProvider


def build_providers(settings: Settings) -> dict[str, Provider]:
    """Build the provider instances named in config. A real provider with no
    API key is skipped, so the gateway still boots on mock alone."""
    out: dict[str, Provider] = {}
    for name, pc in settings.providers.items():
        if pc.type == "mock":
            out[name] = MockProvider(
                name=name,
                ttft_ms=pc.ttft_ms,
                per_token_ms=pc.per_token_ms,
                fail_rate=pc.fail_rate,
            )
        elif pc.type == "openai":
            if not settings.openai_api_key:
                continue
            from .openai_provider import OpenAIProvider

            out[name] = OpenAIProvider(
                name=name,
                api_key=settings.openai_api_key,
                model=pc.model or "gpt-4o-mini",
                timeout_s=pc.timeout_s,
            )
        elif pc.type == "anthropic":
            if not settings.anthropic_api_key:
                continue
            from .anthropic_provider import AnthropicProvider

            out[name] = AnthropicProvider(
                name=name,
                api_key=settings.anthropic_api_key,
                model=pc.model or "claude-3-5-haiku-latest",
                timeout_s=pc.timeout_s,
            )
    return out
