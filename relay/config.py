from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class RateLimitConfig(BaseModel):
    capacity: float = 60
    refill_per_sec: float = 30


class RouteConfig(BaseModel):
    providers: list[str]
    similarity_threshold: float = 0.9
    rate_limit: RateLimitConfig | None = None


class ProviderConfig(BaseModel):
    type: str
    model: str | None = None
    timeout_s: float = 30
    ttft_ms: float = 0
    per_token_ms: float = 0
    fail_rate: float = 0.0

    model_config = {"extra": "allow"}


class PriceConfig(BaseModel):
    input: float = 0.0   # USD per 1M input tokens
    output: float = 0.0  # USD per 1M output tokens


class CacheConfig(BaseModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384
    db_path: str = "data/relay.db"
    index_path: str = "data/cache.idx"
    max_entries: int = 5000
    rebuild_when_stale_frac: float = 0.20


class Settings(BaseModel):
    default_route: str = "default"
    routes: dict[str, RouteConfig]
    providers: dict[str, ProviderConfig]
    cache: CacheConfig = CacheConfig()
    prices: dict[str, PriceConfig] = {}
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None


def _load_dotenv(path: str = ".env") -> None:
    # Tiny .env reader so I don't pull in python-dotenv. Only sets keys that
    # aren't already in the environment.
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def load_settings(path: str = "config.yaml") -> Settings:
    _load_dotenv()
    raw = yaml.safe_load(Path(path).read_text())
    gw = raw.get("gateway", {})
    return Settings(
        default_route=gw.get("default_route", "default"),
        routes=raw.get("routes", {}),
        providers=raw.get("providers", {}),
        cache=raw.get("cache", {}),
        prices=raw.get("prices", {}),
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
    )
