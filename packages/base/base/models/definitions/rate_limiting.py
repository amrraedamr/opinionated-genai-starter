"""What a budget is spent against, and where the counting happens."""

from enum import StrEnum
from typing import Final

RATE_LIMIT_WINDOW_SECONDS: Final = 60.0

REDIS_BREAKER_SECONDS: Final = 5.0
"""How long the Redis limiter stops asking a Redis that just failed, using the in-memory floor."""

MEMORY_SWEEP_SECONDS: Final = 1.0
"""How often the in-memory limiter drops idle keys, rather than on every spend."""


class RateLimitPolicy(StrEnum):
    """One budget. Each has its own allowance and its own counter per key."""

    REQUESTS = "requests"
    """Ordinary HTTP traffic."""

    AUTH = "auth"
    """Sign-in and token endpoints, held far tighter than ordinary traffic."""

    LLM_TOKENS = "llm_tokens"
    WS_CONNECT = "ws_connect"
    WS_MESSAGES = "ws_messages"


class RateLimitBackend(StrEnum):
    """Where the counters live."""

    MEMORY = "memory"
    """Per replica. Correct for one process, and a floor for any number."""

    REDIS = "redis"
