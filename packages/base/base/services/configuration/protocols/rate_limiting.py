from typing import Protocol, runtime_checkable

from base.models.definitions.rate_limiting import RateLimitBackend


@runtime_checkable
class SupportsRateLimitConfig(Protocol):
    """Satisfied by `RateLimitSettingsMixin`."""

    rate_limit_backend: RateLimitBackend
    rate_limit_requests_per_minute: int
    rate_limit_auth_per_minute: int
    rate_limit_llm_tokens_per_minute: int
    rate_limit_ws_connect_per_minute: int
    rate_limit_ws_messages_per_minute: int
