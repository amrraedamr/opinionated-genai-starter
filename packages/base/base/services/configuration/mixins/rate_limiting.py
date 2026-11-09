from pydantic import BaseModel

from base.models.definitions.rate_limiting import RateLimitBackend


class RateLimitSettingsMixin(BaseModel):
    """Budgets per policy, and where they are counted."""

    rate_limit_backend: RateLimitBackend = RateLimitBackend.MEMORY
    rate_limit_requests_per_minute: int = 300
    rate_limit_auth_per_minute: int = 20
    rate_limit_llm_tokens_per_minute: int = 100_000
    rate_limit_ws_connect_per_minute: int = 60
    rate_limit_ws_messages_per_minute: int = 600
