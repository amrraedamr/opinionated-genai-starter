"""Backend selection, shared by everything that spends against a budget."""

from base.models.definitions.rate_limiting import RateLimitBackend
from base.services.configuration import settings_for
from base.services.configuration.mixins.rate_limiting import RateLimitSettingsMixin
from base.services.configuration.protocols import SupportsRateLimitConfig
from base.services.rate_limiting import AbstractRateLimiter
from base.tools.errors import MissingSettingsError


def build_rate_limiter(
    config: SupportsRateLimitConfig | None = None,
) -> AbstractRateLimiter:
    """Construct the configured limiter, using `RateLimitSettingsMixin` defaults when the service composed none."""
    settings = config or _configured()
    backend = RateLimitBackend(settings.rate_limit_backend)

    match backend:
        case RateLimitBackend.REDIS:
            from base.services.rate_limiting.redis import RedisRateLimiter

            return RedisRateLimiter(settings)
        case RateLimitBackend.MEMORY:
            from base.services.rate_limiting.memory import InMemoryRateLimiter

            return InMemoryRateLimiter(settings)

    raise ValueError(f"No limiter implemented for backend {backend.value!r}")


def _configured() -> SupportsRateLimitConfig:
    """The configured budgets, or the mixin's defaults when settings are absent or lack the mixin."""
    try:
        return settings_for(SupportsRateLimitConfig)
    except (MissingSettingsError, RuntimeError):
        return RateLimitSettingsMixin()


__all__ = ["build_rate_limiter"]
