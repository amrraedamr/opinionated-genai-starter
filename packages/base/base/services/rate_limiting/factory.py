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
    """Construct the limiter this service is configured for.

    Each implementation imports lazily, so a service counting in its own
    process needs no `redis` extra.

    Dispatch is exhaustive rather than defaulting: an unhandled backend raises
    instead of quietly returning the in-process one, which would start cleanly,
    report healthy, and hand every replica its own full budget.

    A service that never composed `RateLimitSettingsMixin` gets that mixin's
    defaults rather than an error. `create_app` installs the traffic middleware
    for everyone, and the budgets are the same figures the middleware used
    before it had a limiter behind it — so this is the documented default, not
    a silent absence of limits. Tuning them means composing the mixin.
    """
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
    """The composed budgets, or the mixin's own defaults."""
    try:
        return settings_for(SupportsRateLimitConfig)
    except MissingSettingsError:
        return RateLimitSettingsMixin()


__all__ = ["build_rate_limiter"]
