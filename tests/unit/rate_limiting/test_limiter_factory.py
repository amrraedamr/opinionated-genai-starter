"""Choosing a limiter, and what happens when nobody said which.

The fallback matters more than it looks: `create_app` installs the traffic
middleware for every service, including ones that never composed the rate-limit
mixin. Dispatch is exhaustive on purpose — quietly returning the in-process
limiter would start cleanly, report healthy, and hand every replica its own
full budget.
"""

from typing import Any

import pytest

from base.models.definitions.rate_limiting import RateLimitBackend, RateLimitPolicy
from base.services.configuration.mixins import (
    RateLimitSettingsMixin,
    RedisSettingsMixin,
    ServerSettingsMixin,
)
from base.services.rate_limiting.factory import build_rate_limiter
from base.services.rate_limiting.memory import InMemoryRateLimiter


def test_memory_is_the_default_backend(make_settings: Any) -> None:
    make_settings(RateLimitSettingsMixin)
    assert isinstance(build_rate_limiter(), InMemoryRateLimiter)


def test_redis_is_built_when_asked(make_settings: Any) -> None:
    make_settings(
        RateLimitSettingsMixin,
        RedisSettingsMixin,
        rate_limit_backend=RateLimitBackend.REDIS,
        redis_url="redis://127.0.0.1:1/0",
    )
    from base.services.rate_limiting.redis import RedisRateLimiter

    assert isinstance(build_rate_limiter(), RedisRateLimiter)


def test_configured_budgets_reach_the_limiter(make_settings: Any) -> None:
    make_settings(RateLimitSettingsMixin, rate_limit_requests_per_minute=7)
    assert build_rate_limiter().budget_for(RateLimitPolicy.REQUESTS) == 7


def test_a_service_that_never_composed_the_mixin_gets_the_defaults(
    make_settings: Any,
) -> None:
    """`create_app` installs the middleware for everyone, composed or not."""
    make_settings(ServerSettingsMixin)
    limiter = build_rate_limiter()
    assert limiter.budget_for(RateLimitPolicy.REQUESTS) == 300
    assert limiter.budget_for(RateLimitPolicy.AUTH) == 20


def test_the_fallback_is_the_mixin_not_an_absence_of_limits(
    make_settings: Any,
) -> None:
    """The defaults must be the mixin's, so one place defines a budget."""
    make_settings(ServerSettingsMixin)
    defaults = RateLimitSettingsMixin()
    limiter = build_rate_limiter()
    for policy in RateLimitPolicy:
        assert limiter.budget_for(policy) == getattr(
            defaults, f"rate_limit_{policy.value}_per_minute"
        )


def test_an_explicit_config_wins_over_settings(make_settings: Any) -> None:
    make_settings(RateLimitSettingsMixin, rate_limit_requests_per_minute=7)
    override = RateLimitSettingsMixin(rate_limit_requests_per_minute=99)
    assert build_rate_limiter(override).budget_for(RateLimitPolicy.REQUESTS) == 99


def test_an_unknown_backend_is_refused_rather_than_guessed(
    make_settings: Any,
) -> None:
    """Defaulting here would hand every replica its own full budget."""
    make_settings(RateLimitSettingsMixin)

    class Unknown:
        rate_limit_backend = "carrier-pigeon"

    with pytest.raises(ValueError, match="carrier-pigeon"):
        build_rate_limiter(Unknown())  # type: ignore[arg-type]
