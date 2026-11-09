"""What the Redis limiter does when Redis is not there.

A limiter that refuses everything when its counter store blinks turns a
degraded dependency into an outage, so this one allows the spend instead. That
is a deliberate trade and the kind of thing that quietly regresses, so it is
pinned here rather than left to a docstring.

No Redis is needed: the script call is replaced with one that raises.
"""

from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from base.models.definitions.rate_limiting import RateLimitPolicy
from base.services.configuration.mixins import (
    RateLimitSettingsMixin,
    RedisSettingsMixin,
)
from base.services.rate_limiting.redis import RedisRateLimiter


@pytest.fixture
def limiter(make_settings: Any) -> RedisRateLimiter:
    """Pointed at an address nothing answers on, deliberately.

    Every test here must pass with no Redis anywhere, which is also what proves
    that building the limiter does not open a connection.
    """
    make_settings(
        RedisSettingsMixin,
        RateLimitSettingsMixin,
        redis_url="redis://127.0.0.1:1/0",
        rate_limit_auth_per_minute=7,
    )
    return RedisRateLimiter()


def unreachable(*args: Any, **kwargs: Any) -> Any:
    raise RedisConnectionError("no route to host")


async def aunreachable(*args: Any, **kwargs: Any) -> Any:
    raise RedisConnectionError("no route to host")


def test_building_the_limiter_does_not_reach_redis(
    limiter: RedisRateLimiter,
) -> None:
    """The fixture points nowhere, so reaching this line is the assertion.

    If construction connected, failing open could never happen — the service
    would die at startup instead of degrading.
    """
    assert limiter.budget_for(RateLimitPolicy.AUTH) == 7


def test_a_spend_is_allowed_when_redis_is_unreachable_for_real(
    limiter: RedisRateLimiter,
) -> None:
    """No stub at all: the address is dead and the spend still goes through."""
    assert limiter.spend("acme", RateLimitPolicy.AUTH).allowed


def test_a_spend_is_allowed_when_redis_is_unreachable(
    limiter: RedisRateLimiter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limiter, "_script", unreachable)
    decision = limiter.spend("acme", RateLimitPolicy.AUTH)
    assert decision.allowed
    assert decision.remaining == 7
    assert decision.retry_after_seconds == 0.0


async def test_a_spend_is_allowed_when_redis_is_unreachable_async(
    limiter: RedisRateLimiter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limiter, "_ascript", aunreachable)
    decision = await limiter.aspend("acme", RateLimitPolicy.AUTH)
    assert decision.allowed
    assert decision.remaining == 7


def test_a_reset_survives_an_unreachable_redis(
    limiter: RedisRateLimiter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing a counter that cannot be reached is not worth an exception."""
    monkeypatch.setattr(
        limiter._redis_client.sync_client, "delete", unreachable, raising=False
    )
    limiter.reset("acme", RateLimitPolicy.AUTH)


async def test_a_reset_survives_an_unreachable_redis_async(
    limiter: RedisRateLimiter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        limiter._redis_client.async_client, "delete", aunreachable, raising=False
    )
    await limiter.areset("acme", RateLimitPolicy.AUTH)


def test_a_lua_reply_becomes_a_decision(
    limiter: RedisRateLimiter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lua returns integers only, so the retry arrives in milliseconds."""
    monkeypatch.setattr(limiter, "_script", lambda **_: [0, 3, 2500])
    decision = limiter.spend("acme", RateLimitPolicy.AUTH)
    assert not decision.allowed
    assert decision.remaining == 3
    assert decision.retry_after_seconds == 2.5


def test_the_script_asks_redis_for_the_clock() -> None:
    """Replicas with skewed clocks must still agree about one budget."""
    from base.services.rate_limiting.redis import GCRA_SCRIPT

    assert "redis.call('TIME')" in GCRA_SCRIPT


def test_the_script_sets_an_expiry() -> None:
    """A counter with no expiry is a key retained for every caller ever seen."""
    from base.services.rate_limiting.redis import GCRA_SCRIPT

    assert "'PX'" in GCRA_SCRIPT
