"""What the Redis limiter does when Redis is not there: each replica limits on its own."""

from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from base.models.definitions.rate_limiting import RateLimitPolicy
from base.services.configuration.mixins import (
    RateLimitSettingsMixin,
    RedisSettingsMixin,
)
from base.services.rate_limiting.redis import RedisRateLimiter
from tests.conftest import MakeSettings


@pytest.fixture
def limiter(make_settings: MakeSettings) -> RedisRateLimiter:
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


def test_a_spend_is_counted_in_process_when_redis_is_unreachable(
    limiter: RedisRateLimiter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limiter, "_script", unreachable)
    decision = limiter.spend("acme", RateLimitPolicy.AUTH)
    assert decision.allowed
    assert decision.remaining == 6


async def test_a_spend_is_counted_in_process_when_redis_is_unreachable_async(
    limiter: RedisRateLimiter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limiter, "_ascript", aunreachable)
    decision = await limiter.aspend("acme", RateLimitPolicy.AUTH)
    assert decision.allowed
    assert decision.remaining == 6


def test_the_floor_still_refuses_while_redis_is_down(
    limiter: RedisRateLimiter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failing open used to mean no limit at all, for as long as Redis was gone."""
    monkeypatch.setattr(limiter, "_script", unreachable)
    for _ in range(7):
        assert limiter.spend("acme", RateLimitPolicy.AUTH).allowed
    assert not limiter.spend("acme", RateLimitPolicy.AUTH).allowed


def test_a_failed_redis_is_not_asked_again_straight_away(
    limiter: RedisRateLimiter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each request would otherwise wait out the socket timeout on a dead Redis."""
    calls: list[object] = []

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return unreachable()

    monkeypatch.setattr(limiter, "_script", counting)
    for _ in range(5):
        limiter.spend("acme", RateLimitPolicy.AUTH)
    assert len(calls) == 1


def test_redis_is_asked_again_once_the_breaker_closes(
    limiter: RedisRateLimiter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limiter, "_script", unreachable)
    limiter.spend("acme", RateLimitPolicy.AUTH)
    monkeypatch.setattr(limiter, "_script", lambda **_: [1, 3, 0])
    limiter._open_until = 0.0
    assert limiter.spend("acme", RateLimitPolicy.AUTH).remaining == 3


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
