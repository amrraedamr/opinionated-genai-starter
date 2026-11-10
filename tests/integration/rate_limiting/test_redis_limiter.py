"""The rate limiter against a real Redis, where the point is that it is shared.

An in-process limiter is correct for one replica and useless for ten: each
process holds its own counters, so ten replicas grant ten times the budget.
The whole reason this backend exists is that the counter lives somewhere both
replicas can see, and that is what these prove — by building a genuinely second
limiter object and showing it inherits what the first already spent.
"""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from base.clients.redis_client import RedisClient
from base.models.definitions.rate_limiting import (
    RATE_LIMIT_WINDOW_SECONDS,
    RateLimitPolicy,
)
from base.services.common.meta import Singleton
from base.services.rate_limiting.redis import RedisRateLimiter
from tests.conftest import CoreSettings

pytestmark = pytest.mark.integration

KEY = "integration-tenant"


def another_replica() -> RedisRateLimiter:
    """A second limiter object, as a second process would build.

    Dropping the cached singleton is what makes this a different object rather
    than the same one handed back, which is the only way the test can tell a
    shared counter from a local one.
    """
    Singleton._instances.pop(RedisRateLimiter, None)
    return RedisRateLimiter()


@pytest.fixture
async def limiter(infra_settings: CoreSettings) -> AsyncIterator[RedisRateLimiter]:
    del infra_settings
    instance = another_replica()
    for policy in RateLimitPolicy:
        await instance.areset(KEY, policy)
    try:
        yield instance
    finally:
        for policy in RateLimitPolicy:
            await instance.areset(KEY, policy)
        await RedisClient().close_all()


async def test_a_spend_inside_the_budget_is_allowed(
    limiter: RedisRateLimiter,
) -> None:
    decision = await limiter.aspend(KEY, RateLimitPolicy.AUTH)
    assert decision.allowed
    assert decision.remaining >= 0


async def test_the_budget_runs_out(limiter: RedisRateLimiter) -> None:
    budget = limiter.budget_for(RateLimitPolicy.AUTH)
    for _ in range(budget):
        assert (await limiter.aspend(KEY, RateLimitPolicy.AUTH)).allowed
    refused = await limiter.aspend(KEY, RateLimitPolicy.AUTH)
    assert not refused.allowed
    assert refused.retry_after_seconds > 0


async def test_two_replicas_share_one_budget(limiter: RedisRateLimiter) -> None:
    """The reason this backend exists: ten replicas must not mean ten budgets."""
    budget = limiter.budget_for(RateLimitPolicy.AUTH)
    for _ in range(budget):
        assert (await limiter.aspend(KEY, RateLimitPolicy.AUTH)).allowed

    second = another_replica()
    assert second is not limiter, "the test is not proving anything"
    refused = await second.aspend(KEY, RateLimitPolicy.AUTH)
    assert not refused.allowed, "a second replica was handed a fresh budget"


async def test_a_second_replica_sees_what_the_first_spent(
    limiter: RedisRateLimiter,
) -> None:
    """Not only the refusal — the remaining count carries across too."""
    await limiter.aspend(KEY, RateLimitPolicy.AUTH)
    first = (await limiter.aspend(KEY, RateLimitPolicy.AUTH, cost=0)).remaining

    second = another_replica()
    assert (await second.aspend(KEY, RateLimitPolicy.AUTH, cost=0)).remaining == first


async def test_policies_do_not_share_a_counter(limiter: RedisRateLimiter) -> None:
    budget = limiter.budget_for(RateLimitPolicy.AUTH)
    for _ in range(budget):
        await limiter.aspend(KEY, RateLimitPolicy.AUTH)
    assert not (await limiter.aspend(KEY, RateLimitPolicy.AUTH)).allowed
    assert (await limiter.aspend(KEY, RateLimitPolicy.REQUESTS)).allowed


async def test_keys_do_not_share_a_counter(limiter: RedisRateLimiter) -> None:
    budget = limiter.budget_for(RateLimitPolicy.AUTH)
    for _ in range(budget):
        await limiter.aspend(KEY, RateLimitPolicy.AUTH)
    assert not (await limiter.aspend(KEY, RateLimitPolicy.AUTH)).allowed
    try:
        assert (await limiter.aspend("someone-else", RateLimitPolicy.AUTH)).allowed
    finally:
        await limiter.areset("someone-else", RateLimitPolicy.AUTH)


async def test_a_costly_spend_consumes_proportionally(
    limiter: RedisRateLimiter,
) -> None:
    """One call spending many tokens must cost the same as many small ones."""
    budget = limiter.budget_for(RateLimitPolicy.LLM_TOKENS)
    bulk = await limiter.aspend(KEY, RateLimitPolicy.LLM_TOKENS, cost=budget)
    assert bulk.allowed
    assert not (await limiter.aspend(KEY, RateLimitPolicy.LLM_TOKENS)).allowed


async def test_a_zero_cost_asks_without_spending(
    limiter: RedisRateLimiter,
) -> None:
    await limiter.aspend(KEY, RateLimitPolicy.AUTH)
    before = (await limiter.aspend(KEY, RateLimitPolicy.AUTH, cost=0)).remaining
    after = (await limiter.aspend(KEY, RateLimitPolicy.AUTH, cost=0)).remaining
    assert before == after


async def test_a_refusal_fits_inside_the_window(
    limiter: RedisRateLimiter,
) -> None:
    """A retry longer than the window would park a client for no reason."""
    budget = limiter.budget_for(RateLimitPolicy.AUTH)
    for _ in range(budget):
        await limiter.aspend(KEY, RateLimitPolicy.AUTH)
    refused = await limiter.aspend(KEY, RateLimitPolicy.AUTH)
    assert 0 < refused.retry_after_seconds <= RATE_LIMIT_WINDOW_SECONDS


async def test_a_reset_returns_the_whole_budget(
    limiter: RedisRateLimiter,
) -> None:
    budget = limiter.budget_for(RateLimitPolicy.AUTH)
    for _ in range(budget):
        await limiter.aspend(KEY, RateLimitPolicy.AUTH)
    assert not (await limiter.aspend(KEY, RateLimitPolicy.AUTH)).allowed

    await limiter.areset(KEY, RateLimitPolicy.AUTH)
    assert (await limiter.aspend(KEY, RateLimitPolicy.AUTH)).allowed


async def test_the_counter_expires_rather_than_accumulating(
    limiter: RedisRateLimiter,
) -> None:
    """An idle key must not be kept forever; Redis is told to drop it."""
    await limiter.aspend(KEY, RateLimitPolicy.AUTH)
    counter = limiter._counter_key(KEY, RateLimitPolicy.AUTH)
    ttl = await RedisClient().async_client.pttl(counter)
    assert ttl > 0


async def test_a_charge_past_the_limit_is_recorded(limiter: RedisRateLimiter) -> None:
    """A spend would refuse this; a charge records work already done."""
    key = f"charge-{uuid4()}"
    await limiter.acharge(key, RateLimitPolicy.AUTH, 500)

    decision = await limiter.aspend(key, RateLimitPolicy.AUTH, cost=0)
    assert decision.remaining == 0
    assert decision.retry_after_seconds > 0
