"""The rate limiter's contract, proved against the in-process backend.

A budget is spent against a key, and what one unit means depends on the policy:
one HTTP request, one websocket message, one token. The rules that matter are
that policies and keys never share a counter, that a refusal says when to come
back, and that the window actually slides.

Time is injected rather than slept through, so the sliding window is tested at
exact boundaries instead of approximately.
"""

import pytest

from base.models.definitions.rate_limiting import RateLimitPolicy
from base.models.view_models.rate_limiting import RateDecision
from base.services.configuration.mixins import RateLimitSettingsMixin
from base.services.rate_limiting import WINDOW_SECONDS, AbstractRateLimiter
from base.services.rate_limiting.memory import InMemoryRateLimiter
from tests.conftest import MakeSettings


class Clock:
    """A monotonic clock a test can move on purpose."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    instance = Clock()
    monkeypatch.setattr("base.services.rate_limiting.memory.time.monotonic", instance)
    return instance


@pytest.fixture
def limiter(make_settings: MakeSettings) -> InMemoryRateLimiter:
    make_settings(
        RateLimitSettingsMixin,
        rate_limit_requests_per_minute=5,
        rate_limit_auth_per_minute=2,
        rate_limit_llm_tokens_per_minute=100,
    )
    return InMemoryRateLimiter()


def test_a_spend_inside_the_budget_is_allowed(limiter: InMemoryRateLimiter) -> None:
    decision = limiter.spend("acme", RateLimitPolicy.REQUESTS)
    assert decision.allowed
    assert decision.remaining == 4
    assert decision.retry_after_seconds == 0.0


def test_the_budget_runs_out(limiter: InMemoryRateLimiter) -> None:
    for _ in range(5):
        assert limiter.spend("acme", RateLimitPolicy.REQUESTS).allowed
    refused = limiter.spend("acme", RateLimitPolicy.REQUESTS)
    assert not refused.allowed
    assert refused.remaining == 0


def test_a_refusal_says_when_to_come_back(
    limiter: InMemoryRateLimiter, clock: Clock
) -> None:
    """The retry points at the moment the oldest hit ages out, not a constant."""
    for _ in range(5):
        limiter.spend("acme", RateLimitPolicy.REQUESTS)
    clock.advance(20.0)
    refused = limiter.spend("acme", RateLimitPolicy.REQUESTS)
    assert not refused.allowed
    assert refused.retry_after_seconds == pytest.approx(WINDOW_SECONDS - 20.0)


def test_the_window_slides(limiter: InMemoryRateLimiter, clock: Clock) -> None:
    for _ in range(5):
        limiter.spend("acme", RateLimitPolicy.REQUESTS)
    assert not limiter.spend("acme", RateLimitPolicy.REQUESTS).allowed

    clock.advance(WINDOW_SECONDS + 0.001)
    assert limiter.spend("acme", RateLimitPolicy.REQUESTS).allowed


def test_only_what_aged_out_comes_back(
    limiter: InMemoryRateLimiter, clock: Clock
) -> None:
    """A window is not a bucket that empties all at once."""
    for _ in range(5):
        limiter.spend("acme", RateLimitPolicy.REQUESTS)
        clock.advance(1.0)

    clock.advance(WINDOW_SECONDS - 5.0)
    assert limiter.spend("acme", RateLimitPolicy.REQUESTS).allowed
    assert not limiter.spend("acme", RateLimitPolicy.REQUESTS).allowed


def test_policies_do_not_share_a_counter(limiter: InMemoryRateLimiter) -> None:
    for _ in range(2):
        limiter.spend("acme", RateLimitPolicy.AUTH)
    assert not limiter.spend("acme", RateLimitPolicy.AUTH).allowed
    assert limiter.spend("acme", RateLimitPolicy.REQUESTS).allowed


def test_keys_do_not_share_a_counter(limiter: InMemoryRateLimiter) -> None:
    for _ in range(5):
        limiter.spend("acme", RateLimitPolicy.REQUESTS)
    assert not limiter.spend("acme", RateLimitPolicy.REQUESTS).allowed
    assert limiter.spend("globex", RateLimitPolicy.REQUESTS).allowed


def test_a_spend_can_cost_more_than_one(limiter: InMemoryRateLimiter) -> None:
    """An LLM call spends tokens, not requests."""
    decision = limiter.spend("acme", RateLimitPolicy.LLM_TOKENS, cost=60)
    assert decision.allowed
    assert decision.remaining == 40
    assert not limiter.spend("acme", RateLimitPolicy.LLM_TOKENS, cost=41).allowed


def test_a_cost_larger_than_the_whole_budget_never_fits(
    limiter: InMemoryRateLimiter,
) -> None:
    """Reported as a full window rather than reading past the end of the window."""
    refused = limiter.spend("acme", RateLimitPolicy.LLM_TOKENS, cost=500)
    assert not refused.allowed
    assert refused.retry_after_seconds == WINDOW_SECONDS


def test_a_zero_cost_asks_without_spending(limiter: InMemoryRateLimiter) -> None:
    limiter.spend("acme", RateLimitPolicy.REQUESTS)
    asked = limiter.spend("acme", RateLimitPolicy.REQUESTS, cost=0)
    assert asked.allowed
    assert asked.remaining == 4
    assert limiter.spend("acme", RateLimitPolicy.REQUESTS, cost=0).remaining == 4


def test_a_negative_cost_is_refused_outright(limiter: InMemoryRateLimiter) -> None:
    """Otherwise a caller could refund itself budget it never spent."""
    with pytest.raises(ValueError, match="negative"):
        limiter.spend("acme", RateLimitPolicy.REQUESTS, cost=-1)


async def test_a_negative_cost_is_refused_outright_async(
    limiter: InMemoryRateLimiter,
) -> None:
    with pytest.raises(ValueError, match="negative"):
        await limiter.aspend("acme", RateLimitPolicy.REQUESTS, cost=-1)


async def test_both_counterparts_share_one_budget(
    limiter: InMemoryRateLimiter,
) -> None:
    """The async path is the same counters, not a second set."""
    for _ in range(5):
        await limiter.aspend("acme", RateLimitPolicy.REQUESTS)
    assert not limiter.spend("acme", RateLimitPolicy.REQUESTS).allowed


def test_a_reset_forgets_one_policy_only(limiter: InMemoryRateLimiter) -> None:
    for _ in range(5):
        limiter.spend("acme", RateLimitPolicy.REQUESTS)
    for _ in range(2):
        limiter.spend("acme", RateLimitPolicy.AUTH)

    limiter.reset("acme", RateLimitPolicy.REQUESTS)
    assert limiter.spend("acme", RateLimitPolicy.REQUESTS).allowed
    assert not limiter.spend("acme", RateLimitPolicy.AUTH).allowed


async def test_a_reset_forgets_one_policy_only_async(
    limiter: InMemoryRateLimiter,
) -> None:
    for _ in range(5):
        await limiter.aspend("acme", RateLimitPolicy.REQUESTS)
    await limiter.areset("acme", RateLimitPolicy.REQUESTS)
    assert (await limiter.aspend("acme", RateLimitPolicy.REQUESTS)).allowed


def test_idle_keys_are_forgotten(limiter: InMemoryRateLimiter, clock: Clock) -> None:
    """Without eviction a key is retained for every address ever seen."""
    for index in range(50):
        limiter.spend(f"client-{index}", RateLimitPolicy.REQUESTS)
    assert len(limiter._windows) == 50

    clock.advance(WINDOW_SECONDS + 1.0)
    limiter.spend("someone-else", RateLimitPolicy.REQUESTS)
    assert len(limiter._windows) == 1


def test_every_policy_has_a_budget(limiter: InMemoryRateLimiter) -> None:
    """A policy with no budget would raise at the first spend against it."""
    for policy in RateLimitPolicy:
        assert limiter.budget_for(policy) > 0


def test_the_narrowest_identity_wins() -> None:
    compose = AbstractRateLimiter.compose_key
    assert compose(tenant="acme", user="u1", client_ip="1.2.3.4") == "acme"
    assert compose(user="u1", client_ip="1.2.3.4") == "u1"
    assert compose(client_ip="1.2.3.4") == "1.2.3.4"
    assert compose() == "unknown"


def test_an_empty_identity_falls_through() -> None:
    """A blank tenant is no tenant, not a key everyone shares."""
    assert AbstractRateLimiter.compose_key(tenant="", user="u1") == "u1"


def test_a_refused_decision_never_asks_for_a_zero_second_wait() -> None:
    """`Retry-After: 0` tells a client to retry immediately, which is the bug."""
    decision = RateDecision(allowed=False, remaining=0, retry_after_seconds=0.2)
    assert decision.retry_after_header == "1"


def test_a_retry_header_rounds_up() -> None:
    decision = RateDecision(allowed=False, remaining=0, retry_after_seconds=4.1)
    assert decision.retry_after_header == "5"


def test_remaining_cannot_go_negative() -> None:
    with pytest.raises(ValueError):
        RateDecision(allowed=True, remaining=-1)
