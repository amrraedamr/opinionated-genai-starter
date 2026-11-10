"""Spending a budget against a key, wherever the counters happen to live.

This is deliberately not "count HTTP requests". A request, a websocket message
and an LLM call's tokens are all spends against a budget, differing only in
what one unit means, so they share one mechanism and one set of counters.

Fleet-wide limits are often said to belong at an ingress. An ingress sees
addresses and paths; it does not see tenants, users, or how many tokens a reply
cost. Those are the budgets worth holding, so they are held here.
"""

from abc import ABC, abstractmethod
from typing import final

from base.models.definitions.rate_limiting import RateLimitPolicy
from base.models.view_models.rate_limiting import RateDecision
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsRateLimitConfig

WINDOW_SECONDS: float = 60.0
"""Every budget is expressed per minute, so every window is a minute long."""


class AbstractRateLimiter(ABC):
    """One budget per policy per key, counted somewhere."""

    def __init__(self, config: SupportsRateLimitConfig | None = None) -> None:
        self._config: SupportsRateLimitConfig = config or settings_for(
            SupportsRateLimitConfig
        )

    @final
    def budget_for(self, policy: RateLimitPolicy) -> int:
        """Units allowed per minute for `policy`."""
        budgets = {
            RateLimitPolicy.REQUESTS: self._config.rate_limit_requests_per_minute,
            RateLimitPolicy.AUTH: self._config.rate_limit_auth_per_minute,
            RateLimitPolicy.LLM_TOKENS: self._config.rate_limit_llm_tokens_per_minute,
            RateLimitPolicy.WS_CONNECT: self._config.rate_limit_ws_connect_per_minute,
            RateLimitPolicy.WS_MESSAGES: self._config.rate_limit_ws_messages_per_minute,
        }
        return budgets[policy]

    @staticmethod
    @final
    def compose_key(
        tenant: str | None = None,
        user: str | None = None,
        client_ip: str | None = None,
    ) -> str:
        """The narrowest identity available: tenant, then user, then address.

        An address is the weakest of the three — shared behind a NAT and cheap
        to change — so it is only used when nothing better is known.
        """
        return tenant or user or client_ip or "unknown"

    @abstractmethod
    def _spend(self, key: str, policy: RateLimitPolicy, cost: int) -> RateDecision: ...

    @final
    def spend(self, key: str, policy: RateLimitPolicy, cost: int = 1) -> RateDecision:
        """Spend `cost` units of `policy` against `key`.

        A cost of zero asks what is left without spending anything.
        """
        if cost < 0:
            raise ValueError("cost cannot be negative")
        return self._spend(key, policy, cost)

    @abstractmethod
    async def _aspend(
        self, key: str, policy: RateLimitPolicy, cost: int
    ) -> RateDecision: ...

    @final
    async def aspend(
        self, key: str, policy: RateLimitPolicy, cost: int = 1
    ) -> RateDecision:
        """Spend `cost` units of `policy` against `key`, asynchronously.

        A cost of zero asks what is left without spending anything.
        """
        if cost < 0:
            raise ValueError("cost cannot be negative")
        return await self._aspend(key, policy, cost)

    @abstractmethod
    def _charge(self, key: str, policy: RateLimitPolicy, cost: int) -> RateDecision: ...

    @final
    def charge(self, key: str, policy: RateLimitPolicy, cost: int) -> RateDecision:
        """Record `cost` units already spent, even past the limit."""
        if cost < 0:
            raise ValueError("cost cannot be negative")
        return self._charge(key, policy, cost)

    @abstractmethod
    async def _acharge(
        self, key: str, policy: RateLimitPolicy, cost: int
    ) -> RateDecision: ...

    @final
    async def acharge(
        self, key: str, policy: RateLimitPolicy, cost: int
    ) -> RateDecision:
        """Record `cost` units already spent, even past the limit, asynchronously."""
        if cost < 0:
            raise ValueError("cost cannot be negative")
        return await self._acharge(key, policy, cost)

    @abstractmethod
    def _reset(self, key: str, policy: RateLimitPolicy) -> None: ...

    @final
    def reset(self, key: str, policy: RateLimitPolicy) -> None:
        """Forget what `key` has spent on `policy`."""
        self._reset(key, policy)

    @abstractmethod
    async def _areset(self, key: str, policy: RateLimitPolicy) -> None: ...

    @final
    async def areset(self, key: str, policy: RateLimitPolicy) -> None:
        """Forget what `key` has spent on `policy`, asynchronously."""
        await self._areset(key, policy)

    def _counter_key(self, key: str, policy: RateLimitPolicy) -> str:
        """Each policy counts separately, so each gets its own counter."""
        return f"ratelimit:{policy.value}:{key}"


__all__ = ["WINDOW_SECONDS", "AbstractRateLimiter"]
