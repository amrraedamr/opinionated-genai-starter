"""Spending a budget against a key, wherever the counters happen to live."""

from abc import ABC, abstractmethod
from typing import final

from base.models.definitions.rate_limiting import RateLimitPolicy
from base.models.view_models.rate_limiting import RateDecision
from base.services.configuration import settings_for
from base.services.configuration.protocols import SupportsRateLimitConfig


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
        """Key per user within a tenant, else the tenant, the user, the client IP or "unknown"."""
        if tenant and user:
            return f"{tenant}:{user}"
        return tenant or user or client_ip or "unknown"

    @abstractmethod
    def _spend(self, key: str, policy: RateLimitPolicy, cost: int) -> RateDecision: ...

    @final
    def spend(self, key: str, policy: RateLimitPolicy, cost: int = 1) -> RateDecision:
        """Spend `cost` units of `policy` against `key` if they fit (a zero cost only checks)."""
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
        """Spend `cost` units of `policy` against `key` if they fit (a zero cost only checks), asynchronously."""
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


__all__ = ["AbstractRateLimiter"]
