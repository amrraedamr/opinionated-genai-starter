"""Counters in Redis, so a budget means the same thing across every replica."""

import time
from typing import ClassVar

from base.models.definitions.rate_limiting import (
    RATE_LIMIT_WINDOW_SECONDS,
    REDIS_BREAKER_SECONDS,
    RateLimitPolicy,
)
from base.models.view_models.rate_limiting import RateDecision
from base.services.common.meta import SingletonABC
from base.services.configuration.protocols import SupportsRateLimitConfig
from base.services.logging.console import ConsoleLoggerService
from base.services.rate_limiting import AbstractRateLimiter
from base.tools.errors import MissingExtraError

try:
    from redis.exceptions import RedisError
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("redis", "redis", feature="RedisRateLimiter") from exc

from base.clients.redis_client import RedisClient
from base.services.rate_limiting.memory import InMemoryRateLimiter

_GCRA_SCRIPT = """
local key    = KEYS[1]
local cost   = tonumber(ARGV[1])
local limit  = tonumber(ARGV[2])
local window = tonumber(ARGV[3])
local force  = tonumber(ARGV[4])

local clock = redis.call('TIME')
local now = tonumber(clock[1]) + (tonumber(clock[2]) / 1000000)

local emission = window / limit
local tolerance = window

local tat = tonumber(redis.call('GET', key)) or now
if tat < now then tat = now end

if cost == 0 then
  local free = math.floor((tolerance - (tat - now)) / emission)
  if free > 0 then return {1, free, 0} end
  return {1, 0, math.ceil((tat + emission - tolerance - now) * 1000)}
end

local new_tat = tat + (cost * emission)
local allow_at = new_tat - tolerance

if force == 0 and now < allow_at then
  local free = math.floor((tolerance - (tat - now)) / emission)
  if free < 0 then free = 0 end
  return {0, free, math.ceil((allow_at - now) * 1000)}
end

redis.call('SET', key, new_tat, 'PX', math.ceil((new_tat - now) * 1000) + 1000)
local free = math.floor((tolerance - (new_tat - now)) / emission)
if free < 0 then free = 0 end
return {1, free, 0}
"""
"""GCRA in one round trip.

Redis supplies the clock rather than the caller, so replicas with skewed clocks
still agree about one budget — which is the entire reason for counting here
instead of in each process.
"""


class RedisRateLimiter(AbstractRateLimiter, metaclass=SingletonABC):
    """One shared budget per key; while Redis is unreachable, each replica enforces its own."""

    SERVICE_NAME: ClassVar[str] = "Redis Rate Limiter"

    def __init__(self, config: SupportsRateLimitConfig | None = None) -> None:
        super().__init__(config)
        self._redis_client = RedisClient()
        self._floor = InMemoryRateLimiter(config)
        self._logger = ConsoleLoggerService()
        self._open_until = 0.0
        self._script = self._redis_client.sync_client.register_script(_GCRA_SCRIPT)
        self._ascript = self._redis_client.async_client.register_script(_GCRA_SCRIPT)

    def _spend(self, key: str, policy: RateLimitPolicy, cost: int) -> RateDecision:
        return self._run(key, policy, cost, force=False)

    async def _aspend(
        self, key: str, policy: RateLimitPolicy, cost: int
    ) -> RateDecision:
        return await self._arun(key, policy, cost, force=False)

    def _charge(self, key: str, policy: RateLimitPolicy, cost: int) -> RateDecision:
        return self._run(key, policy, cost, force=True)

    async def _acharge(
        self, key: str, policy: RateLimitPolicy, cost: int
    ) -> RateDecision:
        return await self._arun(key, policy, cost, force=True)

    def _run(
        self, key: str, policy: RateLimitPolicy, cost: int, *, force: bool
    ) -> RateDecision:
        if self._broken():
            return self._fallback(key, policy, cost, force=force)
        limit = self.budget_for(policy)
        try:
            raw = self._script(
                keys=[self._counter_key(key, policy)],
                args=[cost, limit, RATE_LIMIT_WINDOW_SECONDS, int(force)],
            )
        except RedisError as exc:
            self._trip(exc)
            return self._fallback(key, policy, cost, force=force)
        return self._decision(raw)

    async def _arun(
        self, key: str, policy: RateLimitPolicy, cost: int, *, force: bool
    ) -> RateDecision:
        if self._broken():
            return await self._afallback(key, policy, cost, force=force)
        limit = self.budget_for(policy)
        try:
            raw = await self._ascript(
                keys=[self._counter_key(key, policy)],
                args=[cost, limit, RATE_LIMIT_WINDOW_SECONDS, int(force)],
            )
        except RedisError as exc:
            self._trip(exc)
            return await self._afallback(key, policy, cost, force=force)
        return self._decision(raw)

    def _broken(self) -> bool:
        return time.monotonic() < self._open_until

    def _trip(self, exc: RedisError) -> None:
        """Stop asking Redis for a while, so each request does not wait out its timeout."""
        self._open_until = time.monotonic() + REDIS_BREAKER_SECONDS
        self._logger.warning(
            "Rate limiter cannot reach Redis, limiting per replica for %ss: %s",
            REDIS_BREAKER_SECONDS,
            exc,
        )

    def _fallback(
        self, key: str, policy: RateLimitPolicy, cost: int, *, force: bool
    ) -> RateDecision:
        if force:
            return self._floor.charge(key, policy, cost)
        return self._floor.spend(key, policy, cost)

    async def _afallback(
        self, key: str, policy: RateLimitPolicy, cost: int, *, force: bool
    ) -> RateDecision:
        if force:
            return await self._floor.acharge(key, policy, cost)
        return await self._floor.aspend(key, policy, cost)

    def _reset(self, key: str, policy: RateLimitPolicy) -> None:
        try:
            self._redis_client.sync_client.delete(self._counter_key(key, policy))
        except RedisError as exc:
            self._logger.warning("Rate limit reset failed: %s", exc)

    async def _areset(self, key: str, policy: RateLimitPolicy) -> None:
        try:
            await self._redis_client.async_client.delete(self._counter_key(key, policy))
        except RedisError as exc:
            self._logger.warning("Rate limit reset failed: %s", exc)

    @staticmethod
    def _decision(raw: list[int]) -> RateDecision:
        """Lua returns integers only, so the retry comes back in milliseconds."""
        allowed, remaining, retry_ms = (int(value) for value in raw)
        return RateDecision(
            allowed=bool(allowed),
            remaining=remaining,
            retry_after_seconds=retry_ms / 1000.0,
        )


__all__ = ["RedisRateLimiter"]
