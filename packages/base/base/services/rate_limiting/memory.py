"""Counters in this process: exact for one replica, and the floor under any number."""

import asyncio
import threading
import time
from collections import deque
from typing import ClassVar

from base.models.definitions.rate_limiting import MEMORY_SWEEP_SECONDS, RateLimitPolicy
from base.models.view_models.rate_limiting import RateDecision
from base.services.common.meta import SingletonABC
from base.services.configuration.protocols import SupportsRateLimitConfig
from base.services.rate_limiting import WINDOW_SECONDS, AbstractRateLimiter


class _Window:
    """Spends in the last minute for one key, as (time, units) so a large spend is one entry."""

    __slots__ = ("hits", "total")

    def __init__(self) -> None:
        self.hits: deque[tuple[float, int]] = deque()
        self.total = 0

    def expire(self, cutoff: float) -> None:
        while self.hits and self.hits[0][0] <= cutoff:
            self.total -= self.hits.popleft()[1]

    def add(self, now: float, cost: int) -> None:
        if cost == 0:
            return
        if self.hits and self.hits[-1][0] == now:
            self.hits[-1] = (now, self.hits[-1][1] + cost)
        else:
            self.hits.append((now, cost))
        self.total += cost

    def retry_after(self, now: float, cost: int, limit: int) -> float:
        """When enough of the oldest spends age out for `cost` to fit."""
        needed = self.total + cost - limit
        if needed > self.total:
            return WINDOW_SECONDS
        freed = 0
        for at, units in self.hits:
            freed += units
            if freed >= needed:
                return max(0.0, at + WINDOW_SECONDS - now)
        return WINDOW_SECONDS


class InMemoryRateLimiter(AbstractRateLimiter, metaclass=SingletonABC):
    """A sliding window per key, in this process only; a lock covers both counterparts."""

    SERVICE_NAME: ClassVar[str] = "In-Memory Rate Limiter"

    def __init__(self, config: SupportsRateLimitConfig | None = None) -> None:
        super().__init__(config)
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()
        self._swept = 0.0

    def _spend(self, key: str, policy: RateLimitPolicy, cost: int) -> RateDecision:
        with self._lock:
            return self._decide(key, policy, cost, time.monotonic(), force=False)

    async def _aspend(
        self, key: str, policy: RateLimitPolicy, cost: int
    ) -> RateDecision:
        return await asyncio.to_thread(self._spend, key, policy, cost)

    def _charge(self, key: str, policy: RateLimitPolicy, cost: int) -> RateDecision:
        with self._lock:
            return self._decide(key, policy, cost, time.monotonic(), force=True)

    async def _acharge(
        self, key: str, policy: RateLimitPolicy, cost: int
    ) -> RateDecision:
        return await asyncio.to_thread(self._charge, key, policy, cost)

    def _reset(self, key: str, policy: RateLimitPolicy) -> None:
        with self._lock:
            self._windows.pop(self._counter_key(key, policy), None)

    async def _areset(self, key: str, policy: RateLimitPolicy) -> None:
        await asyncio.to_thread(self._reset, key, policy)

    def _decide(
        self, key: str, policy: RateLimitPolicy, cost: int, now: float, *, force: bool
    ) -> RateDecision:
        """Drop what has aged out, then spend `cost` if it fits or `force` says so."""
        limit = self.budget_for(policy)
        counter = self._counter_key(key, policy)
        self._sweep(now)
        window = self._windows.get(counter) or _Window()
        window.expire(now - WINDOW_SECONDS)

        if not force and window.total + cost > limit:
            self._keep(counter, window)
            return RateDecision(
                allowed=False,
                remaining=max(0, limit - window.total),
                retry_after_seconds=window.retry_after(now, cost, limit),
            )

        window.add(now, cost)
        self._keep(counter, window)
        remaining = max(0, limit - window.total)
        retry = window.retry_after(now, 1, limit) if remaining == 0 else 0.0
        return RateDecision(
            allowed=True, remaining=remaining, retry_after_seconds=retry
        )

    def _keep(self, counter: str, window: _Window) -> None:
        if window.total:
            self._windows[counter] = window
        else:
            self._windows.pop(counter, None)

    def _sweep(self, now: float) -> None:
        """Forget idle keys, at most once per `MEMORY_SWEEP_SECONDS`."""
        if now - self._swept < MEMORY_SWEEP_SECONDS:
            return
        self._swept = now
        cutoff = now - WINDOW_SECONDS
        for counter, window in list(self._windows.items()):
            window.expire(cutoff)
            if not window.total:
                del self._windows[counter]


__all__ = ["InMemoryRateLimiter"]
