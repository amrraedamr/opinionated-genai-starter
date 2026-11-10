"""Counters in this process, which is the floor under any number of replicas."""

import asyncio
import threading
import time
from collections import deque
from typing import ClassVar

from base.models.definitions.rate_limiting import RateLimitPolicy
from base.models.view_models.rate_limiting import RateDecision
from base.services.common.meta import SingletonABC
from base.services.rate_limiting import WINDOW_SECONDS, AbstractRateLimiter


class InMemoryRateLimiter(AbstractRateLimiter, metaclass=SingletonABC):
    """A sliding window per key, held in this process only.

    This is the behaviour the traffic middleware has always had, kept as it was
    rather than replaced: with one replica it is exactly right, and with many it
    still bounds what a single process will accept.

    A lock guards the windows because the sync and async paths share them and a
    read-modify-write across an await would otherwise interleave.
    """

    SERVICE_NAME: ClassVar[str] = "In-Memory Rate Limiter"

    def __init__(self, config: object | None = None) -> None:
        super().__init__(config)  # type: ignore[arg-type]
        self._windows: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _spend(self, key: str, policy: RateLimitPolicy, cost: int) -> RateDecision:
        with self._lock:
            return self._decide(key, policy, cost, time.monotonic())

    async def _aspend(
        self, key: str, policy: RateLimitPolicy, cost: int
    ) -> RateDecision:
        return await asyncio.to_thread(self._spend, key, policy, cost)

    def _charge(self, key: str, policy: RateLimitPolicy, cost: int) -> RateDecision:
        with self._lock:
            now = time.monotonic()
            self._decide(key, policy, 0, now)
            window = self._windows.setdefault(self._counter_key(key, policy), deque())
            window.extend([now] * cost)
            limit = self.budget_for(policy)
            return RateDecision(allowed=True, remaining=max(0, limit - len(window)))

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
        self, key: str, policy: RateLimitPolicy, cost: int, now: float
    ) -> RateDecision:
        """Drop what has aged out, then see whether `cost` still fits."""
        limit = self.budget_for(policy)
        counter = self._counter_key(key, policy)
        self._evict_idle(now)
        window = self._windows.setdefault(counter, deque())
        cutoff = now - WINDOW_SECONDS
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) + cost > limit:
            return RateDecision(
                allowed=False,
                remaining=max(0, limit - len(window)),
                retry_after_seconds=self._retry_after(window, now, cost, limit),
            )

        window.extend([now] * cost)
        remaining = max(0, limit - len(window))
        if not window:
            self._windows.pop(counter, None)
        retry = self._retry_after(window, now, 1, limit) if remaining == 0 else 0.0
        return RateDecision(
            allowed=True, remaining=remaining, retry_after_seconds=retry
        )

    @staticmethod
    def _retry_after(window: deque[float], now: float, cost: int, limit: int) -> float:
        """When enough of the oldest hits will have aged out to fit `cost`.

        A cost larger than the whole budget can never fit, so it is reported as
        a full window rather than as an index past the end of the deque.
        """
        needed = len(window) + cost - limit
        if needed > len(window):
            return WINDOW_SECONDS
        return max(0.0, window[needed - 1] + WINDOW_SECONDS - now)

    def _evict_idle(self, now: float) -> None:
        """Without this a key is retained for every address ever seen."""
        cutoff = now - WINDOW_SECONDS
        for counter, window in list(self._windows.items()):
            while window and window[0] <= cutoff:
                window.popleft()
            if not window:
                del self._windows[counter]


__all__ = ["InMemoryRateLimiter"]
