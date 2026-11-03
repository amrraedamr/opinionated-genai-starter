import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

from starlette.types import ASGIApp, Receive, Scope, Send

from base.middleware.responses import reject

WINDOW_SECONDS = 60.0


@dataclass
class _Bucket:
    hits: deque[float] = field(default_factory=deque)


class TrafficControlMiddleware:
    """Per-client request rate plus a process-wide concurrency ceiling.

    Single-process backpressure only. Fleet-wide limits belong at an ingress
    or in a shared store; this is the floor that keeps one replica standing.
    """

    def __init__(
        self,
        app: ASGIApp,
        concurrency: int,
        per_minute: int,
        auth_per_minute: int,
    ) -> None:
        self.app = app
        self._semaphore = asyncio.Semaphore(concurrency)
        self._per_minute = per_minute
        self._auth_per_minute = auth_per_minute
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        is_auth = "/auth/" in path or path.endswith("/auth")
        client = scope.get("client")
        client_key = client[0] if client else "unknown"
        key = f"{client_key}:auth" if is_auth else client_key
        limit = self._auth_per_minute if is_auth else self._per_minute

        now = time.monotonic()
        async with self._lock:
            self._evict_idle(now)
            bucket = self._buckets.setdefault(key, _Bucket())
            cutoff = now - WINDOW_SECONDS
            while bucket.hits and bucket.hits[0] <= cutoff:
                bucket.hits.popleft()
            if len(bucket.hits) >= limit:
                await reject(
                    scope,
                    receive,
                    send,
                    "Too many requests",
                    429,
                    {"Retry-After": "60"},
                )
                return
            bucket.hits.append(now)

        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=0.1)
        except TimeoutError:
            await reject(
                scope, receive, send, "Server is busy", 503, {"Retry-After": "1"}
            )
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self._semaphore.release()

    def _evict_idle(self, now: float) -> None:
        """Without this a client key is retained for every address ever seen."""
        cutoff = now - WINDOW_SECONDS
        for key, bucket in list(self._buckets.items()):
            while bucket.hits and bucket.hits[0] <= cutoff:
                bucket.hits.popleft()
            if not bucket.hits:
                del self._buckets[key]
