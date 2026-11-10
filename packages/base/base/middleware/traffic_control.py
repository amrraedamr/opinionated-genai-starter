"""Admission control: a shared budget first, then this replica's own ceiling.

Two different questions, deliberately in one place. The budget says whether
this caller may be served at all, and it is shared across replicas when Redis
is configured. The semaphore says whether *this* process has room right now,
and is necessarily local.

Fleet-wide limits are often said to belong at an ingress instead. An ingress
sees addresses and paths; it does not see tenants, users, or what a reply cost,
and it does not see a message on an already-open websocket at all. Those are
the budgets worth holding, so they are held here.
"""

import asyncio

from starlette.types import ASGIApp, Receive, Scope, Send

from base.middleware.responses import reject
from base.models.definitions.rate_limiting import RateLimitPolicy
from base.models.definitions.realtime import RealtimeCloseCode
from base.models.view_models.rate_limiting import RateDecision
from base.services.rate_limiting import AbstractRateLimiter
from base.services.rate_limiting.factory import build_rate_limiter


class TrafficControlMiddleware:
    """Spends a request against the shared budget, then takes a local slot."""

    def __init__(
        self,
        app: ASGIApp,
        concurrency: int,
        limiter: AbstractRateLimiter | None = None,
    ) -> None:
        self.app = app
        self._semaphore = asyncio.Semaphore(concurrency)
        self._limiter = limiter

    @property
    def limiter(self) -> AbstractRateLimiter:
        """Resolved on first use: settings are not registered at construction."""
        if self._limiter is None:
            self._limiter = build_rate_limiter()
        return self._limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await self._websocket(scope, receive, send)
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        decision = await self.limiter.aspend(self._key(scope), self._policy(scope))
        if not decision.allowed:
            await self._refuse(scope, receive, send, decision)
            return

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

    async def _websocket(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Admit or refuse a handshake. Messages are the realtime layer's job.

        The concurrency semaphore is deliberately not taken here. It bounds
        bursts of short requests; a socket is held open for minutes, so a held
        slot would exhaust the ceiling with a handful of idle clients.
        """
        decision = await self.limiter.aspend(
            self._key(scope), RateLimitPolicy.WS_CONNECT
        )
        if decision.allowed:
            await self.app(scope, receive, send)
            return

        await receive()
        await send({"type": "websocket.close", "code": RealtimeCloseCode.TRY_LATER})

    async def _refuse(
        self, scope: Scope, receive: Receive, send: Send, decision: RateDecision
    ) -> None:
        """The wait comes from the decision, not a constant.

        A fixed `Retry-After: 60` tells every refused caller to come back at the
        same moment, which is how a limiter produces the spike it exists to
        prevent.
        """
        await reject(
            scope,
            receive,
            send,
            "Too many requests",
            429,
            {"Retry-After": decision.retry_after_header},
        )

    @staticmethod
    def _policy(scope: Scope) -> RateLimitPolicy:
        """Sign-in traffic is held far tighter than anything else."""
        path = str(scope.get("path", ""))
        is_auth = "/auth/" in path or path.endswith("/auth")
        return RateLimitPolicy.AUTH if is_auth else RateLimitPolicy.REQUESTS

    @staticmethod
    def _key(scope: Scope) -> str:
        """The address, because nothing better is known this early.

        Authentication runs as a dependency, after this: there is no principal
        and no tenant yet. Budgets that need one are spent where it is known —
        the LLM boundary for tokens, the realtime layer for messages.
        """
        client = scope.get("client")
        return AbstractRateLimiter.compose_key(client_ip=client[0] if client else None)
