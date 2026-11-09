"""One open socket, with the three bounds a socket does not get for free.

Every HTTP request on this service passes a size limit and a rate limiter on
the way in. A websocket passes neither: both middlewares return early on
anything that is not HTTP, and a socket is admitted once and then sends as
much as it likes for as long as it likes. That is the hole this closes.

Heartbeats are here for a different reason. A peer that vanished without a
close frame — a laptop lid, a dropped tunnel — looks exactly like an idle one.
Nothing notices until something tries to write, which for a subscription may
be never, so the connection and everything it holds leaks.
"""

import asyncio
import time
from typing import TYPE_CHECKING
from uuid import UUID

from base.models.definitions.rate_limiting import RateLimitPolicy
from base.models.definitions.realtime import (
    HEARTBEAT_SECONDS,
    HEARTBEAT_TIMEOUT_SECONDS,
    MAX_MESSAGE_BYTES,
    RealtimeCloseCode,
    RealtimeMessageType,
)
from base.models.view_models.realtime import ConnectionTicket, RealtimeEnvelope
from base.services.logging.console import ConsoleLoggerService
from base.services.rate_limiting import AbstractRateLimiter
from base.services.rate_limiting.factory import build_rate_limiter

if TYPE_CHECKING:
    # Typing only. FastAPI arrives with the `server` extra, and this package
    # has to stay importable without it: a worker that publishes to a channel
    # never accepts a socket.
    from fastapi import WebSocket


class RealtimeConnection:
    """Reads and writes one socket, refusing what it should not carry."""

    def __init__(
        self,
        websocket: "WebSocket",
        ticket: ConnectionTicket,
        *,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        heartbeat_timeout_seconds: float = HEARTBEAT_TIMEOUT_SECONDS,
        limiter: AbstractRateLimiter | None = None,
    ) -> None:
        self._websocket = websocket
        self._ticket = ticket
        self._max_bytes = max_message_bytes
        self._heartbeat_seconds = heartbeat_seconds
        self._heartbeat_timeout = heartbeat_timeout_seconds
        self._limiter = limiter
        self._logger = ConsoleLoggerService()
        self._last_seen = time.monotonic()
        self._closed = False

    @property
    def limiter(self) -> AbstractRateLimiter:
        if self._limiter is None:
            self._limiter = build_rate_limiter()
        return self._limiter

    @property
    def device_id(self) -> str:
        """Which client this is. Stable across reconnects, so a cursor resumes."""
        return self._ticket.device_id

    @property
    def connection_id(self) -> UUID:
        """This socket. A reconnect is the same device and a new connection."""
        return self._ticket.connection_id

    @property
    def user_id(self) -> UUID | None:
        return self._ticket.user_id

    @property
    def subject(self) -> str:
        """The token's subject, for a caller that carries no user id."""
        return self._ticket.subject

    @property
    def key(self) -> str:
        """Per user, not per connection: ten sockets is not ten budgets.

        Deliberately not per device either. A user opening five tabs is one
        person, and letting each tab have its own allowance is how a limit
        gets worked around by accident.
        """
        return AbstractRateLimiter.compose_key(
            tenant=self._ticket.tenant_id,
            user=str(self._ticket.user_id) if self._ticket.user_id else None,
            client_ip=self._ticket.subject,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def stale(self) -> bool:
        """Whether this peer has gone quiet for longer than it should have."""
        return time.monotonic() - self._last_seen > self._heartbeat_timeout

    async def areceive(self) -> RealtimeEnvelope | None:
        """The next frame worth acting on, or `None` when the socket is done.

        A refused frame does not end the connection. Oversize or too fast is
        usually one bad client loop, not an attack, and closing the socket
        turns a recoverable mistake into a reconnect storm.
        """
        raw = await self._websocket.receive_text()
        self._last_seen = time.monotonic()

        if len(raw.encode()) > self._max_bytes:
            await self._refuse("Message too large")
            return None

        decision = await self.limiter.aspend(self.key, RateLimitPolicy.WS_MESSAGES)
        if not decision.allowed:
            await self._refuse("Too many messages")
            return None

        try:
            envelope = RealtimeEnvelope.model_validate_json(raw)
        except ValueError:
            await self._refuse("Malformed frame")
            return None

        if envelope.type == RealtimeMessageType.PONG:
            return None
        if envelope.type == RealtimeMessageType.PING:
            await self.asend(RealtimeEnvelope.pong())
            return None
        return envelope

    async def asend(self, envelope: RealtimeEnvelope) -> None:
        """Write a frame, treating a dead socket as ordinary rather than fatal."""
        if self._closed:
            return
        try:
            await self._websocket.send_text(envelope.jsonify())
        except Exception as exc:
            self._logger.info("Realtime send failed, closing: %s", exc)
            self._closed = True

    async def aheartbeat(self) -> None:
        """Ping until the peer stops answering, then give up on it.

        Runs as its own task: a connection waiting on `receive` would
        otherwise never get round to asking whether anyone is still there.
        """
        while not self._closed:
            await asyncio.sleep(self._heartbeat_seconds)
            if self.stale:
                await self.aclose(
                    RealtimeCloseCode.TIMED_OUT, "No response to heartbeat"
                )
                return
            await self.asend(RealtimeEnvelope.ping())

    async def aclose(
        self, code: int = RealtimeCloseCode.NORMAL, reason: str = ""
    ) -> None:
        """Close once. Closing twice is an error on most implementations."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._websocket.close(code=code, reason=reason)
        except Exception as exc:  # pragma: no cover - already gone
            self._logger.debug("Realtime close failed: %s", exc)

    async def _refuse(self, reason: str) -> None:
        """Tell the client what was dropped, without ending the connection."""
        await self.asend(RealtimeEnvelope.error(reason))


__all__ = ["RealtimeConnection"]
