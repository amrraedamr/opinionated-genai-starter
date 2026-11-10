"""One websocket, bounded by message size, rate limit and heartbeat."""

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
    # FastAPI comes with the `server` extra; this package must import without it.
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
        """Rate-limit key, shared by all of a user's sockets and devices."""
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
        """The next frame worth acting on, or `None` for a refused or heartbeat frame."""
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
        """Ping until the peer stops answering, then give up on it."""
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
