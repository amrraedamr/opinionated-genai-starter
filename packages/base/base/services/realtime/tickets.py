"""Short-lived permission to open one socket.

A browser cannot put an `Authorization` header on a WebSocket. The usual
workaround is a token in the query string, which puts a credential into access
logs, proxy logs and browser history — a bearer token that outlives the
request it was meant for.

A ticket avoids that. It is issued over an authenticated HTTP request, lives
about thirty seconds, works exactly once, and travels in
`Sec-WebSocket-Protocol`, which is the one field a browser will set and
nothing logs. Leaking one costs an attacker a single connection inside half a
minute, rather than the user's session.
"""

import secrets
from datetime import UTC, datetime, timedelta

from base.models.definitions.realtime import (
    TICKET_NAMESPACE,
    TICKET_SUBPROTOCOL_PREFIX,
    TICKET_TTL_SECONDS,
)
from base.models.view_models.auth import Principal
from base.models.view_models.realtime import ConnectionTicket, TicketResponse
from base.services.cache import AbstractCacheService
from base.services.cache.factory import get_cache_service
from base.services.common.meta import Singleton


class TicketService(metaclass=Singleton):
    """Issues and redeems connection tickets.

    Held in the cache rather than signed and stateless, because single use is
    the point: a signed ticket can be replayed until it expires, and there is
    nowhere to record that it was already spent.
    """

    def __init__(self, cache: AbstractCacheService | None = None) -> None:
        self._cache = cache

    @property
    def cache(self) -> AbstractCacheService:
        """Resolved on first use: settings are not registered at construction."""
        if self._cache is None:
            self._cache = get_cache_service()
        return self._cache

    async def aissue(
        self, principal: Principal, device_id: str | None = None
    ) -> TicketResponse:
        """Mint a ticket for the caller this request already authenticated.

        A client that sends no `device_id` is given one. It is expected to
        keep it: without a stable device identity a durable channel cannot
        tell a reconnect from a new client, and replays from the beginning
        every time.
        """
        ticket = secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC) + timedelta(seconds=TICKET_TTL_SECONDS)
        resolved_device = device_id or f"device-{secrets.token_urlsafe(8)}"
        record = ConnectionTicket(
            ticket=ticket,
            expires_at=expires_at,
            subject=principal.subject,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
            device_id=resolved_device,
        )
        await self.cache.aset(
            ticket,
            namespace=TICKET_NAMESPACE,
            value=record.jsonify(),
            ttl_seconds=TICKET_TTL_SECONDS,
        )
        return TicketResponse(
            ticket=ticket,
            expires_at=expires_at,
            subprotocol=f"{TICKET_SUBPROTOCOL_PREFIX}{ticket}",
            device_id=resolved_device,
        )

    async def aredeem(self, ticket: str | None) -> ConnectionTicket | None:
        """Spend a ticket atomically, so two racing sockets cannot both open."""
        if not ticket:
            return None
        stored = await self.cache.atake(ticket, namespace=TICKET_NAMESPACE)
        if stored is None:
            return None
        record = ConnectionTicket.model_validate_json(stored)
        return None if record.expired else record

    @staticmethod
    def from_subprotocols(offered: list[str]) -> str | None:
        """The ticket a client offered, if it offered one."""
        for protocol in offered:
            if protocol.startswith(TICKET_SUBPROTOCOL_PREFIX):
                return protocol[len(TICKET_SUBPROTOCOL_PREFIX) :] or None
        return None


__all__ = ["TicketService"]
