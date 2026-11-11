"""Short-lived, single-use permission to open one socket, sent in `Sec-WebSocket-Protocol`."""

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
from base.tools.tenancy import BearerContext


class TicketService(metaclass=Singleton):
    """Issues and redeems connection tickets."""

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
        """Mint a ticket for the caller this request already authenticated, carrying their token."""
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
            bearer=BearerContext.get(),
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
