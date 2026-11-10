"""Turning an offered ticket into an accepted, authenticated socket."""

from typing import TYPE_CHECKING

from base.models.definitions.realtime import (
    TICKET_SUBPROTOCOL_PREFIX,
    RealtimeCloseCode,
)
from base.models.view_models.realtime import ConnectionTicket
from base.services.realtime.tickets import TicketService

if TYPE_CHECKING:
    from fastapi import WebSocket


async def accept_connection(websocket: "WebSocket") -> ConnectionTicket | None:
    """Redeem the offered ticket and accept, or refuse the handshake.

    The offered subprotocol must be echoed back, or the client fails the connection.
    """
    offered = list(websocket.scope.get("subprotocols") or [])
    ticket_value = TicketService.from_subprotocols(offered)

    record = await TicketService().aredeem(ticket_value)
    if record is None:
        await websocket.close(
            code=RealtimeCloseCode.UNAUTHORIZED, reason="Invalid or expired ticket"
        )
        return None

    await websocket.accept(
        subprotocol=f"{TICKET_SUBPROTOCOL_PREFIX}{ticket_value}"
        if ticket_value
        else None
    )
    return record


__all__ = ["accept_connection"]
