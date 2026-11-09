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

    The accepted subprotocol is echoed back because the handshake requires it:
    a client that offered one and is answered with none must treat the
    connection as failed, which would look like the server being broken rather
    than the ticket being wrong.

    Refusal closes rather than accepting-then-closing. Accepting first tells a
    client it is connected for as long as it takes the close frame to arrive,
    and anything it sent in that window is silently lost.
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
