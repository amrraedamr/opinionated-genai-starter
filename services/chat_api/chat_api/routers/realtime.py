"""The ticket a websocket is opened with, and the conversation it opens.

The route lives here rather than in `base` because a service owns its own HTTP
surface — `base` ships no routers. What it must not own is the ticket's rules:
how long one lives, that it works once, how it is stored. Those are
`TicketService`, so a second service cannot quietly grant itself a longer or
reusable one.
"""

from fastapi import APIRouter, WebSocket

from base.models.definitions.realtime import RealtimeCloseCode
from base.models.view_models.auth import Principal
from base.models.view_models.realtime import TicketRequest, TicketResponse
from base.services.auth import principal_dependency
from base.services.llm.realtime.factory import build_realtime_session
from base.services.realtime import (
    ConnectionManager,
    RealtimeBridge,
    RealtimeConnection,
    RealtimeContext,
    TicketService,
    accept_connection,
)

router = APIRouter(prefix="/realtime", tags=["Realtime"])


@router.post("/tickets", response_model=TicketResponse)
async def issue_ticket(
    body: TicketRequest | None = None,
    principal: Principal = principal_dependency,
) -> TicketResponse:
    """Mint a ticket for the caller to open one socket with.

    Authenticated as an ordinary request, because this is the last point at
    which a bearer token can be sent: a browser cannot put one on the socket
    that follows.

    A client sends the `device_id` it was given last time, so the server can
    tell a reconnecting laptop from a second phone. One that sends none is
    issued one to keep.
    """
    return await TicketService().aissue(principal, (body or TicketRequest()).device_id)


@router.websocket("/conversation")
async def conversation(websocket: WebSocket) -> None:
    """Talk to a model over a socket, start to finish.

    Here rather than in `base` for the same reason as the ticket route: a
    service owns its HTTP surface. What `base` owns is everything this calls,
    so a second service wiring its own conversation gets the same handshake,
    the same bounds and the same guardrails without copying them.

    The whole thing runs inside a connection scope. A socket never passes
    `request_context`, so without one a scoped service resolves whatever the
    last HTTP request left behind, and anything billed here lands in a scope
    nobody reads.
    """
    ticket = await accept_connection(websocket)
    if ticket is None:
        return

    async with RealtimeContext.connection(ticket, channel="conversation"):
        connection = RealtimeConnection(websocket, ticket)
        manager = ConnectionManager()
        if not await manager.aregister(connection):
            await connection.aclose(RealtimeCloseCode.TRY_LATER, "Draining")
            return
        try:
            await RealtimeBridge(connection, build_realtime_session()).arun()
        finally:
            await manager.aunregister(connection)
