"""The ticket a websocket is opened with, and the conversation it opens.

The route lives here rather than in `base` because a service owns its own HTTP
surface — `base` ships no routers. What it must not own is the ticket's rules:
how long one lives, that it works once, how it is stored. Those are
`TicketService`, so a second service cannot quietly grant itself a longer or
reusable one.
"""

import asyncio

from fastapi import APIRouter, WebSocket

from base.models.definitions.realtime import RealtimeCloseCode
from base.models.view_models.auth import Principal
from base.models.view_models.realtime import TicketRequest, TicketResponse
from base.services.auth import principal_dependency
from base.services.guardrails.factory import build_guardrail_pipeline
from base.services.llm.realtime.factory import build_realtime_session
from base.services.realtime import (
    ConnectionManager,
    RealtimeBridge,
    RealtimeConnection,
    RealtimeContext,
    TicketService,
    accept_connection,
)
from base.services.tool_calling import Toolbox

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
    """Talk to a model over a socket, inside a connection scope."""
    ticket = await accept_connection(websocket)
    if ticket is None:
        return

    async with RealtimeContext.connection(ticket, channel="conversation"):
        connection = RealtimeConnection(websocket, ticket)
        manager = ConnectionManager()
        if not await manager.aregister(connection):
            await connection.aclose(RealtimeCloseCode.TRY_LATER, "Draining")
            return
        heartbeat = asyncio.create_task(connection.aheartbeat())
        toolbox = Toolbox()
        try:
            await RealtimeBridge(
                connection,
                build_realtime_session(toolbox=toolbox),
                toolbox=toolbox,
                guardrails=build_guardrail_pipeline(),
                manager=manager,
            ).arun()
        finally:
            heartbeat.cancel()
            await manager.aunregister(connection)
