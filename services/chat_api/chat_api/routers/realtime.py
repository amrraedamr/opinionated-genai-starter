"""The ticket a websocket is opened with, and the conversation it opens."""

import asyncio

from fastapi import APIRouter, WebSocket

from base.models.definitions.realtime import (
    ChannelDelivery,
    RealtimeCloseCode,
    RealtimeMessageType,
)
from base.models.view_models import BaseResponse
from base.models.view_models.auth import Principal
from base.models.view_models.realtime import (
    Presence,
    RealtimeEnvelope,
    TicketRequest,
    TicketResponse,
)
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
from base.tools.tenancy import TenantContext
from chat_api.mcp import MCP
from chat_api.models.view_models.realtime import (
    NotificationRequest,
    NotificationResponse,
)

router = APIRouter(prefix="/realtime", tags=["Realtime"])


@router.post("/tickets", response_model=TicketResponse)
async def issue_ticket(
    body: TicketRequest | None = None,
    principal: Principal = principal_dependency,
) -> TicketResponse:
    """Mint a ticket for the caller to open one socket with.

    Send back the `device_id` from an earlier ticket so a reconnect can resume.
    """
    return await TicketService().aissue(principal, (body or TicketRequest()).device_id)


@router.post("/notifications", response_model=BaseResponse[NotificationResponse])
async def notify_self(
    body: NotificationRequest,
    principal: Principal = principal_dependency,
) -> BaseResponse[NotificationResponse]:
    """Send yourself a durable message: every open device gets it, and one offline resumes it."""
    sent = await ConnectionManager().apublish_to_user(
        _user_key(principal),
        RealtimeEnvelope(
            type=RealtimeMessageType.MESSAGE,
            channel="notifications",
            payload=body.payload,
        ),
        delivery=ChannelDelivery.DURABLE,
    )
    return BaseResponse(data=NotificationResponse(seq=sent.seq or 0))


@router.get("/presence", response_model=BaseResponse[Presence])
async def my_presence(
    principal: Principal = principal_dependency,
) -> BaseResponse[Presence]:
    """Whether you have a socket open on any replica, and when you last did."""
    return BaseResponse(data=await ConnectionManager().apresence(_user_key(principal)))


def _user_key(principal: Principal) -> str:
    """The key a socket of this caller's is registered under, in the tenant its ticket carries."""
    TenantContext.set(principal.tenant_id)
    return str(principal.user_id or principal.subject)


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
        toolbox = await MCP.atoolbox()
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
