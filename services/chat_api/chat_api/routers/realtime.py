"""Issues the short-lived ticket a websocket is opened with.

The route lives here rather than in `base` because a service owns its own HTTP
surface — `base` ships no routers. What it must not own is the ticket's rules:
how long one lives, that it works once, how it is stored. Those are
`TicketService`, so a second service cannot quietly grant itself a longer or
reusable one.
"""

from fastapi import APIRouter

from base.models.view_models.auth import Principal
from base.models.view_models.realtime import TicketRequest, TicketResponse
from base.services.auth import principal_dependency
from base.services.realtime import TicketService

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
