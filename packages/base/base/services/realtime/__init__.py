"""Websockets: the scope one runs in, how it is authorised, and what bounds it.

A socket reaches none of the machinery an HTTP request passes through. It
carries no request scope, no correlation id and no span; it bypasses the size
limit and the rate limiter, both of which return early on anything that is not
HTTP; and it cannot be sent an `Authorization` header at all.

This package closes those, and nothing more. Who is connected and how a
message reaches them is the connection manager's job, which is separate.
"""

from base.services.realtime.bridge import RealtimeBridge
from base.services.realtime.connection import RealtimeConnection
from base.services.realtime.context import RealtimeContext
from base.services.realtime.handshake import accept_connection
from base.services.realtime.inbox import DurableInbox
from base.services.realtime.manager import ConnectionManager
from base.services.realtime.tickets import TicketService

__all__ = [
    "ConnectionManager",
    "RealtimeBridge",
    "DurableInbox",
    "RealtimeConnection",
    "RealtimeContext",
    "TicketService",
    "accept_connection",
]
