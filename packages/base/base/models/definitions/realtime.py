"""What travels over a socket, and the rules that bound it."""

from enum import StrEnum
from typing import Final

CURRENT_REALTIME_VERSION: Final = 1
SUPPORTED_REALTIME_VERSIONS: Final = frozenset({1})
"""Versioned like `EventMeta`: a client written against one shape must be able
to say so, and a server must be able to refuse a shape it does not know."""

TICKET_TTL_SECONDS: Final = 30
"""Long enough to open a socket, short enough that a leaked one is worthless."""

TICKET_SUBPROTOCOL_PREFIX: Final = "ticket."
"""Tickets arrive in `Sec-WebSocket-Protocol`, never the query string.

A browser cannot set an `Authorization` header on a WebSocket, so the
credential has to travel somewhere else. A query string is the obvious place
and the wrong one: it lands in access logs, proxy logs and browser history.
The subprotocol header is the one field a browser will send that nobody logs.
"""

MAX_MESSAGE_BYTES: Final = 65_536
HEARTBEAT_SECONDS: Final = 20.0
HEARTBEAT_TIMEOUT_SECONDS: Final = 60.0
BUDGET_EXHAUSTED_REASON: Final = "Token budget exhausted; try again shortly."
"""Three missed heartbeats. A socket whose peer vanished without a close frame
looks identical to an idle one until something asks."""

REALTIME_FANOUT_CHANNEL: Final = "realtime:fanout"
"""One channel for every replica. A message names the user it is for, and each
replica delivers to whichever of that user's devices it happens to hold."""

REALTIME_PRESENCE_NAMESPACE: Final = "realtime:presence"

PRESENCE_TTL_SECONDS: Final = 90
"""Longer than a heartbeat, so an idle-but-live user does not flicker offline,
and short enough that a replica killed without warning stops claiming them."""

REALTIME_INBOX_NAMESPACE: Final = "realtime:inbox"
REALTIME_INBOX_SEQUENCE_NAMESPACE: Final = "realtime:inbox:seq"
REALTIME_CURSOR_NAMESPACE: Final = "realtime:cursor"

INBOX_RETENTION_SECONDS: Final = 86_400
INBOX_MAX_ENTRIES: Final = 1_000
"""Oldest entries are dropped past this, so an active inbox stays bounded."""


class ChannelDelivery(StrEnum):
    """What a channel promises about a message nobody was there to receive."""

    EPHEMERAL = "ephemeral"
    """Reaches whoever is connected now. Right for a cursor or a typing
    indicator, where yesterday's is worse than none."""

    DURABLE = "durable"
    """Recorded first, so a device that was offline can ask for what it
    missed. Right for anything a user is owed."""


class RealtimeMessageType(StrEnum):
    """What a frame is for."""

    MESSAGE = "message"
    """Application payload, the only type a channel carries."""

    ACK = "ack"
    """A client confirming delivery, which advances its cursor."""

    RESUME = "resume"
    """A reconnecting client saying the last sequence it holds, so the server
    can send the gap. The client states it rather than the server reading its
    cursor, because a message can be delivered and its ack lost, and only the
    device knows which of the two happened."""

    PING = "ping"
    PONG = "pong"

    ERROR = "error"
    """Something was refused. The socket may or may not survive it."""

    CLOSING = "closing"
    """This replica is draining; reconnect somewhere else."""


class RealtimeCloseCode:
    """Close codes this library sends, beyond the ones the protocol defines."""

    NORMAL: Final = 1000
    GOING_AWAY: Final = 1001
    POLICY_VIOLATION: Final = 1008
    TOO_LARGE: Final = 1009
    TRY_LATER: Final = 1013
    INTERNAL_ERROR: Final = 1011
    """Something this end broke. Distinct from a policy refusal, which the
    client caused and can correct."""
    UNAUTHORIZED: Final = 4401
    """4000-4999 is the application range. A handshake refused for want of a
    valid ticket is not a protocol error, so it does not borrow one."""

    TIMED_OUT: Final = 4408
