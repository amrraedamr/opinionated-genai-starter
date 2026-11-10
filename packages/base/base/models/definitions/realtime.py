"""What travels over a socket, and the rules that bound it."""

from enum import StrEnum
from typing import Final

CURRENT_REALTIME_VERSION: Final = 1
SUPPORTED_REALTIME_VERSIONS: Final = frozenset({1})

TICKET_TTL_SECONDS: Final = 30

TICKET_SUBPROTOCOL_PREFIX: Final = "ticket."
"""Tickets arrive in `Sec-WebSocket-Protocol`, never the query string, which gets logged."""
TICKET_NAMESPACE: Final = "realtime:ticket"

MAX_MESSAGE_BYTES: Final = 65_536
HEARTBEAT_SECONDS: Final = 20.0
HEARTBEAT_TIMEOUT_SECONDS: Final = 60.0
BUDGET_EXHAUSTED_REASON: Final = "Token budget exhausted; try again shortly."

REALTIME_FANOUT_CHANNEL: Final = "realtime:fanout"
"""One channel shared by every replica; each message names the user it is for."""

REALTIME_PRESENCE_NAMESPACE: Final = "realtime:presence"

PRESENCE_TTL_SECONDS: Final = 90

REALTIME_INBOX_NAMESPACE: Final = "realtime:inbox"
REALTIME_INBOX_SEQUENCE_NAMESPACE: Final = "realtime:inbox:seq"
REALTIME_CURSOR_NAMESPACE: Final = "realtime:cursor"

INBOX_RETENTION_SECONDS: Final = 86_400
INBOX_MAX_ENTRIES: Final = 1_000
"""Oldest entries are dropped past this, so an active inbox stays bounded."""


class ChannelDelivery(StrEnum):
    """What a channel promises about a message nobody was there to receive."""

    EPHEMERAL = "ephemeral"
    """Reaches only whoever is connected now."""

    DURABLE = "durable"
    """Recorded first, so a device that was offline can ask for what it missed."""


class RealtimeMessageType(StrEnum):
    """What a frame is for."""

    MESSAGE = "message"
    """Application payload, the only type a channel carries."""

    ACK = "ack"
    """A client confirming delivery, which advances its cursor."""

    RESUME = "resume"
    """A reconnecting client saying the last sequence it holds, so the server can send the gap."""

    PING = "ping"
    PONG = "pong"

    ERROR = "error"
    """Something was refused. The socket may or may not survive it."""

    CLOSING = "closing"
    """This replica is draining; reconnect somewhere else."""


class RealtimeCloseCode:
    """WebSocket close codes this library uses, standard and application-range."""

    NORMAL: Final = 1000
    GOING_AWAY: Final = 1001
    POLICY_VIOLATION: Final = 1008
    TOO_LARGE: Final = 1009
    TRY_LATER: Final = 1013
    INTERNAL_ERROR: Final = 1011
    UNAUTHORIZED: Final = 4401

    TIMED_OUT: Final = 4408
