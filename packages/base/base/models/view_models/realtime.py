"""The frame a socket carries, and the ticket that opens one."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from base.models.definitions.realtime import (
    CURRENT_REALTIME_VERSION,
    SUPPORTED_REALTIME_VERSIONS,
    RealtimeMessageType,
)
from base.models.definitions.tenancy import TenantId
from base.models.view_models import BaseModel, JSONObject


class RealtimeEnvelope(BaseModel):
    """One frame, in a shape both ends agreed on.

    Versioned for the same reason an event is: a client built against an older
    server keeps working, and a server told a version it does not know says so
    rather than guessing at the payload.
    """

    type: RealtimeMessageType = RealtimeMessageType.MESSAGE
    channel: str | None = None
    id: UUID = Field(default_factory=uuid4)
    seq: int | None = None
    """Position in a durable channel. Absent on an ephemeral one, which has no
    order worth resuming from."""

    payload: JSONObject = Field(default_factory=dict)
    schema_version: int = CURRENT_REALTIME_VERSION

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if value not in SUPPORTED_REALTIME_VERSIONS:
            raise ValueError(f"Unsupported realtime schema version: {value}")
        return value

    @classmethod
    def error(cls, reason: str, *, channel: str | None = None) -> "RealtimeEnvelope":
        """A refusal the client can read, saying what and not why in detail."""
        return cls(
            type=RealtimeMessageType.ERROR,
            channel=channel,
            payload={"reason": reason},
        )

    @classmethod
    def ping(cls) -> "RealtimeEnvelope":
        return cls(type=RealtimeMessageType.PING)

    @classmethod
    def pong(cls) -> "RealtimeEnvelope":
        return cls(type=RealtimeMessageType.PONG)


class InboxEntry(BaseModel):
    """One durable message, at its place in a user's inbox."""

    sequence: int
    envelope: RealtimeEnvelope

    def delivered(self) -> RealtimeEnvelope:
        """The envelope as a client receives it, carrying its sequence."""
        return self.envelope.model_copy(update={"seq": self.sequence})


class ConnectionTicket(BaseModel):
    """A short-lived, single-use permission to open one socket.

    It carries the identity the socket will act as, so the handshake does not
    have to verify a token it cannot be sent.
    """

    ticket: str
    expires_at: datetime
    subject: str
    user_id: UUID | None = None
    tenant_id: TenantId | None = None

    device_id: str
    """Which of a user's clients this is — a phone, a laptop, one browser.

    Supplied by the client and kept across reconnects, which is the whole
    point: a durable channel tracks how far each device has read, and a device
    that gets a new identity on every reconnect can never resume. Minting one
    server-side per ticket would do exactly that.
    """

    connection_id: UUID = Field(default_factory=uuid4)
    """This socket, as distinct from the device that opened it.

    A device reconnecting is the same device and a different connection, and
    the registry has to be able to drop the old one without forgetting the
    cursor.
    """

    @property
    def expired(self) -> bool:
        return datetime.now(UTC) >= self.expires_at


class TicketRequest(BaseModel):
    """What a client asks for a ticket with."""

    device_id: str | None = None
    """Stable per client, stored by the client. Absent means a fresh identity,
    which is correct for a first connection and wrong for a resume."""


class TicketResponse(BaseModel):
    """What the issuing endpoint hands a client.

    The identity is deliberately absent: the client already knows who it is,
    and a reply that restates it is one more copy to leak.
    """

    ticket: str
    expires_at: datetime
    subprotocol: str
    """Exactly what to put in `Sec-WebSocket-Protocol`, so a client does not
    have to know how the prefix is spelled."""

    device_id: str
    """Echoed back so a client that sent none can store the one it was given
    and resume with it next time."""


class FanoutMessage(BaseModel):
    """One envelope on its way to every replica holding the named user.

    The user travels beside the envelope rather than inside it: a replica has
    to know who a message is for before it looks at what the message says, and
    the payload belongs to the application.
    """

    user_id: str
    envelope: RealtimeEnvelope


class Presence(BaseModel):
    """Whether a user is connected anywhere, and when they last were."""

    user_id: str
    online: bool
    last_seen: datetime | None = None
