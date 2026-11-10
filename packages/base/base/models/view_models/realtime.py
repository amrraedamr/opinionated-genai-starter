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
    """One frame, in a shape both ends agreed on."""

    type: RealtimeMessageType = RealtimeMessageType.MESSAGE
    channel: str | None = None
    id: UUID = Field(default_factory=uuid4)
    seq: int | None = None
    """Position in a durable channel; absent on an ephemeral one."""

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
    """A short-lived, single-use permission to open one socket."""

    ticket: str
    expires_at: datetime
    subject: str
    user_id: UUID | None = None
    tenant_id: TenantId | None = None

    device_id: str
    """Which of a user's clients this is, kept across reconnects so it can resume."""

    connection_id: UUID = Field(default_factory=uuid4)
    """This socket, as distinct from the device that opened it."""

    @property
    def expired(self) -> bool:
        return datetime.now(UTC) >= self.expires_at


class TicketRequest(BaseModel):
    """What a client asks for a ticket with."""

    device_id: str | None = None
    """Stable per client, stored by the client; absent means a fresh identity is minted."""


class TicketResponse(BaseModel):
    """What the issuing endpoint hands a client."""

    ticket: str
    expires_at: datetime
    subprotocol: str
    """Exactly what to put in `Sec-WebSocket-Protocol`."""

    device_id: str
    """Echoed back so a client that sent none can store the one it was given."""


class FanoutMessage(BaseModel):
    """One envelope on its way to every replica holding the named user."""

    user_id: str
    envelope: RealtimeEnvelope


class Presence(BaseModel):
    """Whether a user is connected anywhere, and when they last were."""

    user_id: str
    online: bool
    last_seen: datetime | None = None
