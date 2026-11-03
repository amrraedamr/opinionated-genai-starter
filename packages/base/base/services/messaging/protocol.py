from datetime import datetime
from typing import Any, Protocol
from uuid import UUID


class OutboxRow(Protocol):
    """Columns a producer's own outbox table must provide.

    The table belongs to the service, not to `base` — this is the shape the
    dispatcher needs in order to drive it.
    """

    id: UUID
    topic: str
    event: dict[str, Any]
    occurred_at: datetime
    available_at: datetime
    published_at: datetime | None
    attempts: int
    last_error: str | None


class InboxRow(Protocol):
    """Columns a consumer's own inbox table must provide."""

    event_id: UUID
    topic: str
    state: str
    locked_at: datetime
    attempts: int
    processed_at: datetime | None
    last_error: str | None
