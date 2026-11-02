from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EventMeta(BaseModel):
    """Delivery metadata every event carries."""

    event_id: UUID = Field(default_factory=uuid4)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    producer: str
    correlation_id: UUID | None = None


class Event(BaseModel):
    """A published event: a topic, metadata, and a payload."""

    topic: str
    meta: EventMeta
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def of(
        cls,
        topic: str,
        payload: BaseModel | dict[str, Any],
        *,
        producer: str,
        correlation_id: UUID | None = None,
    ) -> "Event":
        body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        return cls(
            topic=topic,
            meta=EventMeta(producer=producer, correlation_id=correlation_id),
            payload=body,
        )
