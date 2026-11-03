from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

CURRENT_EVENT_VERSION = 1
SUPPORTED_EVENT_VERSIONS = frozenset({CURRENT_EVENT_VERSION})


class EventMeta(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    producer: str
    correlation_id: UUID | None = None
    schema_version: int = CURRENT_EVENT_VERSION

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: int) -> int:
        if value not in SUPPORTED_EVENT_VERSIONS:
            raise ValueError(f"Unsupported event schema version: {value}")
        return value


class Event(BaseModel):
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
        schema_version: int = CURRENT_EVENT_VERSION,
    ) -> "Event":
        body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        return cls(
            topic=topic,
            meta=EventMeta(
                producer=producer,
                correlation_id=correlation_id,
                schema_version=schema_version,
            ),
            payload=body,
        )
