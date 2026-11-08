import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy_utils import UUIDType

from base.models.orm import UtcDateTime
from identity.models.orm import IdentityBase


class OutboxEventDTO(IdentityBase):
    __tablename__ = "outbox_event"
    __table_args__ = (
        Index("idx_identity_outbox_pending", "published_at", "available_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUIDType(binary=True), primary_key=True)
    topic: Mapped[str] = mapped_column(String(255))
    event: Mapped[dict[str, Any]] = mapped_column(JSON)
    occurred_at: Mapped[datetime] = mapped_column(UtcDateTime)
    available_at: Mapped[datetime] = mapped_column(
        UtcDateTime, default_factory=lambda: datetime.now(UTC)
    )
    published_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, default=None)
