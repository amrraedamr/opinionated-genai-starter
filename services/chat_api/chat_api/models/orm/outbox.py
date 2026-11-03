import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy_utils import UUIDType

from chat_api.models.orm import ChatBase


class OutboxEventDTO(ChatBase):
    __tablename__ = "outbox_event"
    __table_args__ = (Index("idx_chat_outbox_pending", "published_at", "available_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUIDType(binary=True), primary_key=True)
    topic: Mapped[str] = mapped_column(String(255))
    event: Mapped[dict[str, Any]] = mapped_column(JSON)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default_factory=lambda: datetime.now(UTC)
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, default=None)
