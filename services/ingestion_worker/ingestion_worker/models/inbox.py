import uuid
from datetime import datetime

from sqlalchemy import Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy_utils import UUIDType

from base.models.orm import UtcDateTime
from ingestion_worker.models import WorkerBase


class InboxEventDTO(WorkerBase):
    __tablename__ = "inbox_event"
    __table_args__ = (Index("idx_inbox_state_locked", "state", "locked_at"),)

    event_id: Mapped[uuid.UUID] = mapped_column(UUIDType(binary=True), primary_key=True)
    topic: Mapped[str] = mapped_column(String(255))
    state: Mapped[str] = mapped_column(String(32))
    locked_at: Mapped[datetime] = mapped_column(UtcDateTime)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    processed_at: Mapped[datetime | None] = mapped_column(UtcDateTime, default=None)
    last_error: Mapped[str | None] = mapped_column(Text, default=None)
