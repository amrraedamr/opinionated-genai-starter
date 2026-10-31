import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy_utils import UUIDType

from base.models.definitions.database import DatabaseTable
from base.models.orm import Base, TimestampMixin

if TYPE_CHECKING:
    from base.models.orm.chat_history_message import ChatHistoryMessageDTO


class ConversationDTO(TimestampMixin, Base):
    __tablename__ = DatabaseTable.CONVERSATION

    __table_args__ = (
        Index(f"idx_{DatabaseTable.CONVERSATION}_user_id", "user_id"),
        Index(f"idx_{DatabaseTable.CONVERSATION}_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(binary=True), primary_key=True, insert_default=uuid.uuid4, init=False
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(binary=True),
        ForeignKey(f"{DatabaseTable.USER}.id"),
    )

    title: Mapped[str | None] = mapped_column(String(255), default=None)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default_factory=dict)

    messages: Mapped[list["ChatHistoryMessageDTO"]] = relationship(
        "ChatHistoryMessageDTO",
        back_populates="conversation",
        lazy="raise",
        init=False,
    )
