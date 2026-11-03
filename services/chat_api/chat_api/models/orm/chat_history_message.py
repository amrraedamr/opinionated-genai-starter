import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy_utils import UUIDType

from base.models.definitions.message import MessageRole
from chat_api.models.definitions.database import DatabaseTable
from chat_api.models.orm import ChatBase, TimestampMixin

if TYPE_CHECKING:
    from chat_api.models.orm.conversation import ConversationDTO


class ChatHistoryMessageDTO(TimestampMixin, ChatBase):
    __tablename__ = DatabaseTable.CHAT_HISTORY_MESSAGE

    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "sequence",
            name=f"uq_{DatabaseTable.CHAT_HISTORY_MESSAGE}_conv_seq",
        ),
        Index(
            f"idx_{DatabaseTable.CHAT_HISTORY_MESSAGE}_conv_seq",
            "conversation_id",
            "sequence",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(binary=True), primary_key=True, insert_default=uuid.uuid4, init=False
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(binary=True),
        ForeignKey(f"{DatabaseTable.CONVERSATION}.id"),
    )
    sequence: Mapped[int] = mapped_column(Integer)

    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole))

    content: Mapped[str | None] = mapped_column(Text, default=None)
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=None)
    tool_call_id: Mapped[str | None] = mapped_column(String(255), default=None)
    token_count: Mapped[int | None] = mapped_column(Integer, default=None)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default_factory=dict)

    conversation: Mapped["ConversationDTO"] = relationship(
        "ConversationDTO",
        back_populates="messages",
        lazy="raise",
        init=False,
    )
