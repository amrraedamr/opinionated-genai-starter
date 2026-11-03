import uuid

from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy_utils import UUIDType

from chat_api.models.definitions.database import DatabaseTable
from chat_api.models.orm import ChatBase, TimestampMixin


class PromptDTO(TimestampMixin, ChatBase):
    __tablename__ = DatabaseTable.PROMPT

    __table_args__ = (
        UniqueConstraint(
            "key", "version", name=f"uq_{DatabaseTable.PROMPT}_key_version"
        ),
        Index(f"idx_{DatabaseTable.PROMPT}_key", "key"),
        Index(f"idx_{DatabaseTable.PROMPT}_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(binary=True), primary_key=True, insert_default=uuid.uuid4, init=False
    )

    key: Mapped[str] = mapped_column(String(255))
    template: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1)


class ActivePromptDTO(TimestampMixin, ChatBase):
    __tablename__ = DatabaseTable.ACTIVE_PROMPT

    __table_args__ = (
        Index(f"idx_{DatabaseTable.ACTIVE_PROMPT}_updated_at", "updated_at"),
    )

    key: Mapped[str] = mapped_column(String(255), primary_key=True)

    prompt_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(binary=True),
        ForeignKey(f"{DatabaseTable.PROMPT}.id"),
    )

    prompt: Mapped[PromptDTO] = relationship(PromptDTO, lazy="joined", init=False)
