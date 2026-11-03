import uuid

from sqlalchemy import ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy_utils import UUIDType

from identity.models.definitions.database import DatabaseTable
from identity.models.orm import IdentityBase, TimestampMixin


class UserIdentityDTO(TimestampMixin, IdentityBase):
    __tablename__ = DatabaseTable.USER_IDENTITY

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "subject",
            name=f"uq_{DatabaseTable.USER_IDENTITY}_provider_subject",
        ),
        Index(f"idx_{DatabaseTable.USER_IDENTITY}_user_id", "user_id"),
        Index(f"idx_{DatabaseTable.USER_IDENTITY}_provider", "provider"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(binary=True), primary_key=True, insert_default=uuid.uuid4, init=False
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(binary=True),
        ForeignKey(f"{DatabaseTable.USER}.id"),
    )
    provider: Mapped[str] = mapped_column(String(64))
    subject: Mapped[str] = mapped_column(String(255))
    email_at_link: Mapped[str | None] = mapped_column(String(255), default=None)
