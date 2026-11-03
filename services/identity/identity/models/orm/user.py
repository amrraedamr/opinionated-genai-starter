import uuid

from sqlalchemy import Index, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy_utils import UUIDType

from identity.models.definitions.database import DatabaseTable
from identity.models.orm import IdentityBase, TimestampMixin


class UserDTO(TimestampMixin, IdentityBase):
    __tablename__ = DatabaseTable.USER

    __table_args__ = (
        Index(
            f"idx_{DatabaseTable.USER}_first_name_last_name", "first_name", "last_name"
        ),
        Index(f"idx_{DatabaseTable.USER}_email", "email"),
        Index(f"idx_{DatabaseTable.USER}_created_at", "created_at"),
        Index(f"idx_{DatabaseTable.USER}_updated_at", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType(binary=True), primary_key=True, insert_default=uuid.uuid4, init=False
    )

    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), default=None)

    first_name: Mapped[str | None] = mapped_column(String(50), default=None)
    last_name: Mapped[str | None] = mapped_column(String(50), default=None)
