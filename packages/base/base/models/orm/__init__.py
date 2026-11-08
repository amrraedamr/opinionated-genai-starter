"""Reusable ORM mixins."""

from dataclasses import asdict, fields
from datetime import UTC, datetime
from typing import Self

from base.tools.errors import MissingExtraError

try:
    from sqlalchemy import String
    from sqlalchemy.orm import MappedAsDataclass, MappedColumn, mapped_column
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("sql", "sqlalchemy", feature="ORM mixins") from exc

from base.models.orm.types import UtcDateTime
from base.tools.misc import _make_json_serializable


class SerializableMixin:
    """Dict conversion helpers for `MappedAsDataclass` declarative models."""

    def to_dict(self, exclude: set[str] | None = None) -> dict:
        result = asdict(self)  # type: ignore[call-overload]
        if exclude:
            result = {k: v for k, v in result.items() if k not in exclude}
        return _make_json_serializable(result)  # type: ignore

    def update_from_dict(self, data: dict) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def clone(self, exclude: set[str] | None = None) -> Self:
        """Copy the columns `__init__` accepts, always dropping the identity."""
        skip = {"id"} | (exclude or set())
        data = {
            column.name: getattr(self, column.name)
            for column in fields(self)  # type: ignore[arg-type]
            if column.init and column.name not in skip
        }
        return self.__class__(**data)


class TimestampMixin(MappedAsDataclass):
    created_at: MappedColumn[datetime] = mapped_column(
        UtcDateTime,
        insert_default=lambda: datetime.now(UTC),
        init=False,
    )
    updated_at: MappedColumn[datetime] = mapped_column(
        UtcDateTime,
        insert_default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        init=False,
    )
    deleted_at: MappedColumn[datetime | None] = mapped_column(
        UtcDateTime,
        default=None,
        nullable=True,
        init=False,
    )

    def soft_delete(self) -> None:
        self.deleted_at = datetime.now(UTC)

    def restore(self) -> None:
        self.deleted_at = None


class TenantScopedMixin(MappedAsDataclass):
    """A row belonging to one customer.

    Carrying this is how a model opts into separation: `SQLStore` narrows every
    read to the tenant in context and stamps every write with it, and refuses
    rather than guessing when nothing set one.
    """

    tenant_id: MappedColumn[str] = mapped_column(String(255), index=True, init=False)


__all__ = [
    "SerializableMixin",
    "TenantScopedMixin",
    "TimestampMixin",
    "UtcDateTime",
]
