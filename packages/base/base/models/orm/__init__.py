"""Reusable ORM mixins."""

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Self

from base.tools.errors import MissingExtraError

try:
    from sqlalchemy import DateTime
    from sqlalchemy.orm import MappedAsDataclass, MappedColumn, mapped_column
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("sql", "sqlalchemy", feature="ORM mixins") from exc

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
        data = self.to_dict(exclude=exclude or {"id"})
        return self.__class__(**data)


class TimestampMixin(MappedAsDataclass):
    created_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True),
        insert_default=lambda: datetime.now(UTC),
        init=False,
    )
    updated_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True),
        insert_default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        init=False,
    )
    deleted_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
        nullable=True,
        init=False,
    )

    def soft_delete(self) -> None:
        self.deleted_at = datetime.now(UTC)

    def restore(self) -> None:
        self.deleted_at = None


__all__ = ["SerializableMixin", "TimestampMixin"]
