from dataclasses import asdict
from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import (
    DeclarativeBase,
    MappedAsDataclass,
    MappedColumn,
    mapped_column,
)

from base.tools.misc import _make_json_serializable


class Base(MappedAsDataclass, DeclarativeBase):
    def to_dict(self, exclude: set[str] | None = None) -> dict:
        result = asdict(self)
        if exclude:
            result = {k: v for k, v in result.items() if k not in exclude}
        return _make_json_serializable(result)  # type: ignore

    def update_from_dict(self, data: dict) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def clone(self, exclude: set[str] | None = None) -> "Base":
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


def _import_all_tables():
    import importlib
    import pkgutil

    for _, name, _ in pkgutil.walk_packages(__path__):
        importlib.import_module(f"base.models.orm.{name}")


_import_all_tables()
