"""Column types shared by every model, whichever backend is behind them."""

from datetime import UTC, datetime

from base.tools.errors import MissingExtraError

try:
    from sqlalchemy import DateTime, Dialect, TypeDecorator
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("sql", "sqlalchemy", feature="ORM column types") from exc


class UtcDateTime(TypeDecorator[datetime]):
    """A timestamp stored in UTC that reads back aware on every backend.

    SQLite keeps no offset, so a plain `DateTime(timezone=True)` written there
    comes back naive and comparing it against an aware `now` raises rather
    than answering. The emitted DDL is unchanged, so a migration written for
    the plain type stays valid.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        if value is None:
            return None
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)

    def process_result_value(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=UTC)


__all__ = ["UtcDateTime"]
