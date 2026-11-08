"""`UtcDateTime`: a timestamp that reads back aware whichever backend stored it.

SQLite keeps no offset, so a plain `DateTime(timezone=True)` written there
comes back naive and comparing it against an aware `now` raises. Every model
carrying `TimestampMixin` reads timestamps, and `InboxProcessor` compares a
claim's age in Python, so the type is pinned here rather than per service.
"""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.schema import CreateTable

from base.models.orm import TimestampMixin, UtcDateTime


class Base(sa.orm.DeclarativeBase):
    pass


class Row(Base):
    __tablename__ = "row"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    at: Mapped[datetime] = mapped_column(UtcDateTime)


@pytest.fixture
def session() -> Any:
    engine = sa.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as opened:
        yield opened


# Round trips


def test_an_aware_moment_reads_back_aware_from_sqlite(session: Any) -> None:
    moment = datetime.now(UTC)
    session.add(Row(id=1, at=moment))
    session.commit()
    session.expire_all()
    assert session.get(Row, 1).at.tzinfo is not None


def test_the_moment_that_comes_back_is_the_one_that_went_in(session: Any) -> None:
    moment = datetime(2026, 3, 1, 12, 30, tzinfo=UTC)
    session.add(Row(id=1, at=moment))
    session.commit()
    session.expire_all()
    assert session.get(Row, 1).at == moment


def test_another_offset_is_stored_as_the_same_instant_in_utc(session: Any) -> None:
    moment = datetime(2026, 3, 1, 15, 30, tzinfo=timezone(timedelta(hours=3)))
    session.add(Row(id=1, at=moment))
    session.commit()
    session.expire_all()
    read = session.get(Row, 1).at
    assert read == moment
    assert read.utcoffset() == timedelta(0)


def test_a_naive_moment_is_taken_as_utc(session: Any) -> None:
    session.add(Row(id=1, at=datetime(2026, 3, 1, 12, 30)))
    session.commit()
    session.expire_all()
    assert session.get(Row, 1).at == datetime(2026, 3, 1, 12, 30, tzinfo=UTC)


def test_nothing_stored_reads_back_as_nothing(session: Any) -> None:
    assert (
        UtcDateTime().process_result_value(None, sa.create_engine("sqlite://").dialect)
        is None
    )


# The comparison this exists for


def test_a_stored_moment_can_be_compared_against_an_aware_now(session: Any) -> None:
    """The inbox asks whether a claim is still fresh; a naive read raises instead."""
    session.add(Row(id=1, at=datetime.now(UTC)))
    session.commit()
    session.expire_all()
    assert session.get(Row, 1).at > datetime.now(UTC) - timedelta(seconds=30)


def test_the_plain_column_type_is_what_would_have_raised() -> None:
    """Why the type exists, pinned so nobody reverts it as redundant."""

    class Plain(Base):
        __tablename__ = "plain"
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
        at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))

    engine = sa.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Plain(id=1, at=datetime.now(UTC)))
        session.commit()
        session.expire_all()
        stored = session.get(Plain, 1).at
    assert stored.tzinfo is None
    with pytest.raises(TypeError, match="offset-naive"):
        _ = stored > datetime.now(UTC)


# Schema


@pytest.mark.parametrize(
    "dialect",
    [postgresql.dialect(), sqlite.dialect(), mysql.dialect()],
    ids=lambda d: d.name,
)
def test_the_ddl_is_what_the_plain_type_would_have_emitted(dialect: Any) -> None:
    """Migrations written against the plain type stay valid."""
    plain = sa.Table("t", sa.MetaData(), sa.Column("at", sa.DateTime(timezone=True)))
    decorated = sa.Table("t", sa.MetaData(), sa.Column("at", UtcDateTime))
    assert str(CreateTable(plain).compile(dialect=dialect)) == str(
        CreateTable(decorated).compile(dialect=dialect)
    )


def test_the_type_is_safe_to_cache_in_a_compiled_statement() -> None:
    assert UtcDateTime.cache_ok is True


# Every model carries it


@pytest.mark.parametrize("column", ["created_at", "updated_at", "deleted_at"])
def test_the_timestamp_mixin_uses_it_for_every_lifecycle_column(column: str) -> None:
    annotation = TimestampMixin.__dict__[column]
    assert isinstance(annotation.column.type, UtcDateTime)
