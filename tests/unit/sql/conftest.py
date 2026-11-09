"""Fixtures for the SQL area's unit tests.

Everything here runs against a throwaway SQLite file, because the behaviour
these tests pin — CRUD, soft delete, protected fields, page arithmetic — is
engine-independent and a container would only make it slower.

`SQLService` is a singleton and `SQLClient` is keyed by backend, so both
survive a test unless something drops them. The autouse purge here is what
stops a service built for SQLite in one test from answering a Postgres test
in another, and what gives every async test an engine bound to its own loop.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator
from datetime import datetime
from typing import Any, ClassVar

import pytest
from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    mapped_column,
)

from base.models.orm import SerializableMixin, TimestampMixin
from base.services.common.meta import KeyedSingleton, Singleton, purge_all
from base.services.configuration.mixins import SQLSettingsMixin
from base.services.sql.service import SQLService
from base.services.store.sql import SQLStore
from tests.conftest import MakeSettings

HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent


def _abandon_engines() -> None:
    """Drop pooled connections without touching a loop that has already closed.

    An async engine built inside a test holds connections owned by that test's
    event loop. `adispose` would need the loop back, so the pool is discarded
    instead and the connections die with the loop.
    """
    for instance in list(Singleton._instances.values()):
        if isinstance(instance, SQLService):
            instance.dispose()
    for client in list(KeyedSingleton._keyed_instances.values()):
        engine = getattr(client, "_async_engine", None)
        if engine is not None:
            engine.sync_engine.dispose(close=False)


@pytest.fixture(autouse=True)
def _fresh_services() -> Iterator[None]:
    """A service built in one test never answers the next."""
    purge_all()
    yield
    _abandon_engines()
    purge_all()


@pytest.fixture
def sqlite_path(unique: str) -> Iterator[pathlib.Path]:
    """A SQLite file of this test's own, removed afterwards."""
    path = HERE / f"store_{unique}.db"
    yield path
    for suffix in ("", "-wal", "-shm"):
        candidate = pathlib.Path(f"{path}{suffix}")
        if candidate.exists():
            candidate.unlink()


@pytest.fixture
def sqlite_settings(make_settings: MakeSettings, sqlite_path: pathlib.Path) -> Any:
    """Registered settings pointing every SQL subsystem at that file."""
    return make_settings(
        SQLSettingsMixin,
        sql_backend="sqlite",
        sqlite_db_path=str(sqlite_path),
        db_ssl_enabled=False,
    )


def build_widget_model(table: str) -> type[Any]:
    """A soft-deleted table of this test's own, on a registry of its own.

    A fresh declarative base per test keeps one test's table out of another's
    metadata, so `create_all` builds exactly the table under test.
    """

    class Base(MappedAsDataclass, DeclarativeBase, SerializableMixin):
        pass

    class Widget(TimestampMixin, Base):
        __tablename__ = table

        id: Mapped[int] = mapped_column(
            Integer, primary_key=True, autoincrement=True, init=False
        )
        name: Mapped[str] = mapped_column(String(200), unique=True)
        note: Mapped[str | None] = mapped_column(String(500), default=None)
        size: Mapped[int] = mapped_column(Integer, default=0)
        expires_at: Mapped[datetime | None] = mapped_column(
            DateTime(timezone=True), default=None
        )

    return Widget


class WidgetStore(SQLStore[Any]):
    """A store that searches the two text columns and protects the defaults."""

    search_fields: ClassVar[tuple[str, ...]] = ("name", "note")


@pytest.fixture
def widget(sqlite_settings: Any, unique: str) -> type[Any]:
    """The model under test, with its table created."""
    model = build_widget_model(f"widget_{unique}")
    with SQLService()._get_session() as session:
        model.metadata.create_all(session.get_bind())
    return model


@pytest.fixture
def store(widget: type[Any]) -> WidgetStore:
    """A `SQLStore` over the model under test."""
    return WidgetStore(widget)
