"""Fixtures for the SQL area's integration tests.

These run against the Postgres container in `deploy/docker-compose.infra.yml`
and skip themselves when it is not up. What they are for is the behaviour
SQLite cannot answer for: a real transaction between two connections, a
constraint the server enforces, and timestamps that keep their offset.

Each test gets a table of its own, named from `unique`, so a rerun or a
parallel run never meets another test's rows.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime
from typing import Any, ClassVar

import pytest
from sqlalchemy import DateTime, Integer, String, Uuid
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    mapped_column,
)

from base.models.orm import SerializableMixin, TimestampMixin
from base.services.common.meta import KeyedSingleton, Singleton, purge_all
from base.services.sql.service import SQLService
from base.services.store.sql import SQLStore


def _abandon_engines() -> None:
    """Drop pooled connections without touching a loop that has already closed."""
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


def build_widget_model(table: str) -> type[Any]:
    """A soft-deleted table of this test's own, on a registry of its own."""

    class Base(MappedAsDataclass, DeclarativeBase, SerializableMixin):
        pass

    class Widget(TimestampMixin, Base):
        __tablename__ = table

        id: Mapped[uuid.UUID] = mapped_column(
            Uuid, primary_key=True, insert_default=uuid.uuid4, init=False
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
def widget(infra_settings: Any, unique: str) -> Iterator[type[Any]]:
    """The model under test, with its table created and then dropped."""
    model = build_widget_model(f"widget_{unique}")
    service = SQLService()
    with service._get_session() as session:
        model.metadata.create_all(session.get_bind())
    yield model
    with service._get_session() as session:
        model.metadata.drop_all(session.get_bind())


@pytest.fixture
def store(widget: type[Any]) -> WidgetStore:
    """A `SQLStore` over the model under test."""
    return WidgetStore(widget)
