"""Fixtures for the outbox and inbox against the real database.

Both patterns are claims about transactions, so there is no way to test them
without one. Each test gets its own pair of tables, created and dropped
around it, and the `SQLService` singleton is disposed in between so no engine
outlives the loop it was built on.
"""

import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import JSON, DateTime, Integer, String, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from base.services.common.meta import purge_all
from contracts.events import Event
from tests.conftest import CoreSettings


class MessagingBase(DeclarativeBase):
    """A declarative root of its own, so no service's metadata is touched."""


class OutboxEventRow(MessagingBase):
    """A producer's outbox table, as `OutboxRow` describes it."""

    __tablename__ = "test_messaging_outbox"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    topic: Mapped[str] = mapped_column(String(255))
    event: Mapped[dict[str, Any]] = mapped_column(JSON)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, default=None)


class InboxEventRow(MessagingBase):
    """A consumer's inbox table, as `InboxRow` describes it."""

    __tablename__ = "test_messaging_inbox"

    event_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    topic: Mapped[str] = mapped_column(String(255))
    state: Mapped[str] = mapped_column(String(32))
    locked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    last_error: Mapped[str | None] = mapped_column(Text, default=None)


class CollectingBroker:
    """A broker that publishes nowhere and remembers everything."""

    def __init__(self) -> None:
        self.published: list[Event] = []

    async def apublish(self, event: Event) -> None:
        self.published.append(event)


class FailingBroker:
    """A broker that is down, as one is during an outage."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or RuntimeError("transport unavailable")
        self.attempts = 0

    async def apublish(self, event: Event) -> None:
        self.attempts += 1
        raise self.error


@pytest.fixture(autouse=True)
def _fresh_singletons() -> Iterator[None]:
    purge_all()
    yield
    purge_all()


@pytest.fixture
async def messaging_tables(infra_settings: CoreSettings) -> AsyncIterator[Any]:
    """An empty outbox and inbox, dropped again afterwards."""
    del infra_settings
    from base.services.sql.service import SQLService

    service = SQLService()
    async with service.atransaction() as session:
        connection = await session.connection()
        await connection.run_sync(MessagingBase.metadata.drop_all)
        await connection.run_sync(MessagingBase.metadata.create_all)
    try:
        yield service
    finally:
        async with service.atransaction() as session:
            connection = await session.connection()
            await connection.run_sync(MessagingBase.metadata.drop_all)
        await service.adispose()


def event(topic: str = "test.messaging", **payload: Any) -> Event:
    return Event.of(topic, payload, producer="test-producer")
