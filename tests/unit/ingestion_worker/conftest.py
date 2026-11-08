"""What the ingestion worker needs before any of it can be exercised.

The service is run the way it is deployed — the real composition root, the
real inbox, the real in-process transport — with one substitution a unit test
has to make: the database is a throwaway SQLite file, as in `tests/unit/sql`.
A claim, a duplicate and a lock whose owner died are engine-independent, and
a container would only make them slower.

Its settings are one module-level object the worker imported by value, so a
test's settings only take effect once that name is rebound wherever it
landed. Every service here is a singleton, and an async one caches an engine
bound to the loop it was built on, so the registries are emptied around each
test and pooled connections are abandoned rather than closed.
"""

from __future__ import annotations

import importlib
import pathlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict
from sqlalchemy import func, select

from base.models.definitions.messaging import InboxState
from base.services.broker import AbstractBroker, Handler
from base.services.common.meta import KeyedSingleton, Singleton, purge_all
from base.services.configuration import configure
from base.services.sql.service import SQLService
from contracts.events import Event
from contracts.v1 import chat, identity
from ingestion_worker import handlers
from ingestion_worker.models import WorkerBase
from ingestion_worker.models.inbox import InboxEventDTO
from ingestion_worker.settings import IngestionWorkerSettings

HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent

SETTINGS_HOLDERS: tuple[str, ...] = (
    "ingestion_worker.settings",
    "ingestion_worker.worker",
)
"""Every module that did `from ingestion_worker.settings import settings`."""

PRODUCER: str = "test-producer"


class IsolatedWorkerSettings(IngestionWorkerSettings):
    """The worker's own settings, with a developer's `.env` kept out."""

    model_config = SettingsConfigDict(
        env_file=None, extra="ignore", populate_by_name=True
    )


class RecordingBroker(AbstractBroker):
    """A transport that goes nowhere and remembers the order it was driven in.

    Startup ordering is a claim about the composition root rather than about
    any one transport, so what `arun` did and when is kept here.
    """

    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, str, Handler]] = []
        self.published: list[Event] = []
        self.trail: list[str] = []

    async def _apublish(self, event: Event) -> None:
        self.trail.append("publish")
        self.published.append(event)

    async def _asubscribe(self, topic: str, group: str, handler: Handler) -> None:
        self.trail.append("subscribe")
        self.subscriptions.append((topic, group, handler))

    async def _astart(self) -> None:
        self.trail.append("start")

    async def _astop(self) -> None:
        self.trail.append("stop")

    async def _areplay_dlq(self, topic: str, *, limit: int = 100) -> int:
        return 0

    async def _ahealth_check(self) -> bool:
        return True

    def handler_for(self, topic: str) -> Handler:
        """The wrapped handler the worker subscribed to one topic."""
        for subscribed, _group, handler in self.subscriptions:
            if subscribed == topic:
                return handler
        raise AssertionError(f"nothing subscribed to {topic}")

    @property
    def topics(self) -> list[str]:
        return [topic for topic, _group, _handler in self.subscriptions]

    @property
    def groups(self) -> list[str]:
        return [group for _topic, group, _handler in self.subscriptions]


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


@pytest.fixture(autouse=True)
def _fresh_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handlers count into a module global that outlives one test."""
    monkeypatch.setattr(handlers, "_processed_count", 0)


@pytest.fixture
def sqlite_path(unique: str) -> Iterator[pathlib.Path]:
    """A SQLite file of this test's own, removed afterwards."""
    path = HERE / f"worker_{unique}.db"
    yield path
    for suffix in ("", "-wal", "-shm"):
        candidate = pathlib.Path(f"{path}{suffix}")
        if candidate.exists():
            candidate.unlink()


@pytest.fixture
def make_worker_settings(
    monkeypatch: pytest.MonkeyPatch, sqlite_path: pathlib.Path
) -> Any:
    """Register settings for one test and give it an empty inbox."""

    def _make(**overrides: Any) -> IngestionWorkerSettings:
        values: dict[str, Any] = {
            "sql_backend": "sqlite",
            "sqlite_db_path": str(sqlite_path),
            "db_ssl_enabled": False,
            "broker": "memory",
            "redis_url": "",
            **overrides,
        }
        settings = IsolatedWorkerSettings(**values)
        for name in SETTINGS_HOLDERS:
            monkeypatch.setattr(importlib.import_module(name), "settings", settings)
        configure(settings)
        with SQLService()._get_session() as session:
            WorkerBase.metadata.create_all(session.get_bind())
        return settings

    return _make


@pytest.fixture
def worker_settings(make_worker_settings: Any) -> IngestionWorkerSettings:
    """SQLite, the in-process transport, and an empty inbox table."""
    return make_worker_settings()


@pytest.fixture
def broker() -> RecordingBroker:
    return RecordingBroker()


class Recorder:
    """A handler that counts its runs and can be told to fail."""

    def __init__(self, failing: bool = False) -> None:
        self.runs: list[Event] = []
        self.failing = failing

    async def __call__(self, incoming: Event) -> None:
        self.runs.append(incoming)
        if self.failing:
            raise ValueError("handler failed")

    @property
    def count(self) -> int:
        return len(self.runs)


def user_registered(**payload: Any) -> Event:
    body: dict[str, Any] = {
        "user_id": "11111111-1111-1111-1111-111111111111",
        "email": "someone@example.com",
        "registered_at": "2026-01-01T00:00:00+00:00",
        **payload,
    }
    return Event.of(identity.USER_REGISTERED, body, producer=PRODUCER)


def conversation_created(**payload: Any) -> Event:
    body: dict[str, Any] = {
        "conversation_id": "22222222-2222-2222-2222-222222222222",
        "user_id": "11111111-1111-1111-1111-111111111111",
        "created_at": "2026-01-01T00:00:00+00:00",
        **payload,
    }
    return Event.of(chat.CONVERSATION_CREATED, body, producer=PRODUCER)


async def stored(delivered: Event) -> InboxEventDTO:
    """The inbox row for one event, which must exist."""
    async with SQLService().atransaction() as session:
        row = await session.get(InboxEventDTO, delivered.meta.event_id)
        assert row is not None
        return row


async def find(delivered: Event) -> InboxEventDTO | None:
    async with SQLService().atransaction() as session:
        return await session.get(InboxEventDTO, delivered.meta.event_id)


async def row_count() -> int:
    async with SQLService().atransaction() as session:
        return int(
            await session.scalar(select(func.count()).select_from(InboxEventDTO)) or 0
        )


async def set_claim(delivered: Event, *, state: InboxState, age_seconds: float) -> None:
    """Rewrite a claim, so a test need not wait out the claim timeout."""
    async with SQLService().atransaction() as session:
        row = await session.get(InboxEventDTO, delivered.meta.event_id)
        assert row is not None
        row.state = state
        row.locked_at = datetime.now(UTC) - timedelta(seconds=age_seconds)
