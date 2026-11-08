"""The worker's own inbox table against the database it runs on in production.

SQLite ignores `FOR UPDATE` and keeps no offset, so the unit tier can show
that the claim logic is right but not that it holds when two replicas reach
one row at the same moment. That, and whether `timestamptz` and the binary
UUID key survive a round trip, is what needs the container.

The table is built by this service's own migration rather than by
`create_all`, so a schema that drifted from the model fails here too.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import types
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select

from base.models.definitions.messaging import InboxState
from base.models.view_models.exceptions.sql import SQLConflictError
from base.services.common.meta import purge_all
from base.services.messaging import InboxProcessor
from base.services.sql.service import SQLService
from contracts.events import Event
from contracts.v1 import identity
from ingestion_worker.models.inbox import InboxEventDTO

pytestmark = pytest.mark.integration

REVISION = (
    pathlib.Path(__file__).resolve().parents[2]
    / "services/ingestion_worker/alembic/versions/2026_11_02_0001-worker_inbox.py"
)


def revision_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("worker_inbox_revision", REVISION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@pytest.fixture(autouse=True)
def _fresh_singletons() -> Any:
    purge_all()
    yield
    purge_all()


@pytest.fixture
async def worker_inbox(infra_settings: Any) -> AsyncIterator[SQLService]:
    """An empty `inbox_event`, created by the migration and dropped afterwards."""
    del infra_settings
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    def migrate(connection: Any, direction: str) -> None:
        module = revision_module()
        with Operations.context(MigrationContext.configure(connection)):
            getattr(module, direction)()

    service = SQLService()
    async with service.atransaction() as session:
        connection = await session.connection()
        await connection.run_sync(
            lambda sync: InboxEventDTO.__table__.drop(sync, checkfirst=True)
        )
        await connection.run_sync(migrate, "upgrade")
    try:
        yield service
    finally:
        async with service.atransaction() as session:
            connection = await session.connection()
            await connection.run_sync(migrate, "downgrade")
        await service.adispose()


def event(**payload: Any) -> Event:
    return Event.of(identity.USER_REGISTERED, payload, producer="test-producer")


async def stored(service: SQLService, delivered: Event) -> InboxEventDTO:
    async with service.atransaction() as session:
        row = await session.get(InboxEventDTO, delivered.meta.event_id)
        assert row is not None
        return row


async def row_count(service: SQLService) -> int:
    async with service.atransaction() as session:
        return int(
            await session.scalar(select(func.count()).select_from(InboxEventDTO)) or 0
        )


async def age_claim(service: SQLService, delivered: Event, seconds: float) -> None:
    async with service.atransaction() as session:
        row = await session.get(InboxEventDTO, delivered.meta.event_id)
        assert row is not None
        row.state = InboxState.PROCESSING
        row.locked_at = datetime.now(UTC) - timedelta(seconds=seconds)


# The schema the migration deployed


async def test_the_migrated_table_takes_a_claim(worker_inbox: SQLService) -> None:
    """A model that drifted from the migration would fail to insert here."""
    delivered = event(user_id="u1")
    await InboxProcessor(InboxEventDTO).wrap(Recorder())(delivered)
    assert (await stored(worker_inbox, delivered)).state == InboxState.PROCESSED


async def test_the_event_id_round_trips_as_a_uuid(worker_inbox: SQLService) -> None:
    delivered = event(user_id="u1")
    await InboxProcessor(InboxEventDTO).wrap(Recorder())(delivered)
    assert (await stored(worker_inbox, delivered)).event_id == delivered.meta.event_id


async def test_a_claim_is_stored_with_its_offset(worker_inbox: SQLService) -> None:
    """The claim's age is compared against an aware `now`, so it must be aware."""
    delivered = event(user_id="u1")
    await InboxProcessor(InboxEventDTO).wrap(Recorder())(delivered)
    row = await stored(worker_inbox, delivered)
    assert row.locked_at.tzinfo is not None
    assert row.processed_at is not None
    assert row.processed_at.tzinfo is not None
    assert row.locked_at.utcoffset() == timedelta(0)


# Redelivery against real row locking


async def test_an_event_delivered_twice_is_handled_once(
    worker_inbox: SQLService,
) -> None:
    handler = Recorder()
    wrapped = InboxProcessor(InboxEventDTO).wrap(handler)
    delivered = event(user_id="u1")
    await wrapped(delivered)
    await wrapped(delivered)
    assert handler.count == 1


async def test_a_restart_does_not_replay_what_was_already_finished(
    worker_inbox: SQLService,
) -> None:
    handler = Recorder()
    delivered = event(user_id="u1")
    await InboxProcessor(InboxEventDTO).wrap(handler)(delivered)
    await InboxProcessor(InboxEventDTO).wrap(handler)(delivered)
    assert handler.count == 1


async def test_a_claim_still_in_flight_turns_a_duplicate_away(
    worker_inbox: SQLService,
) -> None:
    handler = Recorder()
    wrapped = InboxProcessor(InboxEventDTO).wrap(handler)
    delivered = event(user_id="u1")
    await wrapped(delivered)
    await age_claim(worker_inbox, delivered, 1)
    await wrapped(delivered)
    assert handler.count == 1


async def test_a_claim_whose_owner_died_is_taken_over(
    worker_inbox: SQLService,
) -> None:
    handler = Recorder()
    wrapped = InboxProcessor(InboxEventDTO, claim_timeout_seconds=30).wrap(handler)
    delivered = event(user_id="u1")
    await wrapped(delivered)
    await age_claim(worker_inbox, delivered, 600)
    await wrapped(delivered)
    assert handler.count == 2
    assert (await stored(worker_inbox, delivered)).attempts == 2


async def test_a_failing_handler_is_recorded_and_retried(
    worker_inbox: SQLService,
) -> None:
    succeeding = Recorder()
    delivered = event(user_id="u1")
    with pytest.raises(ValueError, match="handler failed"):
        await InboxProcessor(InboxEventDTO).wrap(Recorder(failing=True))(delivered)
    assert (await stored(worker_inbox, delivered)).state == InboxState.FAILED
    await InboxProcessor(InboxEventDTO).wrap(succeeding)(delivered)
    row = await stored(worker_inbox, delivered)
    assert succeeding.count == 1
    assert row.state == InboxState.PROCESSED
    assert row.last_error is None


async def test_two_replicas_racing_on_one_event_run_the_handler_once(
    worker_inbox: SQLService,
) -> None:
    """The losers see the row that beat them, or lose the insert outright."""
    handler = Recorder()
    wrapped = InboxProcessor(InboxEventDTO).wrap(handler)
    delivered = event(user_id="u1")

    outcomes = await asyncio.gather(
        *(wrapped(delivered) for _ in range(4)), return_exceptions=True
    )
    raised = [o for o in outcomes if isinstance(o, BaseException)]

    assert handler.count == 1
    assert await row_count(worker_inbox) == 1
    assert all(isinstance(error, SQLConflictError) for error in raised)


async def test_concurrent_distinct_events_all_run(worker_inbox: SQLService) -> None:
    handler = Recorder()
    wrapped = InboxProcessor(InboxEventDTO).wrap(handler)
    events = [event(user_id=str(n)) for n in range(6)]

    await asyncio.gather(*(wrapped(delivered) for delivered in events))
    assert handler.count == 6
    assert await row_count(worker_inbox) == 6
