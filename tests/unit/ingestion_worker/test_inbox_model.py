"""The inbox table this worker owns, and the migration that creates it.

The claim lives in a database the worker owns alone, so the shape of that
table is the contract between the running service and the schema that was
deployed. Drift between the two is the failure this file is here to catch —
along with the timestamps, which the claim's age is compared against and
which must therefore come back aware whichever backend stored them.
"""

from __future__ import annotations

import importlib.util
import pathlib
import types
import uuid
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
import sqlalchemy as sa

from base.models.definitions.messaging import InboxState
from base.models.orm import UtcDateTime
from base.models.view_models.exceptions.sql import SQLConflictError
from base.services.messaging.protocol import InboxRow
from base.services.sql.service import SQLService
from ingestion_worker.models import WorkerBase
from ingestion_worker.models.inbox import InboxEventDTO
from tests.unit.ingestion_worker.conftest import user_registered

REVISION = (
    pathlib.Path(__file__).resolve().parents[3]
    / "services/ingestion_worker/alembic/versions/2026_11_02_0001-worker_inbox.py"
)

TABLE: sa.Table = InboxEventDTO.__table__


def reflected(connection: sa.Connection) -> dict[str, dict[str, Any]]:
    """Every column of `inbox_event` as the database now has it."""
    return {
        column["name"]: {
            "type": str(column["type"]),
            "nullable": column["nullable"],
        }
        for column in sa.inspect(connection).get_columns("inbox_event")
    }


def indexes(connection: sa.Connection) -> list[tuple[str, list[str]]]:
    return [
        (index["name"], index["column_names"])
        for index in sa.inspect(connection).get_indexes("inbox_event")
    ]


def revision_module() -> types.ModuleType:
    """The migration, loaded from a versions directory that is not a package."""
    spec = importlib.util.spec_from_file_location("worker_inbox_revision", REVISION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def migrate(connection: sa.Connection, direction: str) -> None:
    """Run the revision for real, against the connection a test opened."""
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    with Operations.context(MigrationContext.configure(connection)):
        getattr(revision_module(), direction)()


async def insert(**values: Any) -> uuid.UUID:
    """One claim, written the way the inbox writes one."""
    row = InboxEventDTO(
        event_id=values.pop("event_id", None) or uuid.uuid4(),
        topic=values.pop("topic", "test.topic"),
        state=values.pop("state", InboxState.PROCESSING),
        locked_at=values.pop("locked_at", datetime.now(UTC)),
        **values,
    )
    event_id = row.event_id
    async with SQLService().atransaction() as session:
        session.add(row)
    return event_id


async def read(event_id: uuid.UUID) -> InboxEventDTO:
    async with SQLService().atransaction() as session:
        row = await session.get(InboxEventDTO, event_id)
        assert row is not None
        return row


# The table


def test_the_worker_owns_one_table_and_it_is_the_inbox() -> None:
    assert set(WorkerBase.metadata.tables) == {"inbox_event"}


def test_the_event_id_is_the_primary_key() -> None:
    """Which is what makes a second delivery of one event a duplicate insert."""
    assert [column.name for column in TABLE.primary_key] == ["event_id"]


def test_the_columns_are_the_ones_the_inbox_protocol_names() -> None:
    assert set(TABLE.columns.keys()) == set(InboxRow.__annotations__)


@pytest.mark.parametrize(
    ("column", "nullable"),
    [
        ("event_id", False),
        ("topic", False),
        ("state", False),
        ("locked_at", False),
        ("attempts", False),
        ("processed_at", True),
        ("last_error", True),
    ],
)
def test_only_the_columns_an_unfinished_event_has_not_reached_are_nullable(
    column: str, nullable: bool
) -> None:
    assert TABLE.columns[column].nullable is nullable


def test_a_topic_column_long_enough_for_the_topics_in_use() -> None:
    assert TABLE.columns["topic"].type.length == 255


def test_the_state_column_holds_the_longest_state() -> None:
    assert TABLE.columns["state"].type.length == 32
    assert max(len(state.value) for state in InboxState) <= 32


def test_an_error_is_stored_as_text_rather_than_truncated_by_the_column() -> None:
    assert isinstance(TABLE.columns["last_error"].type, sa.Text)


def test_the_index_is_the_one_a_sweep_for_stale_claims_reads() -> None:
    indexes = {index.name: [c.name for c in index.columns] for index in TABLE.indexes}
    assert indexes == {"idx_inbox_state_locked": ["state", "locked_at"]}


def test_both_timestamps_normalise_to_utc() -> None:
    for column in ("locked_at", "processed_at"):
        assert isinstance(TABLE.columns[column].type, UtcDateTime)


# Timestamps


async def test_a_timestamp_written_aware_reads_back_aware(worker_settings: Any) -> None:
    """SQLite keeps no offset, so the column has to put it back."""
    del worker_settings
    row = await read(await insert(locked_at=datetime.now(UTC)))
    assert row.locked_at.tzinfo is not None
    assert row.locked_at.utcoffset() == timedelta(0)


async def test_a_timestamp_written_in_another_zone_is_stored_as_utc(
    worker_settings: Any,
) -> None:
    del worker_settings
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=5)))
    row = await read(await insert(locked_at=moment))
    assert row.locked_at == moment
    assert row.locked_at.hour == 7


async def test_a_timestamp_written_naive_is_read_as_utc(worker_settings: Any) -> None:
    del worker_settings
    row = await read(await insert(locked_at=datetime(2026, 1, 1, 12, 0)))
    assert row.locked_at == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


async def test_a_processed_timestamp_reads_back_aware_too(
    worker_settings: Any,
) -> None:
    del worker_settings
    row = await read(
        await insert(state=InboxState.PROCESSED, processed_at=datetime.now(UTC))
    )
    assert row.processed_at is not None
    assert row.processed_at.tzinfo is not None


async def test_an_unprocessed_row_keeps_no_timestamp(worker_settings: Any) -> None:
    del worker_settings
    assert (await read(await insert())).processed_at is None


# Defaults


async def test_a_new_claim_counts_as_the_first_attempt(worker_settings: Any) -> None:
    del worker_settings
    row = await read(await insert(topic=user_registered().topic))
    assert row.attempts == 1
    assert row.last_error is None


async def test_one_event_cannot_be_claimed_twice(worker_settings: Any) -> None:
    """The primary key is what makes a concurrent duplicate lose rather than run."""
    del worker_settings
    event_id = uuid.uuid4()
    await insert(event_id=event_id)
    with pytest.raises(SQLConflictError):
        await insert(event_id=event_id)


async def test_an_error_longer_than_a_line_is_stored_whole(
    worker_settings: Any,
) -> None:
    """The inbox truncates at a thousand characters, so the column must take them."""
    del worker_settings
    message = "x" * 1_000
    row = await read(await insert(state=InboxState.FAILED, last_error=message))
    assert row.last_error == message


async def test_a_topic_at_the_column_limit_is_stored(worker_settings: Any) -> None:
    del worker_settings
    topic = "t." + "x" * 253
    assert (await read(await insert(topic=topic))).topic == topic


async def test_a_unicode_topic_is_stored(worker_settings: Any) -> None:
    del worker_settings
    topic = "tópico.ünicode.🎉"
    assert (await read(await insert(topic=topic))).topic == topic


# The migration


def test_the_inbox_migration_is_the_first_one_this_service_has() -> None:
    module = revision_module()
    assert module.revision == "workerinbox001"
    assert module.down_revision is None


def test_the_migration_builds_the_table_the_model_describes() -> None:
    """Drift here is a service that starts and then cannot write a claim."""
    migrated = sa.create_engine("sqlite://")
    declared = sa.create_engine("sqlite://")
    with migrated.begin() as connection:
        migrate(connection, "upgrade")
        from_migration = reflected(connection)
        from_migration_indexes = indexes(connection)
    with declared.begin() as connection:
        WorkerBase.metadata.create_all(connection)
        from_model = reflected(connection)
        from_model_indexes = indexes(connection)

    assert from_migration == from_model
    assert from_migration_indexes == from_model_indexes


def test_the_migration_can_be_rolled_back() -> None:
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        migrate(connection, "upgrade")
        assert sa.inspect(connection).get_table_names() == ["inbox_event"]
        migrate(connection, "downgrade")
        assert sa.inspect(connection).get_table_names() == []
