"""`SQLService` against a real Postgres server.

SQLite answers for the shape of a session; only a server answers for what a
transaction means between two connections, what a constraint does when two
writers race for the same value, and which SQLSTATE a failure carries. Those
are what this file pins.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from sqlalchemy import func, select, text

from base.models.view_models.exceptions.sql import (
    SQLBadRequestError,
    SQLConflictError,
)
from base.services.sql.service import SQLService

from .conftest import WidgetStore


def count_rows(service: SQLService, model: type[Any]) -> int:
    return service.query(select(func.count()).select_from(model), lambda r: r.scalar())


async def acount_rows(service: SQLService, model: type[Any]) -> int:
    return await service.aquery(
        select(func.count()).select_from(model), lambda r: r.scalar()
    )


@pytest.mark.integration
def test_the_service_opens_the_configured_postgres(widget: type[Any]) -> None:
    service = SQLService()
    assert service.get_service_name() == "Postgresql"
    assert service.health_check() is True


@pytest.mark.integration
async def test_the_service_opens_postgres_asynchronously(widget: type[Any]) -> None:
    assert await SQLService().ahealth_check() is True


@pytest.mark.integration
def test_a_sync_session_commits_at_the_end_of_the_block(widget: type[Any]) -> None:
    service = SQLService()
    with service._get_session() as session:
        session.add(widget(name="kept"))
    assert count_rows(service, widget) == 1


@pytest.mark.integration
def test_a_sync_session_rolls_a_flushed_row_back_when_the_block_raises(
    widget: type[Any],
) -> None:
    service = SQLService()
    with pytest.raises(RuntimeError):
        with service._get_session() as session:
            session.add(widget(name="ghost"))
            session.flush()
            raise RuntimeError("boom")
    assert count_rows(service, widget) == 0


@pytest.mark.integration
async def test_an_async_session_rolls_a_flushed_row_back_when_the_block_raises(
    widget: type[Any],
) -> None:
    service = SQLService()
    with pytest.raises(RuntimeError):
        async with service._aget_session() as session:
            session.add(widget(name="ghost"))
            await session.flush()
            raise RuntimeError("boom")
    assert await acount_rows(service, widget) == 0


@pytest.mark.integration
async def test_an_uncommitted_row_is_invisible_to_a_second_session(
    widget: type[Any],
) -> None:
    """Two sessions are two connections, and each sees only committed work."""
    service = SQLService()
    async with service._aget_session() as writer:
        writer.add(widget(name="pending"))
        await writer.flush()
        async with service._aget_session() as reader:
            seen = (
                await reader.execute(select(func.count()).select_from(widget))
            ).scalar_one()
    assert seen == 0
    assert await acount_rows(service, widget) == 1


@pytest.mark.integration
async def test_a_committed_row_is_visible_to_the_next_session(
    widget: type[Any],
) -> None:
    service = SQLService()
    async with service._aget_session() as writer:
        writer.add(widget(name="done"))
    async with service._aget_session() as reader:
        seen = (
            await reader.execute(select(func.count()).select_from(widget))
        ).scalar_one()
    assert seen == 1


@pytest.mark.integration
async def test_two_sessions_open_at_once_are_two_separate_backends(
    widget: type[Any],
) -> None:
    service = SQLService()
    async with service._aget_session() as first:
        first_pid = (await first.execute(text("SELECT pg_backend_pid()"))).scalar_one()
        async with service._aget_session() as second:
            second_pid = (
                await second.execute(text("SELECT pg_backend_pid()"))
            ).scalar_one()
    assert first_pid != second_pid


@pytest.mark.integration
async def test_concurrent_writers_all_land(widget: type[Any]) -> None:
    service = SQLService()
    await asyncio.gather(*(service.aupsert_row(widget(name=f"w{i}")) for i in range(8)))
    assert await acount_rows(service, widget) == 8


@pytest.mark.integration
async def test_two_writers_racing_for_one_unique_value_leave_exactly_one_row(
    widget: type[Any],
) -> None:
    """The server decides, and the loser is reported as a conflict."""
    service = SQLService()
    results = await asyncio.gather(
        service.aupsert_row(widget(name="contested")),
        service.aupsert_row(widget(name="contested")),
        return_exceptions=True,
    )
    failures = [r for r in results if isinstance(r, BaseException)]
    assert len(failures) == 1
    assert isinstance(failures[0], SQLConflictError)
    assert await acount_rows(service, widget) == 1


@pytest.mark.integration
def test_a_unique_violation_carries_the_server_s_own_code(widget: type[Any]) -> None:
    service = SQLService()
    service.upsert_row(widget(name="taken"))
    with pytest.raises(SQLConflictError) as caught:
        service.upsert_row(widget(name="taken"))
    assert getattr(caught.value.provider_error.orig, "pgcode", "") == "23505"


@pytest.mark.integration
async def test_an_async_unique_violation_carries_the_server_s_own_code(
    widget: type[Any],
) -> None:
    service = SQLService()
    await service.aupsert_row(widget(name="taken"))
    with pytest.raises(SQLConflictError) as caught:
        await service.aupsert_row(widget(name="taken"))
    assert getattr(caught.value.provider_error.orig, "sqlstate", "") == "23505"


@pytest.mark.integration
def test_an_unknown_relation_is_reported_as_a_bad_request(widget: type[Any]) -> None:
    with pytest.raises(SQLBadRequestError, match="does not exist"):
        SQLService().query(
            select(text("*")).select_from(text("no_such_table")), lambda r: r.all()
        )


@pytest.mark.integration
def test_a_caller_driven_transaction_commits_several_writes_together(
    widget: type[Any],
) -> None:
    service = SQLService()
    with service.transaction() as session:
        session.add(widget(name="a"))
        session.add(widget(name="b"))
    assert count_rows(service, widget) == 2


@pytest.mark.integration
def test_a_caller_driven_transaction_discards_all_of_them_on_a_failure(
    widget: type[Any],
) -> None:
    service = SQLService()
    service.upsert_row(widget(name="taken"))
    with pytest.raises(SQLConflictError):
        with service.transaction() as session:
            session.add(widget(name="fresh"))
            session.add(widget(name="taken"))
    assert count_rows(service, widget) == 1


@pytest.mark.integration
async def test_an_async_caller_driven_transaction_discards_all_of_them_on_a_failure(
    widget: type[Any],
) -> None:
    service = SQLService()
    await service.aupsert_row(widget(name="taken"))
    with pytest.raises(SQLConflictError):
        async with service.atransaction() as session:
            session.add(widget(name="fresh"))
            session.add(widget(name="taken"))
    assert await acount_rows(service, widget) == 1


@pytest.mark.integration
def test_a_row_lock_held_in_a_transaction_is_visible_to_the_server(
    widget: type[Any],
) -> None:
    service = SQLService()
    row = service.upsert_row(widget(name="locked"))
    with service.transaction() as session:
        locked = session.execute(
            select(widget).where(widget.id == row.id).with_for_update()
        ).scalar_one()
        held = session.execute(
            text(
                "SELECT count(*) FROM pg_locks "
                "WHERE locktype = 'transactionid' AND granted"
            )
        ).scalar_one()
    assert locked.name == "locked"
    assert held >= 1


@pytest.mark.integration
def test_dispose_releases_the_sync_pool_and_a_later_call_reopens_it(
    widget: type[Any],
) -> None:
    service = SQLService()
    assert service.health_check() is True
    service.dispose()
    assert service._client._sync_engine is None
    assert service.health_check() is True


@pytest.mark.integration
async def test_adispose_releases_the_async_pool_and_a_later_call_reopens_it(
    widget: type[Any],
) -> None:
    service = SQLService()
    assert await service.ahealth_check() is True
    await service.adispose()
    assert service._client._async_engine is None
    assert await service.ahealth_check() is True


@pytest.mark.integration
def test_the_pool_is_sized_from_settings(widget: type[Any]) -> None:
    service = SQLService()
    service.health_check()
    assert service._client._sync_engine is not None
    assert service._client._sync_engine.pool.size() == 10


@pytest.mark.integration
def test_the_connection_carries_the_configured_statement_timeout(
    widget: type[Any],
) -> None:
    timeout = SQLService().query(
        select(text("current_setting('statement_timeout')")), lambda r: r.scalar()
    )
    assert timeout == "5min"


@pytest.mark.integration
def test_execute_in_transaction_keeps_a_delete_only_if_the_insert_lands(
    widget: type[Any],
) -> None:
    from sqlalchemy import delete

    service = SQLService()
    service.upsert_rows([widget(name=f"w{i}", size=i) for i in range(4)])
    with pytest.raises(SQLConflictError):
        service.execute_in_transaction(
            statements=[delete(widget).where(widget.size >= 2)],
            rows_to_add=[widget(name="w0")],
        )
    assert count_rows(service, widget) == 4


@pytest.mark.integration
def test_a_value_longer_than_the_column_is_refused_by_the_server(
    store: WidgetStore, widget: type[Any]
) -> None:
    """Unlike SQLite, Postgres enforces the declared length."""
    with pytest.raises(SQLBadRequestError, match="too long"):
        store.add(widget(name="x" * 5000))
