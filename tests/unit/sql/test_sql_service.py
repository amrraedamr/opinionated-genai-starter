"""Session lifecycle and transaction boundaries of `SQLService`.

Every method on `AbstractSQLService` opens its own session, so what a session
does at its edges — commit on a clean exit, roll back and close on a failure —
is the single guarantee the whole store layer rests on. These run on SQLite
because none of it is engine-specific.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from base.models.view_models.exceptions.sql import SQLBadRequestError, SQLError
from base.services.configuration.mixins import SQLSettingsMixin
from base.services.sql.service import SQLService
from base.tools.errors import ServiceHealthCheckError
from tests.conftest import MakeSettings


def count_rows(service: SQLService, model: type[Any]) -> int:
    return service.query(select(func.count()).select_from(model), lambda r: r.scalar())


def test_a_sync_session_is_a_sqlalchemy_session(widget: type[Any]) -> None:
    with SQLService()._get_session() as session:
        assert isinstance(session, Session)


async def test_an_async_session_is_a_sqlalchemy_async_session(
    widget: type[Any],
) -> None:
    async with SQLService()._aget_session() as session:
        assert isinstance(session, AsyncSession)


def test_a_sync_session_commits_what_it_was_given(widget: type[Any]) -> None:
    service = SQLService()
    with service._get_session() as session:
        session.add(widget(name="kept"))
    assert count_rows(service, widget) == 1


async def test_an_async_session_commits_what_it_was_given(widget: type[Any]) -> None:
    service = SQLService()
    async with service._aget_session() as session:
        session.add(widget(name="kept"))
    assert (
        await service.aquery(
            select(func.count()).select_from(widget), lambda r: r.scalar()
        )
        == 1
    )


def test_a_sync_session_rolls_back_a_flushed_row_when_the_block_raises(
    widget: type[Any],
) -> None:
    service = SQLService()
    with pytest.raises(RuntimeError, match="boom"):
        with service._get_session() as session:
            session.add(widget(name="ghost"))
            session.flush()
            raise RuntimeError("boom")
    assert count_rows(service, widget) == 0


async def test_an_async_session_rolls_back_a_flushed_row_when_the_block_raises(
    widget: type[Any],
) -> None:
    service = SQLService()
    with pytest.raises(RuntimeError, match="boom"):
        async with service._aget_session() as session:
            session.add(widget(name="ghost"))
            await session.flush()
            raise RuntimeError("boom")
    assert (
        await service.aquery(
            select(func.count()).select_from(widget), lambda r: r.scalar()
        )
        == 0
    )


def test_a_sync_session_is_closed_after_the_block_raises(widget: type[Any]) -> None:
    escaped: dict[str, Session] = {}
    with pytest.raises(RuntimeError):
        with SQLService()._get_session() as session:
            escaped["session"] = session
            session.add(widget(name="ghost"))
            session.flush()
            raise RuntimeError("boom")

    closed = escaped["session"]
    assert closed.in_transaction() is False
    assert len(closed.identity_map) == 0
    assert not closed.new


async def test_an_async_session_is_closed_after_the_block_raises(
    widget: type[Any],
) -> None:
    escaped: dict[str, AsyncSession] = {}
    with pytest.raises(RuntimeError):
        async with SQLService()._aget_session() as session:
            escaped["session"] = session
            session.add(widget(name="ghost"))
            await session.flush()
            raise RuntimeError("boom")

    closed = escaped["session"]
    assert closed.in_transaction() is False
    assert len(closed.sync_session.identity_map) == 0


def test_a_sync_session_keeps_nothing_a_failed_flush_produced(
    widget: type[Any],
) -> None:
    """A duplicate inside a block loses the whole block, not just the duplicate."""
    service = SQLService()
    service.upsert_row(widget(name="first"))
    with pytest.raises(SQLAlchemyError):
        with service._get_session() as session:
            session.add(widget(name="second"))
            session.add(widget(name="first"))
            session.flush()
    assert count_rows(service, widget) == 1


def test_dispose_drops_the_sync_engine_and_a_later_call_rebuilds_it(
    widget: type[Any],
) -> None:
    service = SQLService()
    assert service.health_check() is True
    assert service._client._sync_engine is not None

    service.dispose()
    assert service._client._sync_engine is None
    assert service.health_check() is True


async def test_adispose_drops_the_async_engine_and_a_later_call_rebuilds_it(
    widget: type[Any],
) -> None:
    service = SQLService()
    assert await service.ahealth_check() is True
    assert service._client._async_engine is not None

    await service.adispose()
    assert service._client._async_engine is None
    assert await service.ahealth_check() is True


def test_dispose_leaves_the_async_engine_alone(widget: type[Any]) -> None:
    """The two engines are disposed independently, by their own method."""
    service = SQLService()
    service._client._ensure_async_initialized()
    service.dispose()
    assert service._client._async_engine is not None


async def test_close_all_drops_both_engines(widget: type[Any]) -> None:
    service = SQLService()
    service.health_check()
    await service.ahealth_check()

    await service._client.close_all()
    assert service._client._sync_engine is None
    assert service._client._async_engine is None


def test_dispose_on_an_unopened_service_is_a_no_op(sqlite_settings: Any) -> None:
    service = SQLService()
    service.dispose()
    assert service._client._sync_engine is None


async def test_adispose_on_an_unopened_service_is_a_no_op(sqlite_settings: Any) -> None:
    service = SQLService()
    await service.adispose()
    assert service._client._async_engine is None


def test_a_caller_driven_transaction_commits_as_one_unit(widget: type[Any]) -> None:
    service = SQLService()
    with service.transaction() as session:
        session.add(widget(name="a"))
        session.add(widget(name="b"))
    assert count_rows(service, widget) == 2


async def test_a_caller_driven_async_transaction_commits_as_one_unit(
    widget: type[Any],
) -> None:
    service = SQLService()
    async with service.atransaction() as session:
        session.add(widget(name="a"))
        session.add(widget(name="b"))
    assert (
        await service.aquery(
            select(func.count()).select_from(widget), lambda r: r.scalar()
        )
        == 2
    )


def test_a_transaction_reports_a_failure_as_this_backend_s_family(
    widget: type[Any],
) -> None:
    with pytest.raises(SQLBadRequestError) as caught:
        with SQLService().transaction() as session:
            session.execute(text("SELECT * FROM table_that_is_not_there"))
    assert isinstance(caught.value, SQLError)


async def test_an_async_transaction_reports_a_failure_as_this_backend_s_family(
    widget: type[Any],
) -> None:
    with pytest.raises(SQLBadRequestError) as caught:
        async with SQLService().atransaction() as session:
            await session.execute(text("SELECT * FROM table_that_is_not_there"))
    assert isinstance(caught.value, SQLError)


def test_a_transaction_lets_a_caller_s_own_exception_through(widget: type[Any]) -> None:
    """Only SQLAlchemy failures are translated; the caller's are their own."""
    with pytest.raises(ZeroDivisionError):
        with SQLService().transaction() as session:
            session.add(widget(name="ghost"))
            1 / 0
    assert count_rows(SQLService(), widget) == 0


def test_health_check_passes_against_a_reachable_database(widget: type[Any]) -> None:
    assert SQLService().health_check() is True


async def test_ahealth_check_passes_against_a_reachable_database(
    widget: type[Any],
) -> None:
    assert await SQLService().ahealth_check() is True


def test_health_check_names_the_service_when_the_database_is_unreachable(
    make_settings: MakeSettings,
) -> None:
    make_settings(
        SQLSettingsMixin,
        sql_backend="sqlite",
        sqlite_db_path="no/such/directory/app.db",
    )
    with pytest.raises(ServiceHealthCheckError, match="Sqlite health check failed"):
        SQLService().health_check()


async def test_ahealth_check_names_the_service_when_the_database_is_unreachable(
    make_settings: MakeSettings,
) -> None:
    make_settings(
        SQLSettingsMixin,
        sql_backend="sqlite",
        sqlite_db_path="no/such/directory/app.db",
    )
    with pytest.raises(ServiceHealthCheckError, match="Sqlite health check failed"):
        await SQLService().ahealth_check()


def test_the_service_name_is_the_backend(sqlite_settings: Any) -> None:
    assert SQLService().get_service_name() == "Sqlite"


def test_the_service_is_one_instance_per_process(sqlite_settings: Any) -> None:
    assert SQLService() is SQLService()
