"""`SQLStore` over a real Postgres table.

The unit suite proves the store's logic on SQLite. What is left for a server
is what SQLite cannot represent: timestamps that keep their UTC offset, a
unique index that a soft-deleted row still occupies, and `ilike` collation
over text the database stores itself.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select

from base.models.view_models.database import FilteredPaginationParams
from base.services.sql.service import SQLService
from base.tools.errors import ErrorsException

from .conftest import WidgetStore

UNICODE_TEXT: str = "héllo 😀 中文 ، مرحبا"


@pytest.fixture
def seeded(store: WidgetStore, widget: type[Any]) -> list[Any]:
    return store.add_many(
        [
            widget(name="alpha", note="first", size=1),
            widget(name="beta", note="second", size=2),
            widget(name="gamma", note="third", size=3),
        ]
    )


def raw_count(model: type[Any]) -> int:
    """Rows in the table, soft-deleted ones included."""
    return SQLService().query(
        select(func.count()).select_from(model), lambda r: r.scalar()
    )


@pytest.mark.integration
def test_the_full_sync_round_trip_works_against_the_server(
    store: WidgetStore, widget: type[Any]
) -> None:
    row = store.add(widget(name="alpha", note="first"))
    assert store.get(row.id).note == "first"

    store.edit(row.id, widget(name="alpha", note="rewritten"))
    assert store.get(row.id).note == "rewritten"

    assert store.delete(row.id) is True
    assert store.get_all() == []


@pytest.mark.integration
async def test_the_full_async_round_trip_works_against_the_server(
    store: WidgetStore, widget: type[Any]
) -> None:
    row = await store.aadd(widget(name="alpha", note="first"))
    assert (await store.aget(row.id)).note == "first"

    await store.aedit(row.id, widget(name="alpha", note="rewritten"))
    assert (await store.aget(row.id)).note == "rewritten"

    assert await store.adelete(row.id) is True
    assert await store.aget_all() == []


@pytest.mark.integration
def test_add_many_writes_every_row_in_one_transaction(
    store: WidgetStore, seeded: list[Any]
) -> None:
    assert sorted(row.name for row in store.get_all()) == ["alpha", "beta", "gamma"]


@pytest.mark.integration
def test_a_failure_part_way_through_add_many_writes_none_of_it(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    with pytest.raises(ErrorsException):
        store.add_many([widget(name="delta"), widget(name="alpha")])
    assert raw_count(widget) == 3


@pytest.mark.integration
def test_created_at_is_timezone_aware_utc(
    store: WidgetStore, seeded: list[Any]
) -> None:
    created = store.get(seeded[0].id).created_at
    assert created.tzinfo is not None
    assert created.utcoffset() == timedelta(0)


@pytest.mark.integration
def test_updated_at_is_timezone_aware_utc(
    store: WidgetStore, seeded: list[Any]
) -> None:
    updated = store.get(seeded[0].id).updated_at
    assert updated.tzinfo is not None
    assert updated.utcoffset() == timedelta(0)


@pytest.mark.integration
def test_both_timestamps_are_set_close_to_now(
    store: WidgetStore, seeded: list[Any]
) -> None:
    row = store.get(seeded[0].id)
    assert abs(datetime.now(UTC) - row.created_at) < timedelta(minutes=5)
    assert abs(datetime.now(UTC) - row.updated_at) < timedelta(minutes=5)


@pytest.mark.integration
def test_an_edit_moves_updated_at_and_leaves_created_at(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    before = store.get(seeded[0].id)
    edited = store.edit(seeded[0].id, widget(name="alpha", note="changed"))
    assert edited is not None
    assert edited.created_at == before.created_at
    assert edited.updated_at > before.updated_at


@pytest.mark.integration
async def test_an_async_edit_moves_updated_at(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    before = await store.aget(seeded[0].id)
    edited = await store.aedit(seeded[0].id, widget(name="alpha", note="changed"))
    assert edited is not None
    assert edited.updated_at > before.updated_at


@pytest.mark.integration
def test_deleted_at_is_timezone_aware_utc(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    store.delete(seeded[0].id)
    stamp = SQLService().query(
        select(widget.deleted_at).where(widget.id == seeded[0].id),
        lambda r: r.scalar(),
    )
    assert stamp is not None
    assert stamp.utcoffset() == timedelta(0)


@pytest.mark.integration
def test_delete_leaves_the_row_in_the_table_and_out_of_every_read(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    store.delete(seeded[0].id)
    assert raw_count(widget) == 3
    assert sorted(row.name for row in store.get_all()) == ["beta", "gamma"]
    assert store.get_many([row.id for row in seeded]) != []
    with pytest.raises(ErrorsException):
        store.get(seeded[0].id)


@pytest.mark.integration
def test_a_soft_deleted_unique_value_still_occupies_the_index(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    """The constraint belongs to the server, and soft delete never reaches it."""
    store.delete(seeded[0].id)
    with pytest.raises(ErrorsException, match="already exists"):
        store.add(widget(name="alpha"))


@pytest.mark.integration
async def test_a_soft_deleted_unique_value_still_occupies_the_index_asynchronously(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    await store.adelete(seeded[0].id)
    with pytest.raises(ErrorsException, match="already exists"):
        await store.aadd(widget(name="alpha"))


@pytest.mark.integration
def test_protected_columns_survive_an_edit_that_tries_to_rewrite_them(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    before = store.get(seeded[0].id)
    patch = widget(name="alpha", note="changed")
    patch.created_at = datetime(1999, 1, 1, tzinfo=UTC)
    patch.deleted_at = datetime(1999, 1, 1, tzinfo=UTC)

    edited = store.edit(seeded[0].id, patch)
    assert edited is not None
    assert edited.id == before.id
    assert edited.created_at == before.created_at
    assert edited.deleted_at is None


@pytest.mark.integration
def test_search_is_case_insensitive_on_the_server(
    store: WidgetStore, seeded: list[Any]
) -> None:
    result = store.get_all_paged(FilteredPaginationParams(search="SECOND"))
    assert [row.name for row in result.data] == ["beta"]


@pytest.mark.integration
def test_paging_reports_the_window_and_the_whole_total(
    store: WidgetStore, seeded: list[Any]
) -> None:
    result = store.get_all_paged(FilteredPaginationParams(page=2, size=2))
    assert len(result.data) == 1
    assert (result.total, result.total_pages) == (3, 2)


@pytest.mark.integration
async def test_async_paging_reports_the_window_and_the_whole_total(
    store: WidgetStore, seeded: list[Any]
) -> None:
    result = await store.aget_all_paged(FilteredPaginationParams(page=2, size=2))
    assert len(result.data) == 1
    assert (result.total, result.total_pages) == (3, 2)


@pytest.mark.integration
def test_a_date_range_narrows_on_the_server_s_own_timestamps(
    store: WidgetStore, seeded: list[Any]
) -> None:
    now = datetime.now(UTC)
    inside = store.get_all_paged(
        FilteredPaginationParams(
            from_date=now - timedelta(hours=1), to_date=now + timedelta(hours=1)
        )
    )
    outside = store.get_all_paged(
        FilteredPaginationParams(from_date=now + timedelta(hours=1))
    )
    assert inside.total == 3
    assert outside.total == 0


@pytest.mark.integration
def test_unicode_round_trips_through_the_server(
    store: WidgetStore, widget: type[Any]
) -> None:
    row = store.add(widget(name=UNICODE_TEXT, note=UNICODE_TEXT))
    assert store.get(row.id).name == UNICODE_TEXT
    assert store.get_all_paged(FilteredPaginationParams(search="中文")).total == 1


@pytest.mark.integration
async def test_concurrent_adds_through_one_store_all_land(
    store: WidgetStore, widget: type[Any]
) -> None:
    await asyncio.gather(*(store.aadd(widget(name=f"w{i}")) for i in range(8)))
    assert len(await store.aget_all()) == 8


@pytest.mark.integration
async def test_two_stores_over_one_table_see_each_other_s_writes(
    widget: type[Any],
) -> None:
    first, second = WidgetStore(widget), WidgetStore(widget)
    row = await first.aadd(widget(name="alpha"))
    assert (await second.aget(row.id)).name == "alpha"
    await second.adelete(row.id)
    assert await first.aget_all() == []


@pytest.mark.integration
def test_editing_a_datetime_column_keeps_it_a_datetime(
    store: WidgetStore, widget: type[Any]
) -> None:
    """The backend-agnostic half: the column type reaches Postgres unchanged."""
    row = store.add(widget(name="alpha"))
    moment = datetime(2030, 1, 1, tzinfo=UTC)

    edited = store.edit(row.id, widget(name="alpha", expires_at=moment))
    assert edited is not None
    assert store.get(row.id).expires_at == moment


@pytest.mark.integration
def test_deleting_twice_succeeds_once(store: WidgetStore, seeded: list[Any]) -> None:
    assert store.delete(seeded[0].id) is True
    assert store.delete(seeded[0].id) is False
