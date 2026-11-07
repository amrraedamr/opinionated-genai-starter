"""The awkward calls: empty inputs, repeats, outsized values, partial failures.

A store is mostly exercised on the happy path, so the edges are where its
behaviour is least known and most surprising. Each test here states what one
of those edges currently does, including the two that look like defects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select

from base.models.view_models.exceptions.sql import SQLUnexpectedError
from base.services.sql.service import SQLService
from base.tools.errors import ErrorsException

from .conftest import WidgetStore

LONG_TEXT: str = "x" * 5000
UNICODE_TEXT: str = "héllo 😀 中文 ، مرحبا"


def test_add_many_of_nothing_writes_nothing_and_returns_nothing(
    store: WidgetStore,
) -> None:
    assert store.add_many([]) == []
    assert store.get_all() == []


async def test_aadd_many_of_nothing_writes_nothing_and_returns_nothing(
    store: WidgetStore,
) -> None:
    assert await store.aadd_many([]) == []
    assert await store.aget_all() == []


def test_get_many_of_nothing_is_an_empty_list(
    store: WidgetStore, widget: type[Any]
) -> None:
    store.add(widget(name="alpha"))
    assert store.get_many([]) == []


async def test_aget_many_of_nothing_is_an_empty_list(
    store: WidgetStore, widget: type[Any]
) -> None:
    await store.aadd(widget(name="alpha"))
    assert await store.aget_many([]) == []


def test_editing_a_row_that_is_not_there_returns_none(
    store: WidgetStore, widget: type[Any]
) -> None:
    assert store.edit(9999, widget(name="ghost")) is None


async def test_aediting_a_row_that_is_not_there_returns_none(
    store: WidgetStore, widget: type[Any]
) -> None:
    assert await store.aedit(9999, widget(name="ghost")) is None


def test_editing_a_row_that_is_not_there_writes_nothing(
    store: WidgetStore, widget: type[Any]
) -> None:
    store.edit(9999, widget(name="ghost"))
    assert store.get_all() == []


def test_deleting_twice_succeeds_once(store: WidgetStore, widget: type[Any]) -> None:
    row = store.add(widget(name="alpha"))
    assert store.delete(row.id) is True
    assert store.delete(row.id) is False


async def test_adeleting_twice_succeeds_once(
    store: WidgetStore, widget: type[Any]
) -> None:
    row = await store.aadd(widget(name="alpha"))
    assert await store.adelete(row.id) is True
    assert await store.adelete(row.id) is False


def test_deleting_a_row_that_never_existed_reports_false(store: WidgetStore) -> None:
    assert store.delete(9999) is False


async def test_adeleting_a_row_that_never_existed_reports_false(
    store: WidgetStore,
) -> None:
    assert await store.adelete(9999) is False


def test_a_string_far_longer_than_the_declared_length_round_trips(
    store: WidgetStore, widget: type[Any]
) -> None:
    """SQLite does not enforce `String(500)`; the value comes back whole."""
    row = store.add(widget(name="long", note=LONG_TEXT))
    assert store.get(row.id).note == LONG_TEXT


def test_unicode_round_trips_unchanged(store: WidgetStore, widget: type[Any]) -> None:
    row = store.add(widget(name=UNICODE_TEXT, note=UNICODE_TEXT))
    assert store.get(row.id).name == UNICODE_TEXT
    assert store.get(UNICODE_TEXT, "name").id == row.id


def test_unicode_is_searchable(store: WidgetStore, widget: type[Any]) -> None:
    from base.models.view_models.database import FilteredPaginationParams

    store.add(widget(name=UNICODE_TEXT))
    result = store.get_all_paged(FilteredPaginationParams(search="中文"))
    assert result.total == 1


def test_a_failure_part_way_through_add_many_writes_none_of_it(
    store: WidgetStore, widget: type[Any]
) -> None:
    store.add(widget(name="taken"))
    with pytest.raises(ErrorsException):
        store.add_many(
            [widget(name="first"), widget(name="taken"), widget(name="last")]
        )
    assert [row.name for row in store.get_all()] == ["taken"]


async def test_a_failure_part_way_through_aadd_many_writes_none_of_it(
    store: WidgetStore, widget: type[Any]
) -> None:
    await store.aadd(widget(name="taken"))
    with pytest.raises(ErrorsException):
        await store.aadd_many(
            [widget(name="first"), widget(name="taken"), widget(name="last")]
        )
    assert [row.name for row in await store.aget_all()] == ["taken"]


def test_two_stores_over_one_table_see_each_other_s_writes(widget: type[Any]) -> None:
    first, second = WidgetStore(widget), WidgetStore(widget)
    row = first.add(widget(name="alpha"))
    assert second.get(row.id).name == "alpha"
    second.delete(row.id)
    assert first.get_all() == []


async def test_two_stores_over_one_table_see_each_other_s_async_writes(
    widget: type[Any],
) -> None:
    first, second = WidgetStore(widget), WidgetStore(widget)
    row = await first.aadd(widget(name="alpha"))
    assert (await second.aget(row.id)).name == "alpha"


def test_adding_the_same_object_twice_stores_one_row(
    store: WidgetStore, widget: type[Any]
) -> None:
    """The second `add` sees a row SQLAlchemy already tracks, so it updates it."""
    entity = widget(name="alpha")
    first = store.add(entity)
    second = store.add(entity)
    assert first.id == second.id
    assert (
        SQLService().query(
            select(func.count()).select_from(widget), lambda r: r.scalar()
        )
        == 1
    )


def test_a_lookup_value_of_the_wrong_type_simply_matches_nothing(
    store: WidgetStore, widget: type[Any]
) -> None:
    store.add(widget(name="alpha"))
    assert store.get_many(["not-an-id"]) == []


def test_editing_a_datetime_column_fails_because_to_dict_hands_back_a_string(
    store: WidgetStore, widget: type[Any]
) -> None:
    """A defect, pinned as it stands.

    `_apply` round-trips the patch through `SerializableMixin.to_dict`, which
    turns a `datetime` into an ISO string, and `update_from_dict` writes that
    string straight onto the column. SQLite refuses it outright, so on the
    default backend a model with an unprotected datetime column cannot be
    edited at all once that column is set. Postgres coerces the string
    instead, which is why this only bites one backend — see the integration
    suite for the other half.
    """
    row = store.add(widget(name="alpha"))
    with pytest.raises(SQLUnexpectedError, match="datetime"):
        store.edit(
            row.id,
            widget(name="alpha", expires_at=datetime(2030, 1, 1, tzinfo=UTC)),
        )


def test_editing_leaves_a_datetime_column_alone_while_it_stays_unset(
    store: WidgetStore, widget: type[Any]
) -> None:
    row = store.add(widget(name="alpha"))
    edited = store.edit(row.id, widget(name="alpha2"))
    assert edited is not None
    assert edited.expires_at is None
