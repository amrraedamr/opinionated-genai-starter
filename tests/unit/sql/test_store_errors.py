"""How a failed store call is reported, and in whose vocabulary.

Two layers translate here. `SQLService` turns SQLAlchemy's exceptions into the
SQL family, and `AbstractStore` turns the one it cares about — a duplicate key
— into the API-facing `DatabaseErrors`. A caller sees the second; a store
author sees the first. Both are pinned, including the case the store does not
translate at all.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

import pytest

from base.models.definitions.errors import DatabaseErrors
from base.models.view_models.exceptions.sql import SQLConflictError
from base.tools.errors import ErrorsException

from .conftest import WidgetStore


@pytest.fixture
def existing(store: WidgetStore, widget: type[Any]) -> Any:
    return store.add(widget(name="taken"))


def test_the_sql_store_names_the_conflict_error_as_its_duplicate(
    store: WidgetStore,
) -> None:
    assert store._duplicate_error_types() == (SQLConflictError,)


def test_add_reports_a_duplicate_as_a_conflict(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    with pytest.raises(ErrorsException) as caught:
        store.add(widget(name="taken"))
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION


async def test_aadd_reports_a_duplicate_as_a_conflict(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await store.aadd(widget(name="taken"))
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION


def test_add_many_reports_a_duplicate_as_a_conflict(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    with pytest.raises(ErrorsException) as caught:
        store.add_many([widget(name="fresh"), widget(name="taken")])
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION


async def test_aadd_many_reports_a_duplicate_as_a_conflict(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    with pytest.raises(ErrorsException) as caught:
        await store.aadd_many([widget(name="fresh"), widget(name="taken")])
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION


def test_edit_into_a_taken_value_reports_a_conflict(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    other = store.add(widget(name="free"))
    with pytest.raises(ErrorsException) as caught:
        store.edit(other.id, widget(name="taken"))
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION


async def test_aedit_into_a_taken_value_reports_a_conflict(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    other = await store.aadd(widget(name="free"))
    with pytest.raises(ErrorsException) as caught:
        await store.aedit(other.id, widget(name="taken"))
    assert caught.value.error_enum is DatabaseErrors.UNIQUE_CONSTRAINT_VIOLATION


def test_a_conflict_carries_the_status_an_api_should_return(
    store: WidgetStore, widget: type[Any], existing: Any
) -> None:
    with pytest.raises(ErrorsException) as caught:
        store.add(widget(name="taken"))
    assert caught.value.error_enum is not None
    assert caught.value.error_enum.value["status_code"] == HTTPStatus.CONFLICT


def test_a_store_that_translates_nothing_lets_the_sql_error_through(
    widget: type[Any], existing: Any
) -> None:
    """The hook is what decides; a store that names no duplicate type opts out."""

    class UntranslatedStore(WidgetStore):
        def _duplicate_error_types(self) -> tuple[type[BaseException], ...]:
            return ()

    with pytest.raises(SQLConflictError):
        UntranslatedStore(widget).add(widget(name="taken"))


async def test_a_store_that_translates_nothing_lets_the_async_sql_error_through(
    widget: type[Any], existing: Any
) -> None:
    class UntranslatedStore(WidgetStore):
        def _duplicate_error_types(self) -> tuple[type[BaseException], ...]:
            return ()

    with pytest.raises(SQLConflictError):
        await UntranslatedStore(widget).aadd(widget(name="taken"))


def test_get_reports_a_missing_row_as_not_found(store: WidgetStore) -> None:
    with pytest.raises(ErrorsException) as caught:
        store.get(9999)
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND
    assert caught.value.error_enum.value["status_code"] == HTTPStatus.NOT_FOUND


async def test_aget_reports_a_missing_row_as_not_found(store: WidgetStore) -> None:
    with pytest.raises(ErrorsException) as caught:
        await store.aget(9999)
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND


def test_get_by_another_column_reports_a_missing_row_as_not_found(
    store: WidgetStore, existing: Any
) -> None:
    with pytest.raises(ErrorsException) as caught:
        store.get("never-stored", "name")
    assert caught.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND


@pytest.mark.parametrize(
    "call",
    [
        lambda store: store.get(1, "no_such_column"),
        lambda store: store.get_many([1], "no_such_column"),
        lambda store: store.delete(1, "no_such_column"),
    ],
)
def test_an_unknown_column_fails_naming_the_model(
    store: WidgetStore, call: Any
) -> None:
    """Unwrapped on purpose or not, this is what a typo in a column name gets."""
    with pytest.raises(AttributeError, match="has no attribute 'no_such_column'"):
        call(store)


async def test_an_unknown_column_fails_the_same_way_on_the_async_path(
    store: WidgetStore,
) -> None:
    with pytest.raises(AttributeError, match="has no attribute 'no_such_column'"):
        await store.aget(1, "no_such_column")
