"""`SQLStore`'s CRUD surface, sync and async.

Every method on `AbstractStore` has an async twin, and the two are separate
implementations rather than one wrapping the other, so each is exercised here
against the same expectations. SQLite stands in because none of this is
engine-specific.
"""

from __future__ import annotations

from typing import Any

import pytest

from base.models.view_models.database import FilteredPaginationParams

from .conftest import WidgetStore


@pytest.fixture
def seeded(store: WidgetStore, widget: type[Any]) -> list[Any]:
    """Three rows, `alpha` / `beta` / `gamma`, sizes 1 to 3."""
    return store.add_many(
        [
            widget(name="alpha", note="first", size=1),
            widget(name="beta", note="second", size=2),
            widget(name="gamma", note="third", size=3),
        ]
    )


def test_add_returns_the_row_with_its_storage_assigned_fields(
    store: WidgetStore, widget: type[Any]
) -> None:
    row = store.add(widget(name="alpha"))
    assert row.id is not None
    assert row.created_at is not None
    assert row.deleted_at is None


async def test_aadd_returns_the_row_with_its_storage_assigned_fields(
    store: WidgetStore, widget: type[Any]
) -> None:
    row = await store.aadd(widget(name="alpha"))
    assert row.id is not None
    assert row.created_at is not None


def test_add_many_returns_every_row_in_the_order_given(
    store: WidgetStore, widget: type[Any]
) -> None:
    rows = store.add_many([widget(name=n) for n in ("c", "a", "b")])
    assert [row.name for row in rows] == ["c", "a", "b"]
    assert all(row.id is not None for row in rows)


async def test_aadd_many_returns_every_row_in_the_order_given(
    store: WidgetStore, widget: type[Any]
) -> None:
    rows = await store.aadd_many([widget(name=n) for n in ("c", "a", "b")])
    assert [row.name for row in rows] == ["c", "a", "b"]


def test_get_finds_a_row_by_id(store: WidgetStore, seeded: list[Any]) -> None:
    assert store.get(seeded[1].id).name == "beta"


async def test_aget_finds_a_row_by_id(store: WidgetStore, seeded: list[Any]) -> None:
    assert (await store.aget(seeded[1].id)).name == "beta"


def test_get_finds_a_row_by_another_column(
    store: WidgetStore, seeded: list[Any]
) -> None:
    assert store.get("gamma", "name").id == seeded[2].id


async def test_aget_finds_a_row_by_another_column(
    store: WidgetStore, seeded: list[Any]
) -> None:
    assert (await store.aget("gamma", "name")).id == seeded[2].id


def test_get_many_returns_only_the_keys_that_exist(
    store: WidgetStore, seeded: list[Any]
) -> None:
    rows = store.get_many([seeded[0].id, seeded[2].id, 9999])
    assert sorted(row.name for row in rows) == ["alpha", "gamma"]


async def test_aget_many_returns_only_the_keys_that_exist(
    store: WidgetStore, seeded: list[Any]
) -> None:
    rows = await store.aget_many([seeded[0].id, seeded[2].id, 9999])
    assert sorted(row.name for row in rows) == ["alpha", "gamma"]


def test_get_many_reads_another_column(store: WidgetStore, seeded: list[Any]) -> None:
    rows = store.get_many(["alpha", "beta"], "name")
    assert sorted(row.name for row in rows) == ["alpha", "beta"]


async def test_aget_many_reads_another_column(
    store: WidgetStore, seeded: list[Any]
) -> None:
    rows = await store.aget_many(["alpha", "beta"], "name")
    assert sorted(row.name for row in rows) == ["alpha", "beta"]


def test_edit_writes_the_columns_the_caller_changed(
    store: WidgetStore, seeded: list[Any], widget: type[Any]
) -> None:
    edited = store.edit(seeded[0].id, widget(name="alpha2", note="rewritten", size=10))
    assert edited is not None
    assert (edited.name, edited.note, edited.size) == ("alpha2", "rewritten", 10)
    assert store.get(seeded[0].id).name == "alpha2"


async def test_aedit_writes_the_columns_the_caller_changed(
    store: WidgetStore, seeded: list[Any], widget: type[Any]
) -> None:
    edited = await store.aedit(
        seeded[0].id, widget(name="alpha2", note="rewritten", size=10)
    )
    assert edited is not None
    assert (edited.name, edited.note, edited.size) == ("alpha2", "rewritten", 10)


def test_edit_matches_on_another_column(
    store: WidgetStore, seeded: list[Any], widget: type[Any]
) -> None:
    edited = store.edit("beta", widget(name="beta", size=99), "name")
    assert edited is not None
    assert store.get("beta", "name").size == 99


async def test_aedit_matches_on_another_column(
    store: WidgetStore, seeded: list[Any], widget: type[Any]
) -> None:
    await store.aedit("beta", widget(name="beta", size=99), "name")
    assert (await store.aget("beta", "name")).size == 99


def test_edit_replaces_every_unprotected_column_including_the_ones_left_unset(
    store: WidgetStore, seeded: list[Any], widget: type[Any]
) -> None:
    """`edit` is a whole-row replace, not a patch: an unset column is cleared."""
    edited = store.edit(seeded[0].id, widget(name="alpha"))
    assert edited is not None
    assert edited.note is None
    assert edited.size == 0


def test_delete_reports_that_it_found_the_row(
    store: WidgetStore, seeded: list[Any]
) -> None:
    assert store.delete(seeded[0].id) is True


async def test_adelete_reports_that_it_found_the_row(
    store: WidgetStore, seeded: list[Any]
) -> None:
    assert await store.adelete(seeded[0].id) is True


def test_delete_matches_on_another_column(
    store: WidgetStore, seeded: list[Any]
) -> None:
    assert store.delete("beta", "name") is True
    assert [row.name for row in store.get_all()] == ["alpha", "gamma"]


async def test_adelete_matches_on_another_column(
    store: WidgetStore, seeded: list[Any]
) -> None:
    assert await store.adelete("beta", "name") is True
    assert [row.name for row in await store.aget_all()] == ["alpha", "gamma"]


def test_get_all_returns_every_active_row(
    store: WidgetStore, seeded: list[Any]
) -> None:
    assert sorted(row.name for row in store.get_all()) == ["alpha", "beta", "gamma"]


async def test_aget_all_returns_every_active_row(
    store: WidgetStore, seeded: list[Any]
) -> None:
    rows = await store.aget_all()
    assert sorted(row.name for row in rows) == ["alpha", "beta", "gamma"]


def test_get_all_on_an_empty_table_is_an_empty_list(store: WidgetStore) -> None:
    assert store.get_all() == []


async def test_aget_all_on_an_empty_table_is_an_empty_list(store: WidgetStore) -> None:
    assert await store.aget_all() == []


def test_get_all_ignores_keyword_arguments_it_was_handed(
    store: WidgetStore, seeded: list[Any]
) -> None:
    """The signature takes `**kwargs` for subclasses; the base store ignores them."""
    assert len(store.get_all(anything="at all")) == 3


def test_get_all_paged_returns_a_page_and_the_whole_total(
    store: WidgetStore, seeded: list[Any]
) -> None:
    page = store.get_all_paged(FilteredPaginationParams(page=1, size=2))
    assert len(page.data) == 2
    assert (page.page, page.size, page.total, page.total_pages) == (1, 2, 3, 2)


async def test_aget_all_paged_returns_a_page_and_the_whole_total(
    store: WidgetStore, seeded: list[Any]
) -> None:
    page = await store.aget_all_paged(FilteredPaginationParams(page=1, size=2))
    assert len(page.data) == 2
    assert (page.total, page.total_pages) == (3, 2)
