"""Soft delete: what `delete` does, and what every read then refuses to see.

`SQLStore` promises that a deleted row is still in the table and never in a
result. Both halves matter — the first is what makes a restore possible, the
second is what keeps a deleted row out of a caller's hands — so both are
pinned here, on every read the store offers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from sqlalchemy import func, select

from base.models.view_models.database import FilteredPaginationParams
from base.services.sql.service import SQLService
from base.tools.errors import ErrorsException

from .conftest import WidgetStore


def raw_count(model: type[Any]) -> int:
    """Rows in the table, soft-deleted ones included."""
    return SQLService().query(
        select(func.count()).select_from(model), lambda r: r.scalar()
    )


@pytest.fixture
def seeded(store: WidgetStore, widget: type[Any]) -> list[Any]:
    return store.add_many(
        [widget(name="alpha", note="shared"), widget(name="beta", note="shared")]
    )


def test_delete_leaves_the_row_in_the_table(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    store.delete(seeded[0].id)
    assert raw_count(widget) == 2


def test_delete_stamps_deleted_at(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    store.delete(seeded[0].id)
    stamp = SQLService().query(
        select(widget.deleted_at).where(widget.id == seeded[0].id), lambda r: r.scalar()
    )
    assert isinstance(stamp, datetime)


async def test_adelete_stamps_deleted_at(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    await store.adelete(seeded[0].id)
    stamp = await SQLService().aquery(
        select(widget.deleted_at).where(widget.id == seeded[0].id), lambda r: r.scalar()
    )
    assert isinstance(stamp, datetime)


def test_get_refuses_a_soft_deleted_row(store: WidgetStore, seeded: list[Any]) -> None:
    store.delete(seeded[0].id)
    with pytest.raises(ErrorsException):
        store.get(seeded[0].id)


async def test_aget_refuses_a_soft_deleted_row(
    store: WidgetStore, seeded: list[Any]
) -> None:
    await store.adelete(seeded[0].id)
    with pytest.raises(ErrorsException):
        await store.aget(seeded[0].id)


def test_get_many_drops_a_soft_deleted_row(
    store: WidgetStore, seeded: list[Any]
) -> None:
    store.delete(seeded[0].id)
    rows = store.get_many([row.id for row in seeded])
    assert [row.name for row in rows] == ["beta"]


async def test_aget_many_drops_a_soft_deleted_row(
    store: WidgetStore, seeded: list[Any]
) -> None:
    await store.adelete(seeded[0].id)
    rows = await store.aget_many([row.id for row in seeded])
    assert [row.name for row in rows] == ["beta"]


def test_get_all_drops_a_soft_deleted_row(
    store: WidgetStore, seeded: list[Any]
) -> None:
    store.delete(seeded[0].id)
    assert [row.name for row in store.get_all()] == ["beta"]


async def test_aget_all_drops_a_soft_deleted_row(
    store: WidgetStore, seeded: list[Any]
) -> None:
    await store.adelete(seeded[0].id)
    assert [row.name for row in await store.aget_all()] == ["beta"]


def test_get_all_paged_counts_only_active_rows(
    store: WidgetStore, seeded: list[Any]
) -> None:
    store.delete(seeded[0].id)
    page = store.get_all_paged(FilteredPaginationParams(page=1, size=10))
    assert page.total == 1
    assert [row.name for row in page.data] == ["beta"]


async def test_aget_all_paged_counts_only_active_rows(
    store: WidgetStore, seeded: list[Any]
) -> None:
    await store.adelete(seeded[0].id)
    page = await store.aget_all_paged(FilteredPaginationParams(page=1, size=10))
    assert page.total == 1


def test_edit_refuses_a_soft_deleted_row(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    store.delete(seeded[0].id)
    assert store.edit(seeded[0].id, widget(name="revived")) is None


async def test_aedit_refuses_a_soft_deleted_row(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    await store.adelete(seeded[0].id)
    assert await store.aedit(seeded[0].id, widget(name="revived")) is None


def test_a_soft_deleted_unique_value_still_blocks_reinsertion(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    """The constraint is the database's, and soft delete does not reach it."""
    store.delete(seeded[0].id)
    with pytest.raises(ErrorsException, match="already exists"):
        store.add(widget(name="alpha"))


async def test_a_soft_deleted_unique_value_still_blocks_async_reinsertion(
    store: WidgetStore, widget: type[Any], seeded: list[Any]
) -> None:
    await store.adelete(seeded[0].id)
    with pytest.raises(ErrorsException, match="already exists"):
        await store.aadd(widget(name="alpha"))


def test_restore_puts_a_row_back_in_every_read(
    store: WidgetStore, seeded: list[Any]
) -> None:
    store.delete(seeded[0].id)
    row = SQLService().query(
        select(seeded[0].__class__).where(seeded[0].__class__.id == seeded[0].id),
        lambda r: r.scalar_one(),
    )
    row.restore()
    SQLService().upsert_row(row)
    assert store.get(seeded[0].id).name == "alpha"


def test_a_store_that_cascades_soft_deletes_its_children_too(
    widget: type[Any], seeded: list[Any]
) -> None:
    class CascadingStore(WidgetStore):
        def _cascade(self, entity: Any) -> list[Any]:
            return [
                row
                for row in self._many(self._active(self._model.note == entity.note))
                if row.id != entity.id
            ]

    cascading = CascadingStore(widget)
    assert cascading.delete(seeded[0].id) is True
    assert cascading.get_all() == []


async def test_a_store_that_cascades_async_soft_deletes_its_children_too(
    widget: type[Any], seeded: list[Any]
) -> None:
    class CascadingStore(WidgetStore):
        async def _acascade(self, entity: Any) -> list[Any]:
            rows = await self._amany(self._active(self._model.note == entity.note))
            return [row for row in rows if row.id != entity.id]

    cascading = CascadingStore(widget)
    assert await cascading.adelete(seeded[0].id) is True
    assert await cascading.aget_all() == []


def test_the_base_store_cascades_to_nothing(
    store: WidgetStore, seeded: list[Any]
) -> None:
    assert store._cascade(seeded[0]) == []


async def test_the_base_store_cascades_to_nothing_asynchronously(
    store: WidgetStore, seeded: list[Any]
) -> None:
    assert await store._acascade(seeded[0]) == []
