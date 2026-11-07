"""Query, execute, upsert and pagination on `AbstractSQLService`.

These are the methods a store is built out of. Each opens its own session, so
what they are pinned for is the result they hand back and what survives the
session closing underneath it.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import delete, func, select, text, update

from base.models.definitions.database import SortingOrder
from base.models.view_models.database import PaginationParams, SortingParams
from base.models.view_models.exceptions.sql import SQLBadRequestError, SQLConflictError
from base.services.sql import AbstractSQLService
from base.services.sql.service import SQLService
from base.tools.database import SQLResultFormats


@pytest.fixture
def seeded(widget: type[Any]) -> type[Any]:
    """Five rows, named `w0` to `w4`, sizes 0 to 4."""
    SQLService().upsert_rows([widget(name=f"w{i}", size=i) for i in range(5)])
    return widget


def test_query_returns_what_the_processor_built(seeded: type[Any]) -> None:
    names = SQLService().query(
        select(seeded.name).order_by(seeded.name), lambda r: [row[0] for row in r]
    )
    assert names == ["w0", "w1", "w2", "w3", "w4"]


def test_query_rows_stay_readable_after_the_session_closes(seeded: type[Any]) -> None:
    rows = SQLService().query(
        select(seeded).order_by(seeded.size), SQLResultFormats.all_scalar_list
    )
    assert [row.name for row in rows] == ["w0", "w1", "w2", "w3", "w4"]


async def test_aquery_rows_stay_readable_after_the_session_closes(
    seeded: type[Any],
) -> None:
    rows = await SQLService().aquery(
        select(seeded).order_by(seeded.size), SQLResultFormats.all_scalar_list
    )
    assert [row.name for row in rows] == ["w0", "w1", "w2", "w3", "w4"]


def test_query_reports_a_bad_statement_as_a_bad_request(widget: type[Any]) -> None:
    with pytest.raises(SQLBadRequestError):
        SQLService().query(
            select(text("nope")).select_from(text(widget.__tablename__)),
            SQLResultFormats.all,
        )


async def test_aquery_reports_a_bad_statement_as_a_bad_request(
    widget: type[Any],
) -> None:
    with pytest.raises(SQLBadRequestError):
        await SQLService().aquery(
            select(text("nope")).select_from(text(widget.__tablename__)),
            SQLResultFormats.all,
        )


def test_execute_applies_an_update(seeded: type[Any]) -> None:
    service = SQLService()
    service.execute(update(seeded).where(seeded.name == "w1").values(size=42))
    assert (
        service.query(
            select(seeded.size).where(seeded.name == "w1"), lambda r: r.scalar()
        )
        == 42
    )


async def test_aexecute_applies_an_update(seeded: type[Any]) -> None:
    service = SQLService()
    await service.aexecute(update(seeded).where(seeded.name == "w1").values(size=42))
    assert (
        await service.aquery(
            select(seeded.size).where(seeded.name == "w1"), lambda r: r.scalar()
        )
        == 42
    )


def test_update_many_applies_to_every_matching_row(seeded: type[Any]) -> None:
    service = SQLService()
    service.update_many(update(seeded).values(size=7))
    assert service.query(
        select(func.min(seeded.size), func.max(seeded.size)), lambda r: r.one()
    ) == (7, 7)


async def test_aupdate_many_applies_to_every_matching_row(seeded: type[Any]) -> None:
    service = SQLService()
    await service.aupdate_many(update(seeded).values(size=7))
    assert await service.aquery(
        select(func.min(seeded.size), func.max(seeded.size)), lambda r: r.one()
    ) == (7, 7)


def test_delete_many_removes_every_matching_row(seeded: type[Any]) -> None:
    service = SQLService()
    service.delete_many(delete(seeded).where(seeded.size >= 3))
    assert (
        service.query(select(func.count()).select_from(seeded), lambda r: r.scalar())
        == 3
    )


async def test_adelete_many_removes_every_matching_row(seeded: type[Any]) -> None:
    service = SQLService()
    await service.adelete_many(delete(seeded).where(seeded.size >= 3))
    assert (
        await service.aquery(
            select(func.count()).select_from(seeded), lambda r: r.scalar()
        )
        == 3
    )


def test_upsert_row_populates_storage_assigned_fields(widget: type[Any]) -> None:
    row = SQLService().upsert_row(widget(name="a"))
    assert row.id is not None
    assert row.created_at is not None
    assert row.updated_at is not None


async def test_aupsert_row_populates_storage_assigned_fields(
    widget: type[Any],
) -> None:
    row = await SQLService().aupsert_row(widget(name="a"))
    assert row.id is not None
    assert row.created_at is not None


def test_upsert_rows_returns_the_rows_in_the_order_given(widget: type[Any]) -> None:
    rows = SQLService().upsert_rows([widget(name=n) for n in ("c", "a", "b")])
    assert [row.name for row in rows] == ["c", "a", "b"]
    assert all(row.id is not None for row in rows)


async def test_aupsert_rows_returns_the_rows_in_the_order_given(
    widget: type[Any],
) -> None:
    rows = await SQLService().aupsert_rows([widget(name=n) for n in ("c", "a", "b")])
    assert [row.name for row in rows] == ["c", "a", "b"]


def test_bulk_upsert_rows_returns_nothing_and_still_writes(widget: type[Any]) -> None:
    service = SQLService()
    assert service.bulk_upsert_rows([widget(name="a"), widget(name="b")]) is None
    assert (
        service.query(select(func.count()).select_from(widget), lambda r: r.scalar())
        == 2
    )


async def test_abulk_upsert_rows_returns_nothing_and_still_writes(
    widget: type[Any],
) -> None:
    service = SQLService()
    assert await service.abulk_upsert_rows([widget(name="a")]) is None
    assert (
        await service.aquery(
            select(func.count()).select_from(widget), lambda r: r.scalar()
        )
        == 1
    )


def test_upsert_row_reports_a_duplicate_as_a_conflict(widget: type[Any]) -> None:
    service = SQLService()
    service.upsert_row(widget(name="a"))
    with pytest.raises(SQLConflictError):
        service.upsert_row(widget(name="a"))


async def test_aupsert_row_reports_a_duplicate_as_a_conflict(
    widget: type[Any],
) -> None:
    service = SQLService()
    await service.aupsert_row(widget(name="a"))
    with pytest.raises(SQLConflictError):
        await service.aupsert_row(widget(name="a"))


def test_execute_in_transaction_runs_statements_before_inserts(
    seeded: type[Any],
) -> None:
    service = SQLService()
    rows = service.execute_in_transaction(
        statements=[delete(seeded).where(seeded.size >= 3)],
        rows_to_add=[seeded(name="new", size=9)],
    )
    assert rows[0].id is not None
    assert service.query(
        select(seeded.name).order_by(seeded.name), lambda r: [row[0] for row in r]
    ) == ["new", "w0", "w1", "w2"]


async def test_aexecute_in_transaction_runs_statements_before_inserts(
    seeded: type[Any],
) -> None:
    service = SQLService()
    rows = await service.aexecute_in_transaction(
        statements=[delete(seeded).where(seeded.size >= 3)],
        rows_to_add=[seeded(name="new", size=9)],
    )
    assert rows[0].id is not None
    assert (
        await service.aquery(
            select(func.count()).select_from(seeded), lambda r: r.scalar()
        )
        == 4
    )


def test_execute_in_transaction_keeps_the_delete_only_if_the_insert_lands(
    seeded: type[Any],
) -> None:
    service = SQLService()
    with pytest.raises(SQLConflictError):
        service.execute_in_transaction(
            statements=[delete(seeded).where(seeded.size >= 3)],
            rows_to_add=[seeded(name="w0")],
        )
    assert (
        service.query(select(func.count()).select_from(seeded), lambda r: r.scalar())
        == 5
    )


async def test_aexecute_in_transaction_keeps_the_delete_only_if_the_insert_lands(
    seeded: type[Any],
) -> None:
    service = SQLService()
    with pytest.raises(SQLConflictError):
        await service.aexecute_in_transaction(
            statements=[delete(seeded).where(seeded.size >= 3)],
            rows_to_add=[seeded(name="w0")],
        )
    assert (
        await service.aquery(
            select(func.count()).select_from(seeded), lambda r: r.scalar()
        )
        == 5
    )


def test_execute_in_transaction_with_nothing_to_do_returns_an_empty_list(
    widget: type[Any],
) -> None:
    assert SQLService().execute_in_transaction() == []


async def test_aexecute_in_transaction_with_nothing_to_do_returns_an_empty_list(
    widget: type[Any],
) -> None:
    assert await SQLService().aexecute_in_transaction() == []


def test_paginate_reports_the_window_and_the_whole_total(seeded: type[Any]) -> None:
    page = SQLService().paginate(
        select(seeded).order_by(seeded.size), PaginationParams(page=2, size=2)
    )
    assert [row.name for row in page.data] == ["w2", "w3"]
    assert (page.page, page.size, page.total, page.total_pages) == (2, 2, 5, 3)


async def test_apaginate_reports_the_window_and_the_whole_total(
    seeded: type[Any],
) -> None:
    page = await SQLService().apaginate(
        select(seeded).order_by(seeded.size), PaginationParams(page=2, size=2)
    )
    assert [row.name for row in page.data] == ["w2", "w3"]
    assert (page.total, page.total_pages) == (5, 3)


def test_paginate_past_the_end_returns_no_rows_and_still_counts(
    seeded: type[Any],
) -> None:
    page = SQLService().paginate(select(seeded), PaginationParams(page=99, size=2))
    assert page.data == []
    assert (page.total, page.total_pages) == (5, 3)


def test_paginate_return_all_hands_back_whole_rows(seeded: type[Any]) -> None:
    page = SQLService().paginate(
        select(seeded.name, seeded.size).order_by(seeded.size),
        PaginationParams(page=1, size=2),
        return_all=True,
    )
    assert [tuple(row) for row in page.data] == [("w0", 0), ("w1", 1)]


def test_paginate_has_subquery_flattens_each_row(seeded: type[Any]) -> None:
    page = SQLService().paginate(
        select(seeded.name).order_by(seeded.name),
        PaginationParams(page=1, size=2),
        has_subquery=True,
    )
    assert page.data == ["w0", "w1"]


def test_extract_paginated_data_chooses_a_shape_per_flag() -> None:
    rows: list[Any] = [("a", 1), ("b", 2)]
    assert AbstractSQLService._extract_paginated_data(rows, True, False) == rows
    assert AbstractSQLService._extract_paginated_data(rows, False, True) == [
        "a",
        1,
        "b",
        2,
    ]
    assert AbstractSQLService._extract_paginated_data(rows, False, False) == ["a", "b"]


@pytest.mark.parametrize(
    ("order", "expected"),
    [(SortingOrder.ASC, "ASC"), (SortingOrder.DESC, "DESC")],
)
def test_sorting_builds_the_clause_the_order_asks_for(
    order: SortingOrder, expected: str
) -> None:
    clause = AbstractSQLService.sorting(SortingParams(sort_by="name", sort_order=order))
    assert clause is not None
    assert str(clause) == f"name {expected}"


@pytest.mark.parametrize(
    "params",
    [
        SortingParams(),
        SortingParams(sort_by="name"),
        SortingParams(sort_order=SortingOrder.ASC),
        SortingParams(sort_by="", sort_order=SortingOrder.ASC),
    ],
)
def test_sorting_is_absent_unless_both_halves_are_given(params: SortingParams) -> None:
    assert AbstractSQLService.sorting(params) is None
