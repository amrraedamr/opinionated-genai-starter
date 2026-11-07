"""Page arithmetic and filtering behind `get_all_paged`.

The page a caller asks for and the total they are told about come from two
different queries, so they are pinned together: a window that is right but a
total that is wrong still breaks the caller's paging. Search is pinned against
`search_fields`, because a store that matches a column it never declared is
leaking data a caller did not ask to be searchable.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from base.models.view_models.database import FilteredPaginationParams, PaginationParams

from .conftest import WidgetStore


@pytest.fixture
def seeded(store: WidgetStore, widget: type[Any]) -> list[Any]:
    """Seven rows, `w0` to `w6`, each with a note of its own."""
    return store.add_many(
        [widget(name=f"w{i}", note=f"note-{i}", size=i) for i in range(7)]
    )


def page_of(store: WidgetStore, page: int, size: int) -> Any:
    return store.get_all_paged(FilteredPaginationParams(page=page, size=size))


@pytest.mark.parametrize(
    ("page", "size", "expected"),
    [
        (1, 3, ["w0", "w1", "w2"]),
        (2, 3, ["w3", "w4", "w5"]),
        (3, 3, ["w6"]),
        (1, 7, ["w0", "w1", "w2", "w3", "w4", "w5", "w6"]),
        (7, 1, ["w6"]),
    ],
)
def test_a_page_holds_the_rows_its_offset_points_at(
    store: WidgetStore, seeded: list[Any], page: int, size: int, expected: list[str]
) -> None:
    assert [row.name for row in page_of(store, page, size).data] == expected


@pytest.mark.parametrize(
    ("size", "total_pages"), [(1, 7), (2, 4), (3, 3), (7, 1), (100, 1)]
)
def test_total_pages_rounds_a_partial_page_up(
    store: WidgetStore, seeded: list[Any], size: int, total_pages: int
) -> None:
    result = page_of(store, 1, size)
    assert result.total == 7
    assert result.total_pages == total_pages


def test_a_page_past_the_end_is_empty_and_still_reports_the_total(
    store: WidgetStore, seeded: list[Any]
) -> None:
    result = page_of(store, 4, 3)
    assert result.data == []
    assert (result.total, result.total_pages) == (7, 3)


async def test_an_async_page_past_the_end_is_empty_and_still_reports_the_total(
    store: WidgetStore, seeded: list[Any]
) -> None:
    result = await store.aget_all_paged(FilteredPaginationParams(page=4, size=3))
    assert result.data == []
    assert (result.total, result.total_pages) == (7, 3)


def test_an_empty_table_reports_no_pages(store: WidgetStore) -> None:
    result = page_of(store, 1, 10)
    assert (result.data, result.total, result.total_pages) == ([], 0, 0)


def test_the_largest_page_the_params_allow_still_works(
    store: WidgetStore, seeded: list[Any]
) -> None:
    result = page_of(store, 1, 100)
    assert len(result.data) == 7
    assert result.size == 100


def test_the_page_echoes_back_what_was_asked_for(
    store: WidgetStore, seeded: list[Any]
) -> None:
    result = page_of(store, 2, 3)
    assert (result.page, result.size) == (2, 3)


def test_search_matches_a_declared_search_field(
    store: WidgetStore, seeded: list[Any]
) -> None:
    result = store.get_all_paged(
        FilteredPaginationParams(page=1, size=10, search="note-3")
    )
    assert [row.name for row in result.data] == ["w3"]
    assert result.total == 1


def test_search_is_case_insensitive(store: WidgetStore, seeded: list[Any]) -> None:
    result = store.get_all_paged(
        FilteredPaginationParams(page=1, size=10, search="NOTE-3")
    )
    assert [row.name for row in result.data] == ["w3"]


def test_search_matches_a_substring(store: WidgetStore, seeded: list[Any]) -> None:
    result = store.get_all_paged(
        FilteredPaginationParams(page=1, size=10, search="note")
    )
    assert result.total == 7


async def test_async_search_matches_a_declared_search_field(
    store: WidgetStore, seeded: list[Any]
) -> None:
    result = await store.aget_all_paged(
        FilteredPaginationParams(page=1, size=10, search="note-3")
    )
    assert [row.name for row in result.data] == ["w3"]


def test_search_never_matches_a_column_outside_search_fields(
    widget: type[Any], seeded: list[Any]
) -> None:
    class NameOnlyStore(WidgetStore):
        search_fields: ClassVar[tuple[str, ...]] = ("name",)

    result = NameOnlyStore(widget).get_all_paged(
        FilteredPaginationParams(page=1, size=10, search="note-3")
    )
    assert result.total == 0


def test_a_store_declaring_no_search_fields_ignores_the_search_term(
    widget: type[Any], seeded: list[Any]
) -> None:
    class UnsearchableStore(WidgetStore):
        search_fields: ClassVar[tuple[str, ...]] = ()

    result = UnsearchableStore(widget).get_all_paged(
        FilteredPaginationParams(page=1, size=10, search="nothing-matches-this")
    )
    assert result.total == 7


def test_an_empty_search_term_filters_nothing(
    store: WidgetStore, seeded: list[Any]
) -> None:
    assert store.get_all_paged(FilteredPaginationParams(search="")).total == 7


def test_search_narrows_the_total_as_well_as_the_page(
    store: WidgetStore, seeded: list[Any]
) -> None:
    result = store.get_all_paged(
        FilteredPaginationParams(page=1, size=2, search="note-")
    )
    assert len(result.data) == 2
    assert (result.total, result.total_pages) == (7, 4)


def test_a_from_date_after_every_row_excludes_them_all(
    store: WidgetStore, seeded: list[Any]
) -> None:
    future = datetime.now(UTC) + timedelta(days=1)
    assert store.get_all_paged(FilteredPaginationParams(from_date=future)).total == 0


def test_a_to_date_before_every_row_excludes_them_all(
    store: WidgetStore, seeded: list[Any]
) -> None:
    past = datetime.now(UTC) - timedelta(days=1)
    assert store.get_all_paged(FilteredPaginationParams(to_date=past)).total == 0


def test_a_range_spanning_now_includes_every_row(
    store: WidgetStore, seeded: list[Any]
) -> None:
    now = datetime.now(UTC)
    params = FilteredPaginationParams(
        from_date=now - timedelta(days=1), to_date=now + timedelta(days=1)
    )
    assert store.get_all_paged(params).total == 7


def test_pagination_defaults_to_the_first_page_of_ten() -> None:
    params = PaginationParams()
    assert (params.page, params.size, params.search) == (1, 10, "")


def test_filtered_pagination_defaults_to_an_unbounded_range() -> None:
    params = FilteredPaginationParams()
    assert params.from_date == datetime.min.replace(tzinfo=UTC)
    assert params.to_date == datetime.max.replace(tzinfo=UTC)


@pytest.mark.parametrize("size", [1, 50, 100])
def test_a_size_within_the_cap_is_accepted(size: int) -> None:
    assert PaginationParams(size=size).size == size


@pytest.mark.parametrize("values", [{"page": 0}, {"page": -1}])
def test_a_page_below_one_is_refused(values: dict[str, int]) -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        PaginationParams(**values)


def test_a_size_below_one_is_refused() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        PaginationParams(size=0)


def test_a_size_above_the_cap_is_refused() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        PaginationParams(size=101)


def test_the_filtered_params_inherit_the_same_bounds() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        FilteredPaginationParams(size=101)
