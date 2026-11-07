"""Paging and vector shapes: `PaginationParams`, `PaginatedResult`, `VectorRecord`.

`PaginationParams` is bound to query strings a client controls, so the caps are
what stop one request asking for the whole table.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from base.models.definitions.database import SortingOrder
from base.models.view_models.database import (
    FilteredPaginationParams,
    PagedDataResponse,
    PagedResponse,
    PaginatedResult,
    PaginationParams,
    SortingParams,
)
from base.models.view_models.vector_search import VectorRecord, VectorSearchResult


def test_pagination_defaults_to_the_first_page_of_ten() -> None:
    params = PaginationParams()
    assert (params.page, params.size, params.search) == (1, 10, "")


def test_a_page_size_of_one_hundred_is_the_most_a_client_may_ask_for() -> None:
    assert PaginationParams(size=100).size == 100
    with pytest.raises(ValidationError, match="less_than_equal"):
        PaginationParams(size=101)


def test_a_page_size_below_one_is_refused() -> None:
    with pytest.raises(ValidationError, match="greater_than_equal"):
        PaginationParams(size=0)


def test_a_page_number_below_one_is_refused() -> None:
    with pytest.raises(ValidationError, match="greater_than_equal"):
        PaginationParams(page=0)


def test_pagination_params_read_their_camel_case_aliases() -> None:
    assert PaginationParams(page=3, size=20).model_dump(by_alias=True) == {
        "page": 3,
        "size": 20,
        "search": "",
    }


def test_a_paginated_result_reports_the_window_and_the_whole() -> None:
    result: PaginatedResult[str] = PaginatedResult(
        data=["a", "b"], page=2, size=2, total=5, total_pages=3
    )
    assert result.data == ["a", "b"]
    assert (result.page, result.size, result.total, result.total_pages) == (2, 2, 5, 3)


def test_a_paginated_result_camelises_its_page_count_on_dump() -> None:
    result: PaginatedResult[int] = PaginatedResult(
        data=[1], page=1, size=1, total=1, total_pages=1
    )
    assert "totalPages" in result.serialize()


def test_a_paginated_result_requires_every_count_it_reports() -> None:
    with pytest.raises(ValidationError):
        PaginatedResult(data=[], page=1, size=1, total=1)  # type: ignore[call-arg]


def test_filtered_pagination_keeps_the_page_caps_it_inherits() -> None:
    assert issubclass(FilteredPaginationParams, PaginationParams)
    with pytest.raises(ValidationError):
        FilteredPaginationParams(size=101)


def test_filtered_pagination_defaults_to_a_window_that_excludes_nothing() -> None:
    params = FilteredPaginationParams()
    assert params.from_date == datetime.min.replace(tzinfo=UTC)
    assert params.to_date == datetime.max.replace(tzinfo=UTC)


def test_filtered_pagination_accepts_an_explicit_window() -> None:
    params = FilteredPaginationParams(
        from_date=datetime(2026, 1, 1, tzinfo=UTC),
        to_date=datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert params.from_date < params.to_date


def test_sorting_params_default_to_no_opinion() -> None:
    params = SortingParams()
    assert params.sort_by is None
    assert params.sort_order is None
    assert SortingParams(sort_order="desc").sort_order is SortingOrder.DESC


def test_the_paged_response_envelope_wraps_data_beside_its_counts() -> None:
    response: PagedDataResponse[int] = PagedDataResponse(
        data=[1, 2],
        pagination=PagedResponse(
            total_records=2, total_pages=1, page_number=1, page_size=10
        ),
    )
    dumped = response.serialize()
    assert dumped["data"] == [1, 2]
    assert dumped["pagination"]["totalRecords"] == 2


def test_a_vector_record_starts_with_empty_metadata() -> None:
    record = VectorRecord(id="doc-1", vector=[0.1, 0.2])
    assert record.metadata == {}
    assert record.vector == [0.1, 0.2]


def test_each_vector_record_gets_its_own_metadata_mapping() -> None:
    first = VectorRecord(id="a", vector=[0.0])
    second = VectorRecord(id="b", vector=[0.0])
    first.metadata["source"] = "x"
    assert second.metadata == {}


def test_a_vector_record_requires_an_id_and_a_vector() -> None:
    with pytest.raises(ValidationError):
        VectorRecord(id="a")  # type: ignore[call-arg]


def test_a_search_result_carries_its_score_and_may_omit_the_vector() -> None:
    result = VectorSearchResult(id="doc-1", score=0.83)
    assert result.score == 0.83
    assert result.vector is None
    assert result.metadata == {}
