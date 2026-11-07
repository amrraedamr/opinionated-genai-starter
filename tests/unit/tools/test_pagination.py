"""`tools/pagination.py`: the page arithmetic every paginated backend shares.

A page past the end must not reach the store at all, which is the only reason
`fetch` is a callback rather than a list the caller already built.
"""

from base.models.view_models.database import PaginationParams
from base.tools.pagination import apaginate, paginate

ROWS = list(range(25))


def fetch(skip: int, limit: int) -> list[int]:
    return ROWS[skip : skip + limit]


async def afetch(skip: int, limit: int) -> list[int]:
    return fetch(skip, limit)


def test_the_first_page_starts_at_the_beginning() -> None:
    result = paginate(len(ROWS), PaginationParams(page=1, size=10), fetch)
    assert result.data == list(range(10))
    assert (result.page, result.size, result.total, result.total_pages) == (
        1,
        10,
        25,
        3,
    )


def test_a_later_page_is_offset_by_the_pages_before_it() -> None:
    result = paginate(len(ROWS), PaginationParams(page=3, size=10), fetch)
    assert result.data == list(range(20, 25))


def test_the_page_count_rounds_up_for_a_partial_last_page() -> None:
    assert paginate(25, PaginationParams(size=10), fetch).total_pages == 3
    assert paginate(20, PaginationParams(size=10), fetch).total_pages == 2


def test_an_empty_table_reports_no_pages_and_fetches_nothing() -> None:
    calls: list[tuple[int, int]] = []

    def record(skip: int, limit: int) -> list[int]:
        calls.append((skip, limit))
        return []

    result = paginate(0, PaginationParams(), record)
    assert result.data == []
    assert result.total_pages == 0
    assert calls == []


def test_a_page_past_the_end_never_reaches_the_store() -> None:
    calls: list[tuple[int, int]] = []

    def record(skip: int, limit: int) -> list[int]:
        calls.append((skip, limit))
        return fetch(skip, limit)

    result = paginate(25, PaginationParams(page=4, size=10), record)
    assert result.data == []
    assert result.page == 4
    assert calls == []


def test_the_last_page_still_reaches_the_store() -> None:
    calls: list[tuple[int, int]] = []

    def record(skip: int, limit: int) -> list[int]:
        calls.append((skip, limit))
        return fetch(skip, limit)

    paginate(25, PaginationParams(page=3, size=10), record)
    assert calls == [(20, 10)]


def test_the_reported_window_is_what_was_asked_for_not_what_came_back() -> None:
    result = paginate(25, PaginationParams(page=3, size=10), fetch)
    assert result.size == 10
    assert len(result.data) == 5


async def test_the_async_form_pages_the_same_way() -> None:
    result = await apaginate(len(ROWS), PaginationParams(page=2, size=10), afetch)
    assert result.data == list(range(10, 20))
    assert result.total_pages == 3


async def test_the_async_form_also_skips_the_store_past_the_end() -> None:
    calls: list[tuple[int, int]] = []

    async def record(skip: int, limit: int) -> list[int]:
        calls.append((skip, limit))
        return []

    result = await apaginate(5, PaginationParams(page=9, size=10), record)
    assert result.data == []
    assert calls == []
