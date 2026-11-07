"""Page arithmetic against a real server, including the page past the end.

The parameter object is `PaginationParams` and its field is `size`, not
`page_size`: a caller writing `page_size=20` gets the default 10 silently,
because the model ignores what it does not declare. That is pinned here along
with the arithmetic, since both are easy to get wrong and quiet when wrong.
"""

import pytest

from base.models.view_models.database import PaginationParams
from base.services.nosql.mongodb import MongoDBService

pytestmark = pytest.mark.integration


async def seed(mongo: MongoDBService, collection: str, count: int) -> None:
    """`count` documents numbered from zero."""
    await mongo.ainsert_many(collection, [{"i": i} for i in range(count)])


def test_the_page_size_field_is_called_size() -> None:
    assert "size" in PaginationParams.model_fields
    assert "page_size" not in PaginationParams.model_fields


def test_an_unknown_page_size_argument_is_ignored_rather_than_refused() -> None:
    """Pins the trap: `page_size=20` silently leaves the default in place."""
    assert PaginationParams(page_size=20).size == 10  # type: ignore[call-arg]


def test_a_page_number_below_one_is_refused() -> None:
    with pytest.raises(ValueError):
        PaginationParams(page=0)


def test_a_page_size_above_the_cap_is_refused() -> None:
    with pytest.raises(ValueError):
        PaginationParams(size=101)


async def test_the_first_page_holds_the_first_size_documents(
    mongo: MongoDBService, collection: str
) -> None:
    await seed(mongo, collection, 7)
    page = await mongo.apaginate(
        collection, {}, PaginationParams(page=1, size=3), sort=[("i", 1)]
    )
    assert [d["i"] for d in page.data] == [0, 1, 2]


async def test_a_later_page_skips_the_pages_before_it(
    mongo: MongoDBService, collection: str
) -> None:
    await seed(mongo, collection, 7)
    page = await mongo.apaginate(
        collection, {}, PaginationParams(page=3, size=3), sort=[("i", 1)]
    )
    assert [d["i"] for d in page.data] == [6]


async def test_the_total_counts_every_match_not_just_the_page(
    mongo: MongoDBService, collection: str
) -> None:
    await seed(mongo, collection, 7)
    page = await mongo.apaginate(collection, {}, PaginationParams(page=1, size=3))
    assert page.total == 7
    assert len(page.data) == 3


async def test_the_page_count_rounds_up(mongo: MongoDBService, collection: str) -> None:
    await seed(mongo, collection, 7)
    page = await mongo.apaginate(collection, {}, PaginationParams(page=1, size=3))
    assert page.total_pages == 3


async def test_an_exact_multiple_does_not_add_a_trailing_page(
    mongo: MongoDBService, collection: str
) -> None:
    await seed(mongo, collection, 6)
    page = await mongo.apaginate(collection, {}, PaginationParams(page=1, size=3))
    assert page.total_pages == 2


async def test_the_page_echoes_the_parameters_it_was_given(
    mongo: MongoDBService, collection: str
) -> None:
    await seed(mongo, collection, 7)
    page = await mongo.apaginate(collection, {}, PaginationParams(page=2, size=3))
    assert (page.page, page.size) == (2, 3)


async def test_a_page_past_the_end_is_empty_but_still_reports_the_total(
    mongo: MongoDBService, collection: str
) -> None:
    await seed(mongo, collection, 7)
    page = await mongo.apaginate(collection, {}, PaginationParams(page=9, size=3))
    assert page.data == []
    assert (page.total, page.total_pages, page.page) == (7, 3, 9)


async def test_paging_an_empty_collection_reports_no_pages(
    mongo: MongoDBService, collection: str
) -> None:
    page = await mongo.apaginate(collection, {}, PaginationParams(page=1, size=10))
    assert (page.data, page.total, page.total_pages) == ([], 0, 0)


async def test_the_filter_narrows_the_total_as_well_as_the_page(
    mongo: MongoDBService, collection: str
) -> None:
    await seed(mongo, collection, 10)
    page = await mongo.apaginate(
        collection, {"i": {"$lt": 4}}, PaginationParams(page=1, size=2)
    )
    assert page.total == 4
    assert page.total_pages == 2


async def test_the_projection_reaches_the_documents_on_the_page(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"i": 1, "hidden": True}])
    page = await mongo.apaginate(
        collection, {}, PaginationParams(page=1, size=10), projection={"i": 1}
    )
    assert "hidden" not in page.data[0]


async def test_every_document_appears_on_exactly_one_page(
    mongo: MongoDBService, collection: str
) -> None:
    await seed(mongo, collection, 7)
    seen: list[int] = []
    for number in (1, 2, 3):
        page = await mongo.apaginate(
            collection, {}, PaginationParams(page=number, size=3), sort=[("i", 1)]
        )
        seen.extend(d["i"] for d in page.data)
    assert seen == list(range(7))


def test_the_sync_pagination_arithmetic_matches_the_async_one(
    mongo: MongoDBService, collection: str
) -> None:
    mongo.insert_many(collection, [{"i": i} for i in range(7)])
    page = mongo.paginate(
        collection, {}, PaginationParams(page=3, size=3), sort=[("i", 1)]
    )
    assert [d["i"] for d in page.data] == [6]
    assert (page.total, page.total_pages) == (7, 3)
