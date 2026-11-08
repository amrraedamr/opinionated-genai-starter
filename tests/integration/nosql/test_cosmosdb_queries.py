"""What the emulator answers to the SQL this adapter writes.

Counting, distinct values, raw aggregation stages and the page arithmetic on
top of them, plus the two things a SQL translation layer has to get right:
that a value is bound rather than interpolated, and that a query with no
partition key reaches every partition. The service fixtures are the sibling
module's, because the directory's conftest belongs to MongoDB.
"""

from typing import Any

import pytest
from azure.cosmos import exceptions

from base.models.view_models.database import PaginationParams
from base.models.view_models.exceptions.nosql import NoSqlBadRequestError
from base.services.nosql.cosmosdb import CosmosDBService

from . import test_cosmosdb_crud

database = test_cosmosdb_crud.database
cosmos = test_cosmosdb_crud.cosmos
container = test_cosmosdb_crud.container
tenant_container = test_cosmosdb_crud.tenant_container
fields = test_cosmosdb_crud.fields

pytestmark = pytest.mark.integration


async def seed(cosmos: CosmosDBService, container: str, count: int) -> None:
    """`count` items numbered from zero."""
    await cosmos.ainsert_many(
        container, [{"id": f"i{i}", "n": i} for i in range(count)]
    )


# Count


async def test_a_count_of_an_empty_container_is_zero(
    cosmos: CosmosDBService, container: str
) -> None:
    assert await cosmos.acount(container) == 0


async def test_a_count_without_a_filter_counts_everything(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 3)
    assert await cosmos.acount(container) == 3


async def test_a_count_narrows_with_the_filter(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 5)
    assert await cosmos.acount(container, {"n": 2}) == 1


async def test_a_count_matching_nothing_is_zero(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 3)
    assert await cosmos.acount(container, {"n": 99}) == 0


def test_a_count_counts_the_whole_container_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_many(container, [{"id": "a"}, {"id": "b"}])
    assert cosmos.count(container) == 2


# Parameter binding


async def test_a_value_that_looks_like_sql_matches_only_itself(
    cosmos: CosmosDBService, container: str
) -> None:
    """The guard against a value closing the quote and continuing the query."""
    await cosmos.ainsert_many(
        container, [{"id": "a", "name": "' OR 1=1 --"}, {"id": "b", "name": "other"}]
    )
    assert await cosmos.acount(container, {"name": "' OR 1=1 --"}) == 1


async def test_a_value_carrying_a_quote_round_trips_and_filters(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "name": "O'Brien"})
    found = await cosmos.afind_one(container, {"name": "O'Brien"})
    assert found is not None and found["id"] == "a"


async def test_several_filter_fields_are_bound_as_separate_parameters(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(
        container,
        [{"id": "a", "g": 1, "n": 1}, {"id": "b", "g": 1, "n": 2}],
    )
    assert await cosmos.acount(container, {"g": 1, "n": 2}) == 1


async def test_a_filter_on_a_boolean_matches_the_boolean_not_its_string(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(
        container, [{"id": "a", "ok": True}, {"id": "b", "ok": False}]
    )
    assert await cosmos.acount(container, {"ok": True}) == 1


async def test_a_filter_on_none_matches_an_explicit_null(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(
        container, [{"id": "a", "x": None}, {"id": "b", "x": 1}, {"id": "c"}]
    )
    assert await cosmos.acount(container, {"x": None}) == 1


# Distinct


async def test_distinct_returns_each_value_once(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(
        container,
        [{"id": "a", "c": "fr"}, {"id": "b", "c": "fr"}, {"id": "c", "c": "de"}],
    )
    assert sorted(await cosmos.adistinct(container, "c")) == ["de", "fr"]


async def test_distinct_narrows_with_the_filter(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(
        container,
        [{"id": "a", "c": "fr", "ok": True}, {"id": "b", "c": "de", "ok": False}],
    )
    assert await cosmos.adistinct(container, "c", {"ok": True}) == ["fr"]


async def test_distinct_of_a_field_nothing_has_returns_nothing(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a"})
    assert await cosmos.adistinct(container, "absent") == []


def test_distinct_returns_each_value_once_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_many(container, [{"id": "a", "c": "fr"}, {"id": "b", "c": "fr"}])
    assert cosmos.distinct(container, "c") == ["fr"]


# Aggregation


async def test_an_aggregation_stage_runs_the_sql_the_caller_wrote(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(
        container,
        [{"id": "a", "g": "x"}, {"id": "b", "g": "x"}, {"id": "c", "g": "y"}],
    )
    rows = await cosmos.aaggregate(
        container,
        [{"query": "SELECT c.g, COUNT(1) AS cnt FROM c GROUP BY c.g"}],
    )
    assert sorted(rows, key=lambda r: r["g"]) == [
        {"g": "x", "cnt": 2},
        {"g": "y", "cnt": 1},
    ]


async def test_an_aggregation_stage_binds_its_own_parameters(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 4)
    rows = await cosmos.aaggregate(
        container,
        [
            {
                "query": "SELECT VALUE COUNT(1) FROM c WHERE c.n >= @floor",
                "parameters": [{"name": "@floor", "value": 2}],
            }
        ],
    )
    assert rows == [2]


async def test_every_stage_contributes_its_rows(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 2)
    rows = await cosmos.aaggregate(
        container,
        [
            {"query": "SELECT VALUE COUNT(1) FROM c"},
            {"query": "SELECT VALUE c.n FROM c WHERE c.n = 0"},
        ],
    )
    assert rows == [2, 0]


async def test_an_empty_pipeline_queries_nothing(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 2)
    assert await cosmos.aaggregate(container, []) == []


def test_an_aggregation_stage_runs_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_many(container, [{"id": "a"}, {"id": "b"}])
    assert cosmos.aggregate(container, [{"query": "SELECT VALUE COUNT(1) FROM c"}]) == [
        2
    ]


# Partitions


async def test_a_query_reaches_every_partition_without_being_told_to(
    cosmos: CosmosDBService, tenant_container: str
) -> None:
    """Six items over three partition keys, and no partition key in the API."""
    await cosmos.ainsert_many(
        tenant_container,
        [{"id": f"m{i}", "pk": f"tenant-{i % 3}"} for i in range(6)],
    )
    assert len(await cosmos.afind_many(tenant_container, {})) == 6


async def test_a_count_sees_every_partition(
    cosmos: CosmosDBService, tenant_container: str
) -> None:
    await cosmos.ainsert_many(
        tenant_container,
        [{"id": f"m{i}", "pk": f"tenant-{i % 3}"} for i in range(6)],
    )
    assert await cosmos.acount(tenant_container) == 6


async def test_a_filter_narrows_to_one_partitions_worth(
    cosmos: CosmosDBService, tenant_container: str
) -> None:
    await cosmos.ainsert_many(
        tenant_container,
        [{"id": f"m{i}", "pk": f"tenant-{i % 3}"} for i in range(6)],
    )
    assert await cosmos.acount(tenant_container, {"pk": "tenant-0"}) == 2


async def test_a_sorted_query_merges_the_partitions_in_order(
    cosmos: CosmosDBService, tenant_container: str
) -> None:
    await cosmos.ainsert_many(
        tenant_container,
        [{"id": f"m{i}", "pk": f"tenant-{i % 3}", "n": i} for i in range(6)],
    )
    found = await cosmos.afind_many(tenant_container, {}, sort=[("n", -1)])
    assert [d["n"] for d in found] == [5, 4, 3, 2, 1, 0]


async def test_a_cross_partition_query_pages_the_same_way(
    cosmos: CosmosDBService, tenant_container: str
) -> None:
    await cosmos.ainsert_many(
        tenant_container,
        [{"id": f"m{i}", "pk": f"tenant-{i % 3}", "n": i} for i in range(6)],
    )
    page = await cosmos.apaginate(
        tenant_container, {}, PaginationParams(page=2, size=2), sort=[("n", 1)]
    )
    assert [d["n"] for d in page.data] == [2, 3]
    assert page.total == 6


def test_a_query_reaches_every_partition_on_the_sync_path(
    cosmos: CosmosDBService, tenant_container: str
) -> None:
    """The async SDK never pops `enable_cross_partition_query`; neither path sends it."""
    cosmos.insert_many(
        tenant_container, [{"id": f"m{i}", "pk": f"tenant-{i}"} for i in range(3)]
    )
    assert len(cosmos.find_many(tenant_container, {})) == 3


# Sorting


async def test_a_sort_on_two_fields_is_answered_without_a_composite_index(
    cosmos: CosmosDBService, container: str
) -> None:
    """A divergence worth naming: real Cosmos needs a composite index here.

    The vNext emulator is Postgres underneath and sorts on anything, so this
    passes locally and the same call against a real account is a bad request
    until the index is added by hand. The adapter offers no way to declare one.
    """
    await cosmos.ainsert_many(
        container,
        [{"id": "a", "g": 1, "n": 2}, {"id": "b", "g": 1, "n": 1}],
    )
    found = await cosmos.afind_many(container, {}, sort=[("g", 1), ("n", -1)])
    assert [d["id"] for d in found] == ["a", "b"]


async def test_an_item_without_the_sort_field_comes_before_the_ones_with_it(
    cosmos: CosmosDBService, container: str
) -> None:
    """Cosmos's own semantics: undefined sorts ahead of every value."""
    await cosmos.ainsert_many(container, [{"id": "a", "n": 1}, {"id": "b"}])
    found = await cosmos.afind_many(container, {}, sort=[("n", 1)])
    assert [d["id"] for d in found] == ["b", "a"]


# Pagination


async def test_the_first_page_holds_the_first_size_items(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 7)
    page = await cosmos.apaginate(
        container, {}, PaginationParams(page=1, size=3), sort=[("n", 1)]
    )
    assert [d["n"] for d in page.data] == [0, 1, 2]


async def test_a_later_page_skips_the_pages_before_it(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 7)
    page = await cosmos.apaginate(
        container, {}, PaginationParams(page=3, size=3), sort=[("n", 1)]
    )
    assert [d["n"] for d in page.data] == [6]


async def test_the_total_counts_every_match_not_just_the_page(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 7)
    page = await cosmos.apaginate(container, {}, PaginationParams(page=1, size=3))
    assert page.total == 7
    assert len(page.data) == 3


async def test_the_page_count_rounds_up(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 7)
    page = await cosmos.apaginate(container, {}, PaginationParams(page=1, size=3))
    assert page.total_pages == 3


async def test_an_exact_multiple_does_not_add_a_trailing_page(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 6)
    page = await cosmos.apaginate(container, {}, PaginationParams(page=1, size=3))
    assert page.total_pages == 2


async def test_the_page_echoes_the_parameters_it_was_given(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 7)
    page = await cosmos.apaginate(container, {}, PaginationParams(page=2, size=3))
    assert (page.page, page.size) == (2, 3)


async def test_a_page_past_the_end_is_empty_but_still_reports_the_total(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 7)
    page = await cosmos.apaginate(container, {}, PaginationParams(page=9, size=3))
    assert page.data == []
    assert (page.total, page.total_pages, page.page) == (7, 3, 9)


async def test_paging_an_empty_container_reports_no_pages(
    cosmos: CosmosDBService, container: str
) -> None:
    page = await cosmos.apaginate(container, {}, PaginationParams(page=1, size=10))
    assert (page.data, page.total, page.total_pages) == ([], 0, 0)


async def test_the_filter_narrows_the_total_as_well_as_the_page(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(
        container, [{"id": f"i{i}", "g": i % 2} for i in range(10)]
    )
    page = await cosmos.apaginate(container, {"g": 0}, PaginationParams(page=1, size=2))
    assert (page.total, page.total_pages) == (5, 3)


async def test_the_projection_reaches_the_items_on_the_page(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "n": 1, "hidden": True})
    page = await cosmos.apaginate(
        container, {}, PaginationParams(page=1, size=10), projection={"n": 1}
    )
    assert fields(page.data[0]) == {"n": 1}


async def test_every_item_appears_on_exactly_one_page(
    cosmos: CosmosDBService, container: str
) -> None:
    await seed(cosmos, container, 7)
    seen: list[int] = []
    for number in (1, 2, 3):
        page = await cosmos.apaginate(
            container, {}, PaginationParams(page=number, size=3), sort=[("n", 1)]
        )
        seen.extend(d["n"] for d in page.data)
    assert seen == list(range(7))


def test_the_sync_pagination_arithmetic_matches_the_async_one(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_many(container, [{"id": f"i{i}", "n": i} for i in range(7)])
    page = cosmos.paginate(
        container, {}, PaginationParams(page=3, size=3), sort=[("n", 1)]
    )
    assert [d["n"] for d in page.data] == [6]
    assert (page.total, page.total_pages) == (7, 3)


# What the emulator will not do


async def test_a_stored_procedure_is_out_of_the_adapters_reach(
    cosmos: CosmosDBService, container: str, database: str
) -> None:
    """Named so the gap is visible: no method here reaches `scripts`.

    The vNext emulator accepts the registration, so the limitation is the
    adapter's surface rather than the server's.
    """
    scripts = cosmos._client.get_container(database, container).scripts
    scripts.create_stored_procedure({"id": "noop", "body": "function () {}"})
    assert not hasattr(cosmos, "call_stored_procedure")


async def test_a_query_the_emulator_cannot_parse_is_reported_not_swallowed(
    cosmos: CosmosDBService, container: str
) -> None:
    with pytest.raises(NoSqlBadRequestError):
        await cosmos.aaggregate(container, [{"query": "SELECT FROM WHERE"}])


def test_the_adapter_never_lets_an_azure_exception_through(
    cosmos: CosmosDBService, container: str
) -> None:
    caught: Any = None
    try:
        cosmos.aggregate(container, [{"query": "SELECT FROM WHERE"}])
    except Exception as exc:
        caught = exc
    assert not isinstance(caught, exceptions.CosmosHttpResponseError)
    assert isinstance(caught, NoSqlBadRequestError)
