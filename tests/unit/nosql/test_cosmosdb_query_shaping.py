"""The Cosmos SQL the adapter writes, and the parameters it binds.

Cosmos has no document-level filter language, so every Mongo-shaped call is
translated into a SQL string here. There is no Cosmos account and no emulator
available to this repository, so the whole adapter is pinned by reading the
query it would have sent: the clause builders, the fast path for a read by
id, and the way `skip` and `limit` become `OFFSET`/`LIMIT` or `SELECT TOP`.
"""

from typing import Any

import pytest

from base.models.view_models.database import PaginationParams
from base.services.configuration.mixins import CosmosSettingsMixin
from base.services.nosql import AbstractNoSQLService
from base.services.nosql.cosmosdb import CosmosDBService
from tests.unit.nosql.cosmos_doubles import AsyncFakeContainer, FakeContainer


@pytest.fixture
def service(make_settings: Any) -> CosmosDBService:
    make_settings(
        CosmosSettingsMixin,
        azure_cosmos_account_uri="https://account.documents.azure.com",
        azure_cosmos_database_name="app",
    )
    return CosmosDBService()


@pytest.fixture
def container(
    service: CosmosDBService, monkeypatch: pytest.MonkeyPatch
) -> FakeContainer:
    fake = FakeContainer()
    monkeypatch.setattr(service, "_get_container", lambda collection: fake)
    return fake


@pytest.fixture
def acontainer(
    service: CosmosDBService, monkeypatch: pytest.MonkeyPatch
) -> AsyncFakeContainer:
    fake = AsyncFakeContainer()
    monkeypatch.setattr(service, "_aget_container", lambda collection: fake)
    return fake


# Clause builders


def test_a_filter_becomes_an_and_of_bound_parameters(
    service: CosmosDBService,
) -> None:
    where, params = service._build_where_clause({"status": "active", "age": 30})
    assert where == "c.status = @p0 AND c.age = @p1"
    assert params == [
        {"name": "@p0", "value": "active"},
        {"name": "@p1", "value": 30},
    ]


def test_an_empty_filter_builds_no_clause(service: CosmosDBService) -> None:
    assert service._build_where_clause({}) == ("", [])


def test_values_are_bound_rather_than_interpolated(service: CosmosDBService) -> None:
    """The guard against a value closing the quote and continuing the query."""
    where, params = service._build_where_clause({"name": "' OR 1=1 --"})
    assert where == "c.name = @p0"
    assert params[0]["value"] == "' OR 1=1 --"


def test_a_projection_selects_only_the_included_fields(
    service: CosmosDBService,
) -> None:
    assert service._build_select_clause({"name": 1, "age": 1}) == "c.name, c.age"


def test_a_projection_that_excludes_everything_selects_the_whole_document(
    service: CosmosDBService,
) -> None:
    """Cosmos cannot express exclusion, so an exclude-only projection is dropped."""
    assert service._build_select_clause({"_id": 0}) == "c"


def test_no_projection_selects_the_whole_document(service: CosmosDBService) -> None:
    assert service._build_select_clause(None) == "c"


def test_sort_tuples_become_an_order_by_clause(service: CosmosDBService) -> None:
    assert (
        service._build_order_clause([("name", 1), ("age", -1)])
        == " ORDER BY c.name ASC, c.age DESC"
    )


def test_a_zero_direction_sorts_ascending(service: CosmosDBService) -> None:
    assert service._build_order_clause([("name", 0)]) == " ORDER BY c.name ASC"


def test_no_sort_builds_no_order_clause(service: CosmosDBService) -> None:
    assert service._build_order_clause(None) == ""


# Find


def test_a_lone_id_filter_reads_the_item_directly(
    service: CosmosDBService, container: FakeContainer
) -> None:
    assert service.find_one("users", {"id": "abc"}) == {"id": "abc"}
    assert container.call("read_item") == {"item": "abc", "partition_key": "abc"}
    assert container.called("query_items") == 0


def test_a_missing_item_on_the_fast_path_is_none_rather_than_an_error(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.missing = True
    assert service.find_one("users", {"id": "abc"}) is None


def test_an_id_alongside_another_field_goes_through_a_query(
    service: CosmosDBService, container: FakeContainer
) -> None:
    service.find_one("users", {"id": "abc", "active": True})
    assert container.called("read_item") == 0
    assert (
        container.queries[-1]
        == "SELECT TOP 1 c FROM c WHERE c.id = @p0 AND c.active = @p1"
    )


def test_find_one_asks_for_a_single_row(
    service: CosmosDBService, container: FakeContainer
) -> None:
    service.find_one("users", {"active": True})
    assert container.queries[-1].startswith("SELECT TOP 1 ")


def test_find_one_returns_none_when_the_query_matched_nothing(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = []
    assert service.find_one("users", {"active": True}) is None


def test_every_query_crosses_partitions(
    service: CosmosDBService, container: FakeContainer
) -> None:
    """No partition key is threaded through the API, so queries must fan out."""
    service.find_many("users", {})
    assert container.call("query_items")["enable_cross_partition_query"] is True


def test_a_skip_becomes_offset_and_limit(
    service: CosmosDBService, container: FakeContainer
) -> None:
    service.find_many("users", {"a": 1}, skip=5, limit=2)
    assert container.queries[-1] == "SELECT c FROM c WHERE c.a = @p0 OFFSET 5 LIMIT 2"


def test_a_skip_without_a_limit_borrows_the_largest_one_cosmos_takes(
    service: CosmosDBService, container: FakeContainer
) -> None:
    service.find_many("users", {}, skip=4)
    assert container.queries[-1] == "SELECT c FROM c OFFSET 4 LIMIT 2147483647"


def test_a_limit_without_a_skip_becomes_select_top(
    service: CosmosDBService, container: FakeContainer
) -> None:
    service.find_many("users", {}, limit=3)
    assert container.queries[-1] == "SELECT TOP 3 c FROM c"


def test_a_sort_is_placed_before_the_paging_clause(
    service: CosmosDBService, container: FakeContainer
) -> None:
    service.find_many("users", {}, sort=[("name", 1)], skip=1, limit=2)
    assert (
        container.queries[-1] == "SELECT c FROM c ORDER BY c.name ASC OFFSET 1 LIMIT 2"
    )


def test_an_empty_parameter_list_is_sent_as_none(
    service: CosmosDBService, container: FakeContainer
) -> None:
    service.find_many("users", {})
    assert container.call("query_items")["parameters"] is None


# Count, distinct and aggregation


def test_count_asks_for_a_scalar(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [7]
    assert service.count("users", {"active": True}) == 7
    assert container.queries[-1] == "SELECT VALUE COUNT(1) FROM c WHERE c.active = @p0"


def test_count_without_a_filter_counts_the_whole_container(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [3]
    assert service.count("users") == 3
    assert container.queries[-1] == "SELECT VALUE COUNT(1) FROM c"


def test_count_reads_an_empty_answer_as_zero(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = []
    assert service.count("users") == 0


def test_distinct_asks_for_distinct_values_of_one_field(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = ["fr", "de"]
    assert service.distinct("users", "country") == ["fr", "de"]
    assert container.queries[-1] == "SELECT DISTINCT VALUE c.country FROM c"


def test_distinct_narrows_with_the_filter(
    service: CosmosDBService, container: FakeContainer
) -> None:
    service.distinct("users", "country", {"active": True})
    assert (
        container.queries[-1]
        == "SELECT DISTINCT VALUE c.country FROM c WHERE c.active = @p0"
    )


def test_an_aggregation_stage_is_a_raw_cosmos_query(
    service: CosmosDBService, container: FakeContainer
) -> None:
    """Cosmos has no pipeline, so a stage carries the SQL the caller wrote."""
    container.results = [{"status": "a", "cnt": 2}]
    stage = {
        "query": "SELECT c.status, COUNT(1) AS cnt FROM c GROUP BY c.status",
        "parameters": [{"name": "@p0", "value": 1}],
    }
    assert service.aggregate("users", [stage]) == [{"status": "a", "cnt": 2}]
    sent = container.call("query_items")
    assert sent["query"] == stage["query"]
    assert sent["parameters"] == stage["parameters"]


def test_an_empty_pipeline_queries_nothing(
    service: CosmosDBService, container: FakeContainer
) -> None:
    assert service.aggregate("users", []) == []
    assert container.called("query_items") == 0


def test_every_stage_contributes_its_rows(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [{"n": 1}]
    rows = service.aggregate("users", [{"query": "SELECT 1"}, {"query": "SELECT 2"}])
    assert rows == [{"n": 1}, {"n": 1}]


# Pagination


def test_pagination_counts_then_pages_with_offset_and_limit(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [30]
    page = service.paginate("users", {}, PaginationParams(page=3, size=10))
    assert page.page == 3
    assert "OFFSET 20 LIMIT 10" in container.queries[-1]


def test_a_page_past_the_end_runs_only_the_count(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [5]
    page = service.paginate("users", {}, PaginationParams(page=4, size=10))
    assert page.data == []
    assert len(container.queries) == 1


# Async counterparts


async def test_the_async_fast_path_reads_the_item_directly(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    assert await service.afind_one("users", {"id": "abc"}) == {"id": "abc"}
    assert acontainer.called("query_items") == 0


async def test_the_async_find_many_builds_the_same_sql(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    await service.afind_many("users", {"a": 1}, skip=5, limit=2)
    assert acontainer.queries[-1] == "SELECT c FROM c WHERE c.a = @p0 OFFSET 5 LIMIT 2"


async def test_the_async_count_reads_the_scalar(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    acontainer.results = [9]
    assert await service.acount("users") == 9


async def test_the_async_distinct_builds_the_same_sql(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    acontainer.results = ["fr"]
    assert await service.adistinct("users", "country") == ["fr"]
    assert acontainer.queries[-1] == "SELECT DISTINCT VALUE c.country FROM c"


async def test_the_async_aggregation_runs_every_stage(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    acontainer.results = [{"n": 1}]
    rows = await service.aaggregate("users", [{"query": "a"}, {"query": "b"}])
    assert rows == [{"n": 1}, {"n": 1}]


async def test_the_async_pagination_pages_with_offset_and_limit(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    acontainer.results = [30]
    page = await service.apaginate("users", {}, PaginationParams(page=2, size=10))
    assert (page.total, page.total_pages) == (30, 3)
    assert "OFFSET 10 LIMIT 10" in acontainer.queries[-1]


def test_the_service_implements_every_abstract_method(
    service: CosmosDBService,
) -> None:
    assert CosmosDBService.__abstractmethods__ == frozenset()
    assert isinstance(service, AbstractNoSQLService)


def test_the_service_names_itself_cosmosdb(service: CosmosDBService) -> None:
    assert service.get_service_name() == "CosmosDB"
