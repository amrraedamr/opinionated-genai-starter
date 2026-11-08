"""What a real Cosmos refuses, and which domain error the caller sees.

The adapter's job at a failure is to replace the Azure SDK's exception with
one from the document store family, so a caller catches one family whichever
backend is behind it. These tests make each failure actually happen rather
than injecting it, because a handler that has been bypassed by another layer
still looks correct from the inside — which is how a sibling backend's error
handling turned out to be inert.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from azure.core.exceptions import AzureError
from azure.cosmos import exceptions

from base.models.view_models.exceptions.nosql import (
    NoSqlBadRequestError,
    NoSqlConflictError,
    NoSqlConnectionError,
    NoSqlError,
    NoSqlNotFoundError,
    NoSqlUnexpectedError,
)
from base.services.configuration.mixins import CosmosSettingsMixin
from base.services.error_handling.nosql.cosmos import CosmosErrorHandler
from base.services.nosql.cosmosdb import CosmosDBService
from base.tools.errors import ServiceHealthCheckError

from . import test_cosmosdb_crud

database = test_cosmosdb_crud.database
cosmos = test_cosmosdb_crud.cosmos
container = test_cosmosdb_crud.container
tenant_container = test_cosmosdb_crud.tenant_container
fields = test_cosmosdb_crud.fields

pytestmark = pytest.mark.integration


@pytest.fixture
async def unreachable(make_settings: Any) -> AsyncIterator[CosmosDBService]:
    """A service pointed at a port nothing is listening on, retried once."""
    make_settings(
        CosmosSettingsMixin,
        azure_cosmos_account_uri="http://127.0.0.1:9/",
        azure_cosmos_account_key=test_cosmosdb_crud.ACCOUNT_KEY,
        azure_cosmos_database_name="unreachable",
    )
    service = CosmosDBService()
    service._error_handler = CosmosErrorHandler(retries=1, base_wait=0)
    yield service
    await service.adispose()
    service.dispose()


# Missing things


async def test_a_missing_item_is_none_rather_than_an_error(
    cosmos: CosmosDBService, container: str
) -> None:
    assert await cosmos.afind_one(container, {"id": "gone"}) is None


async def test_a_query_against_a_missing_container_is_a_not_found_error(
    cosmos: CosmosDBService,
) -> None:
    with pytest.raises(NoSqlNotFoundError):
        await cosmos.afind_many("no_such_container", {})


async def test_a_count_of_a_missing_container_is_a_not_found_error(
    cosmos: CosmosDBService,
) -> None:
    """Unlike MongoDB, which answers zero for a collection it has never seen."""
    with pytest.raises(NoSqlNotFoundError):
        await cosmos.acount("no_such_container")


async def test_a_write_to_a_missing_container_is_a_not_found_error(
    cosmos: CosmosDBService,
) -> None:
    with pytest.raises(NoSqlNotFoundError):
        await cosmos.ainsert_one("no_such_container", {"id": "a"})


def test_a_missing_container_is_a_not_found_error_on_the_sync_path(
    cosmos: CosmosDBService,
) -> None:
    with pytest.raises(NoSqlNotFoundError):
        cosmos.find_many("no_such_container", {})


async def test_a_missing_database_is_a_not_found_error(
    make_settings: Any, database: str
) -> None:
    make_settings(
        CosmosSettingsMixin,
        azure_cosmos_account_uri=test_cosmosdb_crud.ACCOUNT_URI,
        azure_cosmos_account_key=test_cosmosdb_crud.ACCOUNT_KEY,
        azure_cosmos_database_name=f"{database}_absent",
    )
    service = CosmosDBService()
    try:
        with pytest.raises(NoSqlNotFoundError):
            await service.acount("anything")
    finally:
        await service.adispose()
        service.dispose()


# Malformed requests


async def test_a_query_the_server_cannot_parse_is_a_bad_request(
    cosmos: CosmosDBService, container: str
) -> None:
    with pytest.raises(NoSqlBadRequestError):
        await cosmos.aaggregate(container, [{"query": "SELECT NONSENSE FROM"}])


async def test_a_bad_query_carries_the_servers_own_status(
    cosmos: CosmosDBService, container: str
) -> None:
    with pytest.raises(NoSqlBadRequestError) as caught:
        await cosmos.aaggregate(container, [{"query": "SELECT NONSENSE FROM"}])
    assert caught.value.status_code == 400


async def test_a_document_without_an_id_is_a_bad_request(
    cosmos: CosmosDBService, container: str
) -> None:
    """Cosmos will not name a document for you, so an id is the caller's job."""
    with pytest.raises(NoSqlBadRequestError, match="id field"):
        await cosmos.ainsert_one(container, {"n": 1})


async def test_an_upserting_update_whose_filter_carries_no_id_is_a_bad_request(
    cosmos: CosmosDBService, container: str
) -> None:
    """The document it builds from the filter has nothing to be named by."""
    with pytest.raises(NoSqlBadRequestError, match="id field"):
        await cosmos.aupdate_one(container, {"g": 1}, {"$set": {"n": 1}}, upsert=True)


async def test_an_id_the_sdk_refuses_never_reaches_the_server(
    cosmos: CosmosDBService, container: str
) -> None:
    """A `/` in an id is a `ValueError`, which is not an `AzureError` at all."""
    with pytest.raises(NoSqlUnexpectedError, match="illegal chars"):
        await cosmos.ainsert_one(container, {"id": "a/b"})


def test_an_id_the_sdk_refuses_is_unexpected_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    with pytest.raises(NoSqlUnexpectedError):
        cosmos.insert_one(container, {"id": "a#b"})


# Conflicts


async def test_the_same_id_twice_overwrites_rather_than_conflicting(
    cosmos: CosmosDBService, container: str
) -> None:
    """Every insert here is an upsert, so this adapter never raises a 409."""
    await cosmos.ainsert_one(container, {"id": "a", "n": 1})
    await cosmos.ainsert_one(container, {"id": "a", "n": 2})
    assert await cosmos.acount(container) == 1


def test_a_conflict_the_server_reports_is_a_conflict_the_caller_catches(
    cosmos: CosmosDBService, container: str, database: str
) -> None:
    """`create_item` is the only call that conflicts, and no method exposes it.

    Driven through the service's own handler so the classification covering
    that 409 is exercised against a real one rather than a constructed error.
    """
    proxy = cosmos._client.get_container(database, container)
    cosmos._error_handler.run(proxy.create_item, body={"id": "a"})
    with pytest.raises(NoSqlConflictError) as caught:
        cosmos._error_handler.run(proxy.create_item, body={"id": "a"})
    assert caught.value.status_code == 409
    assert isinstance(caught.value.provider_error, exceptions.CosmosResourceExistsError)


# Batches


async def test_an_ordered_batch_stops_at_the_first_rejection(
    cosmos: CosmosDBService, container: str
) -> None:
    with pytest.raises(NoSqlBadRequestError):
        await cosmos.ainsert_many(container, [{"id": "a"}, {"n": 1}, {"id": "c"}])


async def test_an_ordered_batch_keeps_what_it_wrote_before_the_rejection(
    cosmos: CosmosDBService, container: str
) -> None:
    """Pins that a failed batch is not rolled back: there is no transaction."""
    with pytest.raises(NoSqlError):
        await cosmos.ainsert_many(container, [{"id": "a"}, {"n": 1}, {"id": "c"}])
    assert await cosmos.acount(container) == 1


async def test_an_unordered_batch_skips_what_cosmos_rejected(
    cosmos: CosmosDBService, container: str
) -> None:
    ids = await cosmos.ainsert_many(
        container, [{"id": "a"}, {"n": 1}, {"id": "c"}], ordered=False
    )
    assert ids == ["a", "c"]
    assert await cosmos.acount(container) == 2


def test_an_unordered_batch_skips_what_cosmos_rejected_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    assert cosmos.insert_many(container, [{"n": 1}], ordered=False) == []


# The partition key this service assumes


async def test_an_item_partitioned_on_anything_else_is_invisible_to_a_read_by_id(
    cosmos: CosmosDBService, tenant_container: str
) -> None:
    """The convention's cost: a point read sends the id as the partition key.

    Cosmos answers 404 for a partition that holds nothing, and the fast path
    reads a 404 as "no such item", so the item is simply missing.
    """
    await cosmos.ainsert_one(tenant_container, {"id": "a", "pk": "tenant-1"})
    assert await cosmos.afind_one(tenant_container, {"id": "a"}) is None


async def test_the_same_item_is_found_by_a_query_that_does_not_read_by_id(
    cosmos: CosmosDBService, tenant_container: str
) -> None:
    await cosmos.ainsert_one(tenant_container, {"id": "a", "pk": "tenant-1"})
    found = await cosmos.afind_many(tenant_container, {"id": "a"})
    assert fields(found[0]) == {"id": "a", "pk": "tenant-1"}


async def test_deleting_an_item_partitioned_on_anything_else_is_a_not_found_error(
    cosmos: CosmosDBService, tenant_container: str
) -> None:
    """The query finds it, and the delete that follows aims at the wrong partition."""
    await cosmos.ainsert_one(tenant_container, {"id": "a", "pk": "tenant-1"})
    with pytest.raises(NoSqlNotFoundError) as caught:
        await cosmos.adelete_one(tenant_container, {"pk": "tenant-1"})
    assert caught.value.status_code == 404


async def test_an_update_through_a_wrong_partition_key_still_writes(
    cosmos: CosmosDBService, tenant_container: str
) -> None:
    """Writes are routed by the body, so only the read and delete paths break."""
    await cosmos.ainsert_one(tenant_container, {"id": "a", "pk": "tenant-1"})
    assert await cosmos.aupdate_one(tenant_container, {"id": "a"}, {"n": 1}) == 0
    assert await cosmos.aupdate_many(tenant_container, {"id": "a"}, {"n": 1}) == 1


# Classification, and whether it is doing anything


async def test_no_azure_exception_reaches_a_caller(
    cosmos: CosmosDBService, container: str
) -> None:
    """Whatever went wrong, it arrives as the document store family."""
    for call in (
        lambda: cosmos.afind_many("no_such_container", {}),
        lambda: cosmos.ainsert_one(container, {"n": 1}),
        lambda: cosmos.aaggregate(container, [{"query": "SELECT FROM"}]),
    ):
        with pytest.raises(NoSqlError) as caught:
            await call()
        assert not isinstance(caught.value, AzureError)


async def test_a_failure_arrives_classified_rather_than_already_wrapped(
    cosmos: CosmosDBService,
) -> None:
    """The inertness check: the handler saw the SDK's own exception.

    A layer that wrapped the error first would leave `provider_error` holding
    something domain-shaped and the status code gone, and every failure would
    land on the unexpected member of the family.
    """
    with pytest.raises(NoSqlNotFoundError) as caught:
        await cosmos.acount("no_such_container")
    assert isinstance(
        caught.value.provider_error, exceptions.CosmosResourceNotFoundError
    )
    assert caught.value.status_code == 404


async def test_the_kind_of_failure_decides_the_class_not_the_method(
    cosmos: CosmosDBService, container: str
) -> None:
    not_found = (NoSqlNotFoundError, None)
    bad_request = (NoSqlBadRequestError, 400)
    for expected, status, call in (
        (*not_found, lambda: cosmos.acount("no_such_container")),
        (*not_found, lambda: cosmos.adelete_many("no_such_container", {})),
        (*bad_request, lambda: cosmos.ainsert_one(container, {"n": 1})),
        (*bad_request, lambda: cosmos.aaggregate(container, [{"query": "SELECT"}])),
    ):
        with pytest.raises(expected) as caught:
            await call()
        if status is not None:
            assert caught.value.status_code == status


async def test_the_id_cosmos_correlates_with_is_not_the_one_the_handler_reads(
    cosmos: CosmosDBService,
) -> None:
    """Cosmos answers with `x-ms-activity-id`, which the shared Azure handler
    does not look for, so `request_id` is always None behind this adapter."""
    with pytest.raises(NoSqlNotFoundError) as caught:
        await cosmos.acount("no_such_container")
    assert caught.value.request_id is None
    provider = caught.value.provider_error
    assert isinstance(provider, exceptions.CosmosHttpResponseError)
    assert "x-ms-activity-id" in provider.response.headers


# Retrying


async def test_an_unreachable_account_is_retried_and_then_reported_as_exhausted(
    unreachable: CosmosDBService,
) -> None:
    """A connection failure is the one kind this suite can really provoke."""
    with pytest.raises(NoSqlConnectionError) as caught:
        await unreachable.acount("anything")
    assert caught.value.attempts == 2


def test_an_unreachable_account_is_retried_on_the_sync_path_too(
    unreachable: CosmosDBService,
) -> None:
    with pytest.raises(NoSqlConnectionError) as caught:
        unreachable.count("anything")
    assert caught.value.attempts == 2


async def test_a_retried_failure_still_carries_the_sdk_error_underneath(
    unreachable: CosmosDBService,
) -> None:
    with pytest.raises(NoSqlConnectionError) as caught:
        await unreachable.afind_many("anything", {})
    assert isinstance(caught.value.provider_error, AzureError)


async def test_a_failure_that_will_not_improve_is_not_retried(
    cosmos: CosmosDBService, container: str
) -> None:
    """A bad request is not a `NoSqlRetryExhausted`, so it never had attempts."""
    with pytest.raises(NoSqlBadRequestError) as caught:
        await cosmos.ainsert_one(container, {"n": 1})
    assert not hasattr(caught.value, "attempts")


async def test_an_unreachable_account_fails_its_health_check(
    unreachable: CosmosDBService,
) -> None:
    with pytest.raises(ServiceHealthCheckError):
        await unreachable.ahealth_check()


def test_an_unreachable_account_fails_its_health_check_on_the_sync_path(
    unreachable: CosmosDBService,
) -> None:
    with pytest.raises(ServiceHealthCheckError):
        unreachable.health_check()


# What the emulator will not let us prove


async def test_a_wrong_account_key_cannot_be_told_apart_here(
    make_settings: Any, database: str
) -> None:
    """The emulator accepts any key, so the auth path has no local coverage.

    Written rather than omitted: against a real account this is a 401 and so
    a `NoSqlAuthError`, and nothing short of a real account can show it.
    """
    pytest.skip("the vNext emulator does not validate the account key")
