"""How the Cosmos adapter writes, where Cosmos has no Mongo-shaped operation.

Cosmos has no `update`, no `replace` and no multi-document delete: every write
is a read followed by `upsert_item` or `delete_item`. That translation is the
part most likely to drift, so the item calls it makes and the counts it
returns are pinned here. No Cosmos account is reachable, so this is the only
level at which the behaviour can be covered.
"""

from typing import Any

import pytest
from azure.cosmos import exceptions

from base.models.view_models.exceptions.nosql import NoSqlError, NoSqlUnexpectedError
from base.services.configuration.mixins import CosmosSettingsMixin
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


def http_error() -> Exception:
    return exceptions.CosmosHttpResponseError(message="rejected")


# Insert


def test_an_insert_is_an_upsert_and_returns_the_items_own_id(
    service: CosmosDBService, container: FakeContainer
) -> None:
    assert service.insert_one("users", {"id": "a", "n": 1}) == "a"
    assert container.call("upsert_item") == {"body": {"id": "a", "n": 1}}


def test_insert_many_upserts_one_item_per_document(
    service: CosmosDBService, container: FakeContainer
) -> None:
    assert service.insert_many("users", [{"id": "a"}, {"id": "b"}]) == ["a", "b"]
    assert container.called("upsert_item") == 2


def test_an_empty_batch_writes_nothing_and_returns_nothing(
    service: CosmosDBService, container: FakeContainer
) -> None:
    assert service.insert_many("users", []) == []
    assert container.called("upsert_item") == 0


def test_an_ordered_batch_stops_at_the_first_rejection(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.upsert_error = http_error()
    with pytest.raises(NoSqlError):
        service.insert_many("users", [{"id": "a"}, {"id": "b"}], ordered=True)


def test_an_unordered_batch_skips_what_cosmos_rejected(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.upsert_error = http_error()
    assert service.insert_many("users", [{"id": "a"}], ordered=False) == []
    assert container.called("upsert_item") == 1


# Update


def test_an_update_reads_the_document_then_writes_the_merged_one(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [{"id": "a", "n": 1, "keep": True}]
    assert service.update_one("users", {"n": 1}, {"$set": {"n": 2}}) == 1
    assert container.call("upsert_item")["body"] == {"id": "a", "n": 2, "keep": True}


def test_an_update_without_an_operator_is_treated_as_a_set(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [{"id": "a", "n": 1}]
    service.update_one("users", {"n": 1}, {"n": 5})
    assert container.call("upsert_item")["body"] == {"id": "a", "n": 5}


def test_an_update_matching_nothing_writes_nothing(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = []
    assert service.update_one("users", {"n": 1}, {"$set": {"n": 2}}) == 0
    assert container.called("upsert_item") == 0


def test_an_upserting_update_builds_the_document_from_the_filter(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = []
    assert (
        service.update_one("users", {"id": "a"}, {"$set": {"n": 2}}, upsert=True) == 1
    )
    assert container.call("upsert_item")["body"] == {"id": "a", "n": 2}


def test_update_many_writes_every_matched_document(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert service.update_many("users", {}, {"$set": {"n": 1}}) == 3
    assert container.called("upsert_item") == 3


def test_update_many_matching_nothing_writes_nothing(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = []
    assert service.update_many("users", {}, {"$set": {"n": 1}}) == 0
    assert container.called("upsert_item") == 0


def test_update_many_with_upsert_inserts_a_single_document(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = []
    assert (
        service.update_many("users", {"id": "a"}, {"$set": {"n": 1}}, upsert=True) == 1
    )
    assert container.called("upsert_item") == 1


# Replace


def test_a_replace_writes_the_replacement_whole(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [{"id": "a", "old": True}]
    assert service.replace_one("users", {"id": "a"}, {"id": "a", "new": True}) == 1
    assert container.call("upsert_item")["body"] == {"id": "a", "new": True}


def test_a_replace_matching_nothing_writes_nothing(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.missing = True
    assert service.replace_one("users", {"id": "a"}, {"n": 1}) == 0
    assert container.called("upsert_item") == 0


def test_an_upserting_replace_takes_its_id_from_the_filter(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.missing = True
    assert service.replace_one("users", {"id": "a"}, {"n": 1}, upsert=True) == 1
    assert container.call("upsert_item")["body"] == {"n": 1, "id": "a"}


# Delete


def test_a_delete_reads_the_document_then_removes_it_by_id(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [{"id": "a"}]
    assert service.delete_one("users", {"n": 1}) == 1
    assert container.call("delete_item") == {"item": "a", "partition_key": "a"}


def test_a_delete_matching_nothing_removes_nothing(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = []
    assert service.delete_one("users", {"n": 1}) == 0
    assert container.called("delete_item") == 0


def test_delete_many_removes_one_item_per_match(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [{"id": "a"}, {"id": "b"}]
    assert service.delete_many("users", {}) == 2
    assert container.called("delete_item") == 2


def test_delete_many_matching_nothing_removes_nothing(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = []
    assert service.delete_many("users", {}) == 0
    assert container.called("delete_item") == 0


def test_the_partition_key_is_assumed_to_be_the_id(
    service: CosmosDBService, container: FakeContainer
) -> None:
    """Pins the convention: a container partitioned on anything else will fail."""
    container.results = [{"id": "a", "tenant": "t1"}]
    service.delete_one("users", {"id": "a"})
    assert container.call("delete_item")["partition_key"] == "a"


def test_a_document_without_an_id_fails_as_an_unexpected_error(
    service: CosmosDBService, container: FakeContainer
) -> None:
    container.results = [{"n": 1}]
    with pytest.raises(NoSqlUnexpectedError):
        service.delete_one("users", {"n": 1})


# Async counterparts


async def test_the_async_insert_returns_the_items_own_id(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    assert await service.ainsert_one("users", {"id": "a"}) == "a"


async def test_the_async_batch_upserts_one_item_per_document(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    assert await service.ainsert_many("users", [{"id": "a"}, {"id": "b"}]) == ["a", "b"]


async def test_the_async_unordered_batch_skips_what_cosmos_rejected(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    acontainer.upsert_error = http_error()
    assert await service.ainsert_many("users", [{"id": "a"}], ordered=False) == []


async def test_the_async_update_writes_the_merged_document(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    acontainer.results = [{"id": "a", "n": 1}]
    assert await service.aupdate_one("users", {"n": 1}, {"$set": {"n": 2}}) == 1
    assert acontainer.call("upsert_item")["body"] == {"id": "a", "n": 2}


async def test_the_async_update_many_writes_every_match(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    acontainer.results = [{"id": "a"}, {"id": "b"}]
    assert await service.aupdate_many("users", {}, {"$set": {"n": 1}}) == 2


async def test_the_async_replace_writes_the_replacement_whole(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    acontainer.results = [{"id": "a"}]
    assert await service.areplace_one("users", {"id": "a"}, {"id": "a", "n": 2}) == 1
    assert acontainer.call("upsert_item")["body"] == {"id": "a", "n": 2}


async def test_the_async_delete_removes_the_item_by_id(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    acontainer.results = [{"id": "a"}]
    assert await service.adelete_one("users", {"n": 1}) == 1
    assert acontainer.call("delete_item") == {"item": "a", "partition_key": "a"}


async def test_the_async_delete_many_removes_one_item_per_match(
    service: CosmosDBService, acontainer: AsyncFakeContainer
) -> None:
    acontainer.results = [{"id": "a"}, {"id": "b"}]
    assert await service.adelete_many("users", {}) == 2
