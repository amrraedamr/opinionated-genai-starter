"""What each service method hands the driver, and what it reads back.

Every method here is a thin translation: a filter, a projection and a cursor
chain go down, a count or an id comes back. These tests run the translation
against a fake collection so the argument order, the keyword names and the
field read off each result are pinned without a server.
"""

import pytest
from bson import ObjectId

from base.models.view_models.database import PaginationParams
from base.services.configuration.mixins import MongoSettingsMixin
from base.services.nosql.mongodb import MongoDBService
from tests.conftest import MakeSettings
from tests.unit.nosql.mongo_doubles import (
    AsyncFakeDatabase,
    FakeDatabase,
    FakeWriteResult,
)


@pytest.fixture
def service(make_settings: MakeSettings) -> MongoDBService:
    make_settings(MongoSettingsMixin, mongodb_host="db:27017", mongodb_database="app")
    return MongoDBService()


@pytest.fixture
def db(service: MongoDBService, monkeypatch: pytest.MonkeyPatch) -> FakeDatabase:
    fake = FakeDatabase()
    monkeypatch.setattr(service, "_get_database", lambda: fake)
    return fake


@pytest.fixture
def adb(service: MongoDBService, monkeypatch: pytest.MonkeyPatch) -> AsyncFakeDatabase:
    fake = AsyncFakeDatabase()
    monkeypatch.setattr(service, "_aget_database", lambda: fake)
    return fake


# Find


async def test_afind_one_passes_the_filter_positionally_and_the_projection_by_name(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.afind_one("users", {"_id": "a"}, projection={"name": 1})
    args, kwargs = adb["users"].call("find_one")
    assert args == ({"_id": "a"},)
    assert kwargs == {"projection": {"name": 1}}


def test_find_one_passes_the_same_arguments_as_its_async_twin(
    service: MongoDBService, db: FakeDatabase
) -> None:
    service.find_one("users", {"_id": "a"}, projection={"name": 1})
    args, kwargs = db["users"].call("find_one")
    assert args == ({"_id": "a"},)
    assert kwargs == {"projection": {"name": 1}}


async def test_afind_one_returns_none_when_the_collection_is_empty(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    assert await service.afind_one("users", {"_id": "missing"}) is None


async def test_afind_many_applies_sort_skip_and_limit_to_the_cursor(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.afind_many(
        "users", {"active": True}, sort=[("name", 1)], skip=10, limit=5
    )
    cursor = adb["users"].cursors[-1]
    assert cursor.sorted_by == [("name", 1)]
    assert cursor.skipped == 10
    assert cursor.limited == 5


async def test_afind_many_leaves_the_cursor_untouched_without_sort_skip_or_limit(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.afind_many("users", {})
    cursor = adb["users"].cursors[-1]
    assert (cursor.sorted_by, cursor.skipped, cursor.limited) == (None, None, None)


async def test_a_zero_limit_is_not_sent_as_a_limit(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    """Zero means `no limit` to the driver, so the service must not call `limit`."""
    await service.afind_many("users", {}, limit=0, skip=0)
    cursor = adb["users"].cursors[-1]
    assert cursor.limited is None
    assert cursor.skipped is None


def test_find_many_applies_the_same_cursor_chain_as_its_async_twin(
    service: MongoDBService, db: FakeDatabase
) -> None:
    service.find_many("users", {}, sort=[("name", -1)], skip=2, limit=3)
    cursor = db["users"].cursors[-1]
    assert (cursor.sorted_by, cursor.skipped, cursor.limited) == (
        [("name", -1)],
        2,
        3,
    )


async def test_afind_many_returns_the_documents_the_cursor_yielded(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    adb["users"].documents = [{"_id": 1}, {"_id": 2}]
    assert await service.afind_many("users", {}) == [{"_id": 1}, {"_id": 2}]


# Insert


async def test_ainsert_one_returns_the_generated_id_as_a_string(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    identifier = ObjectId()
    adb["users"].result = FakeWriteResult(inserted_id=identifier)
    assert await service.ainsert_one("users", {"n": 1}) == str(identifier)


async def test_ainsert_one_sends_the_document_positionally(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.ainsert_one("users", {"n": 1})
    args, _ = adb["users"].call("insert_one")
    assert args == ({"n": 1},)


def test_insert_one_returns_the_generated_id_as_a_string(
    service: MongoDBService, db: FakeDatabase
) -> None:
    identifier = ObjectId()
    db["users"].result = FakeWriteResult(inserted_id=identifier)
    assert service.insert_one("users", {"n": 1}) == str(identifier)


async def test_ainsert_many_defaults_to_an_ordered_batch(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.ainsert_many("users", [{"n": 1}])
    _, kwargs = adb["users"].call("insert_many")
    assert kwargs == {"ordered": True}


async def test_ainsert_many_forwards_an_unordered_batch(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.ainsert_many("users", [{"n": 1}], ordered=False)
    _, kwargs = adb["users"].call("insert_many")
    assert kwargs == {"ordered": False}


async def test_ainsert_many_returns_every_id_as_a_string(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    ids = [ObjectId(), ObjectId()]
    adb["users"].result = FakeWriteResult(inserted_ids=ids)
    assert await service.ainsert_many("users", [{}, {}]) == [str(i) for i in ids]


# Update and replace


async def test_aupdate_one_sends_the_filter_and_update_positionally(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.aupdate_one("users", {"_id": "a"}, {"$set": {"n": 2}})
    args, kwargs = adb["users"].call("update_one")
    assert args == ({"_id": "a"}, {"$set": {"n": 2}})
    assert kwargs == {"upsert": False}


async def test_aupdate_one_forwards_an_upsert(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.aupdate_one("users", {"_id": "a"}, {"$set": {}}, upsert=True)
    _, kwargs = adb["users"].call("update_one")
    assert kwargs == {"upsert": True}


async def test_aupdate_one_returns_the_modified_count(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    adb["users"].result = FakeWriteResult(modified_count=0)
    assert await service.aupdate_one("users", {}, {"$set": {}}) == 0


async def test_aupdate_many_returns_the_modified_count(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    adb["users"].result = FakeWriteResult(modified_count=7)
    assert await service.aupdate_many("users", {}, {"$set": {}}) == 7


def test_update_many_sends_the_same_arguments_as_its_async_twin(
    service: MongoDBService, db: FakeDatabase
) -> None:
    service.update_many("users", {"a": 1}, {"$set": {"b": 2}}, upsert=True)
    args, kwargs = db["users"].call("update_many")
    assert args == ({"a": 1}, {"$set": {"b": 2}})
    assert kwargs == {"upsert": True}


async def test_areplace_one_sends_the_replacement_whole(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.areplace_one("users", {"_id": "a"}, {"_id": "a", "n": 9})
    args, kwargs = adb["users"].call("replace_one")
    assert args == ({"_id": "a"}, {"_id": "a", "n": 9})
    assert kwargs == {"upsert": False}


def test_replace_one_returns_the_modified_count(
    service: MongoDBService, db: FakeDatabase
) -> None:
    db["users"].result = FakeWriteResult(modified_count=1)
    assert service.replace_one("users", {}, {}) == 1


# Delete


async def test_adelete_one_returns_the_deleted_count(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    adb["users"].result = FakeWriteResult(deleted_count=1)
    assert await service.adelete_one("users", {"_id": "a"}) == 1


async def test_adelete_many_returns_the_deleted_count(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    adb["users"].result = FakeWriteResult(deleted_count=4)
    assert await service.adelete_many("users", {}) == 4


def test_delete_many_sends_the_filter_positionally(
    service: MongoDBService, db: FakeDatabase
) -> None:
    service.delete_many("users", {"n": {"$lt": 3}})
    args, _ = db["users"].call("delete_many")
    assert args == ({"n": {"$lt": 3}},)


# Count and distinct


async def test_acount_turns_a_missing_filter_into_an_empty_one(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.acount("users")
    args, _ = adb["users"].call("count_documents")
    assert args == ({},)


async def test_acount_returns_what_the_driver_counted(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    adb["users"].count_value = 42
    assert await service.acount("users", {"active": True}) == 42


def test_count_turns_a_missing_filter_into_an_empty_one(
    service: MongoDBService, db: FakeDatabase
) -> None:
    service.count("users")
    args, _ = db["users"].call("count_documents")
    assert args == ({},)


async def test_adistinct_sends_the_key_positionally_and_the_filter_by_name(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.adistinct("users", "country", {"active": True})
    args, kwargs = adb["users"].call("distinct")
    assert args == ("country",)
    assert kwargs == {"filter": {"active": True}}


async def test_adistinct_turns_a_missing_filter_into_an_empty_one(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    await service.adistinct("users", "country")
    _, kwargs = adb["users"].call("distinct")
    assert kwargs == {"filter": {}}


def test_distinct_returns_the_values_the_driver_gave_back(
    service: MongoDBService, db: FakeDatabase
) -> None:
    db["users"].distinct_values = ["fr", "de"]
    assert service.distinct("users", "country") == ["fr", "de"]


# Aggregation


async def test_aaggregate_sends_the_pipeline_unchanged(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    pipeline = [{"$match": {"a": 1}}, {"$group": {"_id": "$b"}}]
    await service.aaggregate("users", pipeline)
    args, _ = adb["users"].call("aggregate")
    assert args == (pipeline,)


async def test_aaggregate_returns_every_stage_result(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    adb["users"].documents = [{"_id": "x", "n": 2}]
    assert await service.aaggregate("users", [{"$match": {}}]) == [{"_id": "x", "n": 2}]


def test_aggregate_materialises_the_cursor(
    service: MongoDBService, db: FakeDatabase
) -> None:
    db["users"].documents = [{"_id": "x"}]
    assert service.aggregate("users", [{"$match": {}}]) == [{"_id": "x"}]


# Pagination


async def test_apaginate_counts_with_the_same_filter_it_queries_with(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    adb["users"].count_value = 30
    await service.apaginate(
        "users", {"active": True}, PaginationParams(page=2, size=10)
    )
    count_args, _ = adb["users"].call("count_documents")
    find_args, _ = adb["users"].call("find")
    assert count_args == ({"active": True},)
    assert find_args == ({"active": True},)


async def test_apaginate_turns_the_page_number_into_a_skip(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    adb["users"].count_value = 30
    await service.apaginate("users", {}, PaginationParams(page=3, size=10))
    cursor = adb["users"].cursors[-1]
    assert (cursor.skipped, cursor.limited) == (20, 10)


async def test_apaginate_forwards_the_projection_and_sort(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    adb["users"].count_value = 5
    await service.apaginate(
        "users",
        {},
        PaginationParams(page=1, size=10),
        projection={"name": 1},
        sort=[("name", 1)],
    )
    _, kwargs = adb["users"].call("find")
    assert kwargs == {"projection": {"name": 1}}
    assert adb["users"].cursors[-1].sorted_by == [("name", 1)]


async def test_apaginate_does_not_query_at_all_for_a_page_past_the_end(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    adb["users"].count_value = 5
    page = await service.apaginate("users", {}, PaginationParams(page=4, size=10))
    assert page.data == []
    assert adb["users"].called("find") == 0


def test_paginate_turns_the_page_number_into_a_skip(
    service: MongoDBService, db: FakeDatabase
) -> None:
    db["users"].count_value = 30
    service.paginate("users", {}, PaginationParams(page=2, size=10))
    cursor = db["users"].cursors[-1]
    assert (cursor.skipped, cursor.limited) == (10, 10)


# Health and lifecycle


def test_the_health_check_pings_the_database(
    service: MongoDBService, db: FakeDatabase
) -> None:
    assert service.health_check() is True
    assert db.commands == ["ping"]


async def test_the_async_health_check_pings_the_database(
    service: MongoDBService, adb: AsyncFakeDatabase
) -> None:
    assert await service.ahealth_check() is True
    assert adb.commands == ["ping"]
