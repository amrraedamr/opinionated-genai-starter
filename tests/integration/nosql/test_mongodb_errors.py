"""What a real MongoDB refuses, and which domain error the caller sees.

The adapter's job at a failure is to replace pymongo's exception with one
from the document store family, so a caller catches one family whichever
backend is behind it. A duplicate key is a conflict, a query the server will
not parse is a bad request, and a `$inc` is applied exactly once because the
update methods are not repeated.

Which server codes count as retryable is covered by the error-handling tests
and is not repeated here.
"""

import pytest

from base.models.view_models.exceptions.nosql import (
    NoSqlBadRequestError,
    NoSqlConflictError,
    NoSqlError,
    NoSqlUnexpectedError,
)
from base.services.nosql.mongodb import MongoDBService

pytestmark = pytest.mark.integration


async def test_a_duplicate_id_is_a_conflict(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a"})
    with pytest.raises(NoSqlConflictError):
        await mongo.ainsert_one(collection, {"_id": "a"})


async def test_a_duplicate_id_carries_the_servers_own_code(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a"})
    with pytest.raises(NoSqlConflictError) as caught:
        await mongo.ainsert_one(collection, {"_id": "a"})
    assert caught.value.status_code == 11000


def test_a_duplicate_id_is_a_conflict_on_the_sync_path(
    mongo: MongoDBService, collection: str
) -> None:
    mongo.insert_one(collection, {"_id": "a"})
    with pytest.raises(NoSqlConflictError):
        mongo.insert_one(collection, {"_id": "a"})


async def test_a_duplicate_in_a_batch_is_reported_as_a_conflict(
    mongo: MongoDBService, collection: str
) -> None:
    """The same clash reads the same whether it arrives alone or in a batch.

    pymongo wraps per-write failures in `BulkWriteError`, an
    `OperationFailure` carrying code 65 with the real 11000 inside
    `writeErrors`, so a caller catching `NoSqlConflictError` used to miss it.
    """
    with pytest.raises(NoSqlConflictError) as caught:
        await mongo.ainsert_many(collection, [{"_id": "a"}, {"_id": "a"}])
    assert "11000" in str(caught.value)


async def test_an_ordered_batch_keeps_what_it_wrote_before_the_duplicate(
    mongo: MongoDBService, collection: str
) -> None:
    """Pins that a failed batch is not rolled back: there is no transaction."""
    with pytest.raises(NoSqlError):
        await mongo.ainsert_many(collection, [{"_id": "a"}, {"_id": "b"}, {"_id": "b"}])
    assert await mongo.acount(collection) == 2


async def test_an_unknown_query_operator_is_a_bad_request(
    mongo: MongoDBService, collection: str
) -> None:
    with pytest.raises(NoSqlBadRequestError):
        await mongo.afind_many(collection, {"n": {"$bogus": 1}})


async def test_a_bad_filter_carries_the_servers_own_code(
    mongo: MongoDBService, collection: str
) -> None:
    with pytest.raises(NoSqlBadRequestError) as caught:
        await mongo.afind_many(collection, {"n": {"$bogus": 1}})
    assert caught.value.status_code == 2


async def test_an_unknown_update_operator_is_a_bad_request(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a"})
    with pytest.raises(NoSqlBadRequestError):
        await mongo.aupdate_one(collection, {"_id": "a"}, {"$bogus": {"n": 1}})


async def test_an_update_without_an_operator_is_refused_by_the_driver(
    mongo: MongoDBService, collection: str
) -> None:
    """A replacement document is `replace_one`'s argument, not `update_one`'s.

    pymongo rejects it before the wire with a `ValueError`, which is not a
    `PyMongoError`, so it surfaces as the unexpected member of the family.
    """
    await mongo.ainsert_one(collection, {"_id": "a"})
    with pytest.raises(NoSqlUnexpectedError, match=r"\$ operators"):
        await mongo.aupdate_one(collection, {"_id": "a"}, {"n": 1})


async def test_a_malformed_aggregation_stage_is_a_bad_request(
    mongo: MongoDBService, collection: str
) -> None:
    with pytest.raises(NoSqlBadRequestError):
        await mongo.aaggregate(collection, [{"$bogus": {}}])


async def test_every_failure_belongs_to_the_document_store_family(
    mongo: MongoDBService, collection: str
) -> None:
    """No pymongo exception reaches a caller, whatever went wrong."""
    with pytest.raises(NoSqlError):
        await mongo.afind_many(collection, {"n": {"$bogus": 1}})


async def test_an_empty_batch_writes_nothing_and_says_so(
    mongo: MongoDBService, collection: str
) -> None:
    """Inserting no documents is a no-op, not a driver `TypeError`."""
    assert await mongo.ainsert_many(collection, []) == []
    assert await mongo.acount(collection, {}) == 0


def test_an_empty_batch_writes_nothing_on_the_sync_path(
    mongo: MongoDBService, collection: str
) -> None:
    assert mongo.insert_many(collection, []) == []
    assert mongo.count(collection, {}) == 0


# Non-idempotent updates


async def test_an_inc_is_applied_exactly_once(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a", "hits": 0})
    await mongo.aupdate_one(collection, {"_id": "a"}, {"$inc": {"hits": 1}})
    found = await mongo.afind_one(collection, {"_id": "a"})
    assert found is not None and found["hits"] == 1


async def test_an_inc_accumulates_across_separate_calls(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a", "hits": 0})
    for _ in range(3):
        await mongo.aupdate_one(collection, {"_id": "a"}, {"$inc": {"hits": 2}})
    found = await mongo.afind_one(collection, {"_id": "a"})
    assert found is not None and found["hits"] == 6


async def test_an_inc_across_many_documents_is_applied_once_each(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"g": 1, "hits": 0} for _ in range(3)])
    assert await mongo.aupdate_many(collection, {"g": 1}, {"$inc": {"hits": 1}}) == 3
    assert await mongo.acount(collection, {"hits": 1}) == 3


async def test_an_inc_on_a_missing_field_starts_from_zero(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a"})
    await mongo.aupdate_one(collection, {"_id": "a"}, {"$inc": {"hits": 5}})
    found = await mongo.afind_one(collection, {"_id": "a"})
    assert found is not None and found["hits"] == 5


async def test_an_inc_on_a_non_numeric_field_is_a_bad_request(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a", "hits": "many"})
    with pytest.raises(NoSqlBadRequestError):
        await mongo.aupdate_one(collection, {"_id": "a"}, {"$inc": {"hits": 1}})


def test_the_sync_inc_is_applied_exactly_once(
    mongo: MongoDBService, collection: str
) -> None:
    mongo.insert_one(collection, {"_id": "a", "hits": 0})
    mongo.update_one(collection, {"_id": "a"}, {"$inc": {"hits": 1}})
    found = mongo.find_one(collection, {"_id": "a"})
    assert found is not None and found["hits"] == 1
