"""Every MongoDB method against the real server, async first and sync beside it.

The adapter is thin, so what a real server adds over a fake is the parts the
driver decides: what an id looks like coming back, what a modified count is
when nothing changed, what a projection actually removes. Each test here is
one method's observable behaviour rather than its call shape.
"""

import pytest

from base.services.nosql.mongodb import MongoDBService

pytestmark = pytest.mark.integration


# Find


async def test_afind_one_returns_the_document_that_was_inserted(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a", "n": 1})
    assert await mongo.afind_one(collection, {"_id": "a"}) == {"_id": "a", "n": 1}


async def test_afind_one_returns_none_when_nothing_matches(
    mongo: MongoDBService, collection: str
) -> None:
    assert await mongo.afind_one(collection, {"_id": "missing"}) is None


async def test_afind_one_applies_the_projection(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a", "n": 1, "hidden": True})
    found = await mongo.afind_one(collection, {"_id": "a"}, projection={"n": 1})
    assert found == {"_id": "a", "n": 1}


def test_find_one_returns_the_document_that_was_inserted(
    mongo: MongoDBService, collection: str
) -> None:
    mongo.insert_one(collection, {"_id": "a", "n": 1})
    assert mongo.find_one(collection, {"_id": "a"}) == {"_id": "a", "n": 1}


async def test_afind_many_returns_every_match(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"i": i} for i in range(5)])
    assert len(await mongo.afind_many(collection, {"i": {"$lt": 3}})) == 3


async def test_afind_many_returns_nothing_when_the_filter_matches_nothing(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"i": 1}])
    assert await mongo.afind_many(collection, {"i": 99}) == []


async def test_afind_many_sorts_in_the_direction_it_was_given(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"i": 3}, {"i": 1}, {"i": 2}])
    found = await mongo.afind_many(collection, {}, sort=[("i", -1)])
    assert [d["i"] for d in found] == [3, 2, 1]


async def test_afind_many_skips_and_limits(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"i": i} for i in range(5)])
    found = await mongo.afind_many(collection, {}, sort=[("i", 1)], skip=1, limit=2)
    assert [d["i"] for d in found] == [1, 2]


def test_find_many_sorts_in_the_direction_it_was_given(
    mongo: MongoDBService, collection: str
) -> None:
    mongo.insert_many(collection, [{"i": 2}, {"i": 1}])
    assert [d["i"] for d in mongo.find_many(collection, {}, sort=[("i", 1)])] == [1, 2]


# Insert


async def test_ainsert_one_returns_the_id_the_server_stored(
    mongo: MongoDBService, collection: str
) -> None:
    identifier = await mongo.ainsert_one(collection, {"n": 1})
    found = await mongo.afind_many(collection, {})
    assert str(found[0]["_id"]) == identifier


async def test_ainsert_one_honours_an_explicit_id(
    mongo: MongoDBService, collection: str
) -> None:
    assert await mongo.ainsert_one(collection, {"_id": "chosen"}) == "chosen"


async def test_ainsert_many_returns_one_id_per_document(
    mongo: MongoDBService, collection: str
) -> None:
    ids = await mongo.ainsert_many(collection, [{"i": 1}, {"i": 2}, {"i": 3}])
    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_insert_many_returns_one_id_per_document(
    mongo: MongoDBService, collection: str
) -> None:
    assert len(mongo.insert_many(collection, [{"i": 1}, {"i": 2}])) == 2


# Update


async def test_aupdate_one_changes_a_single_document(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"g": 1, "n": 0}, {"g": 1, "n": 0}])
    assert await mongo.aupdate_one(collection, {"g": 1}, {"$set": {"n": 5}}) == 1
    assert await mongo.acount(collection, {"n": 5}) == 1


async def test_aupdate_one_reports_nothing_modified_when_nothing_matched(
    mongo: MongoDBService, collection: str
) -> None:
    assert (
        await mongo.aupdate_one(collection, {"missing": True}, {"$set": {"n": 1}}) == 0
    )


async def test_aupdate_one_reports_nothing_modified_when_the_value_is_unchanged(
    mongo: MongoDBService, collection: str
) -> None:
    """A matched document and a modified document are different counts."""
    await mongo.ainsert_one(collection, {"_id": "a", "n": 1})
    assert await mongo.aupdate_one(collection, {"_id": "a"}, {"$set": {"n": 1}}) == 0


async def test_aupdate_one_can_insert_what_it_did_not_find(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.aupdate_one(collection, {"_id": "a"}, {"$set": {"n": 1}}, upsert=True)
    assert await mongo.afind_one(collection, {"_id": "a"}) == {"_id": "a", "n": 1}


async def test_aupdate_many_changes_every_match(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"g": 1, "n": 0} for _ in range(3)])
    assert await mongo.aupdate_many(collection, {"g": 1}, {"$set": {"n": 5}}) == 3


def test_update_many_changes_every_match(
    mongo: MongoDBService, collection: str
) -> None:
    mongo.insert_many(collection, [{"g": 1} for _ in range(2)])
    assert mongo.update_many(collection, {"g": 1}, {"$set": {"n": 5}}) == 2


# Replace


async def test_areplace_one_swaps_the_whole_document(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a", "old": True, "n": 1})
    assert await mongo.areplace_one(collection, {"_id": "a"}, {"new": True}) == 1
    assert await mongo.afind_one(collection, {"_id": "a"}) == {"_id": "a", "new": True}


async def test_areplace_one_reports_nothing_modified_when_nothing_matched(
    mongo: MongoDBService, collection: str
) -> None:
    assert await mongo.areplace_one(collection, {"_id": "missing"}, {"n": 1}) == 0


async def test_areplace_one_can_insert_what_it_did_not_find(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.areplace_one(collection, {"_id": "a"}, {"n": 1}, upsert=True)
    assert await mongo.acount(collection, {"n": 1}) == 1


def test_replace_one_swaps_the_whole_document(
    mongo: MongoDBService, collection: str
) -> None:
    mongo.insert_one(collection, {"_id": "a", "old": True})
    mongo.replace_one(collection, {"_id": "a"}, {"new": True})
    assert mongo.find_one(collection, {"_id": "a"}) == {"_id": "a", "new": True}


# Delete


async def test_adelete_one_removes_a_single_document(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"g": 1}, {"g": 1}])
    assert await mongo.adelete_one(collection, {"g": 1}) == 1
    assert await mongo.acount(collection) == 1


async def test_adelete_one_removes_nothing_when_nothing_matches(
    mongo: MongoDBService, collection: str
) -> None:
    assert await mongo.adelete_one(collection, {"missing": True}) == 0


async def test_adelete_many_removes_every_match(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"g": 1} for _ in range(4)])
    assert await mongo.adelete_many(collection, {"g": 1}) == 4
    assert await mongo.acount(collection) == 0


async def test_adelete_many_with_an_empty_filter_empties_the_collection(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"i": 1}, {"i": 2}])
    assert await mongo.adelete_many(collection, {}) == 2


def test_delete_many_removes_every_match(
    mongo: MongoDBService, collection: str
) -> None:
    mongo.insert_many(collection, [{"g": 1}, {"g": 1}])
    assert mongo.delete_many(collection, {"g": 1}) == 2


# Count and distinct


async def test_acount_counts_the_whole_collection_without_a_filter(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"i": i} for i in range(3)])
    assert await mongo.acount(collection) == 3


async def test_acount_narrows_with_the_filter(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"i": i} for i in range(5)])
    assert await mongo.acount(collection, {"i": {"$gte": 3}}) == 2


async def test_acount_of_a_collection_that_does_not_exist_is_zero(
    mongo: MongoDBService, collection: str
) -> None:
    assert await mongo.acount(collection) == 0


def test_count_counts_the_whole_collection(
    mongo: MongoDBService, collection: str
) -> None:
    mongo.insert_many(collection, [{"i": 1}, {"i": 2}])
    assert mongo.count(collection) == 2


async def test_adistinct_returns_each_value_once(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"c": "fr"}, {"c": "fr"}, {"c": "de"}])
    assert sorted(await mongo.adistinct(collection, "c")) == ["de", "fr"]


async def test_adistinct_narrows_with_the_filter(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(
        collection, [{"c": "fr", "ok": True}, {"c": "de", "ok": False}]
    )
    assert await mongo.adistinct(collection, "c", {"ok": True}) == ["fr"]


def test_distinct_returns_each_value_once(
    mongo: MongoDBService, collection: str
) -> None:
    mongo.insert_many(collection, [{"c": "fr"}, {"c": "fr"}])
    assert mongo.distinct(collection, "c") == ["fr"]


# Aggregation


async def test_aaggregate_runs_the_pipeline_on_the_server(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(
        collection, [{"g": "a", "n": 1}, {"g": "a", "n": 2}, {"g": "b", "n": 4}]
    )
    rows = await mongo.aaggregate(
        collection,
        [
            {"$group": {"_id": "$g", "total": {"$sum": "$n"}}},
            {"$sort": {"_id": 1}},
        ],
    )
    assert rows == [{"_id": "a", "total": 3}, {"_id": "b", "total": 4}]


async def test_aaggregate_of_an_empty_collection_returns_nothing(
    mongo: MongoDBService, collection: str
) -> None:
    assert await mongo.aaggregate(collection, [{"$match": {}}]) == []


async def test_an_empty_pipeline_returns_every_document(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a"})
    assert await mongo.aaggregate(collection, []) == [{"_id": "a"}]


def test_aggregate_runs_the_pipeline_on_the_server(
    mongo: MongoDBService, collection: str
) -> None:
    mongo.insert_many(collection, [{"n": 1}, {"n": 2}])
    assert mongo.aggregate(collection, [{"$count": "total"}]) == [{"total": 2}]


# Health


async def test_the_async_health_check_reaches_the_server(
    mongo: MongoDBService,
) -> None:
    assert await mongo.ahealth_check() is True


def test_the_health_check_reaches_the_server(mongo: MongoDBService) -> None:
    assert mongo.health_check() is True


def test_the_service_works_against_a_server_without_authentication(
    mongo: MongoDBService,
) -> None:
    """The whole point of the connection-string fix, proved end to end."""
    assert "@" not in mongo._client._connection_string
    assert mongo.health_check() is True
