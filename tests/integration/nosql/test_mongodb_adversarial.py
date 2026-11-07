"""The documents and filters that are easy to get wrong, against a real server.

Unicode keys, emoji values, nulls, deep nesting and a document near the
sixteen-megabyte limit all go through the same `model_dump()` path the
repository uses to store Pydantic models, so what BSON does with them is what
a caller gets back. The filters here are the ones whose semantics surprise:
`None` matches a missing field, and a projection can remove `_id`.
"""

import pytest

from base.models.view_models.exceptions.nosql import NoSqlError
from base.services.nosql.mongodb import MongoDBService

pytestmark = pytest.mark.integration


# Unusual documents


async def test_a_document_with_unicode_keys_and_values_round_trips(
    mongo: MongoDBService, collection: str
) -> None:
    document = {"_id": "u", "名前": "テスト", "ключ": "значение", "clé": "valeur"}
    await mongo.ainsert_one(collection, document)
    assert await mongo.afind_one(collection, {"_id": "u"}) == document


async def test_emoji_survive_the_round_trip(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "e", "text": "ship it 🚀🙈🎉"})
    found = await mongo.afind_one(collection, {"_id": "e"})
    assert found is not None and found["text"] == "ship it 🚀🙈🎉"


async def test_a_unicode_value_can_be_filtered_on(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "u", "名前": "テスト"})
    assert await mongo.acount(collection, {"名前": "テスト"}) == 1


async def test_a_deeply_nested_document_round_trips(
    mongo: MongoDBService, collection: str
) -> None:
    document = {"_id": "n", "a": {"b": {"c": {"d": [1, {"e": "deep"}]}}}}
    await mongo.ainsert_one(collection, document)
    assert await mongo.afind_one(collection, {"_id": "n"}) == document


async def test_a_nested_field_can_be_filtered_with_dotted_notation(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "n", "a": {"b": {"c": 1}}})
    assert await mongo.acount(collection, {"a.b.c": 1}) == 1


async def test_a_null_value_is_stored_as_null_rather_than_dropped(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "z", "maybe": None})
    found = await mongo.afind_one(collection, {"_id": "z"})
    assert found is not None and "maybe" in found and found["maybe"] is None


async def test_filtering_on_none_also_matches_a_missing_field(
    mongo: MongoDBService, collection: str
) -> None:
    """Mongo's own semantics, and a common surprise: absent equals null."""
    await mongo.ainsert_many(collection, [{"_id": "a", "x": None}, {"_id": "b"}])
    assert await mongo.acount(collection, {"x": None}) == 2


async def test_an_explicit_null_can_be_told_apart_from_a_missing_field(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"_id": "a", "x": None}, {"_id": "b"}])
    assert await mongo.acount(collection, {"x": {"$type": "null"}}) == 1


async def test_an_empty_document_is_stored_with_only_its_generated_id(
    mongo: MongoDBService, collection: str
) -> None:
    identifier = await mongo.ainsert_one(collection, {})
    found = await mongo.afind_many(collection, {})
    assert len(found) == 1
    assert str(found[0]["_id"]) == identifier


async def test_an_empty_array_round_trips(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a", "tags": []})
    found = await mongo.afind_one(collection, {"_id": "a"})
    assert found is not None and found["tags"] == []


async def test_a_large_document_round_trips(
    mongo: MongoDBService, collection: str
) -> None:
    """A megabyte of text, comfortably inside the sixteen-megabyte limit."""
    blob = "x" * 1_000_000
    await mongo.ainsert_one(collection, {"_id": "big", "blob": blob})
    found = await mongo.afind_one(collection, {"_id": "big"})
    assert found is not None and len(found["blob"]) == 1_000_000


async def test_a_document_over_the_wire_limit_is_refused(
    mongo: MongoDBService, collection: str
) -> None:
    with pytest.raises(NoSqlError):
        await mongo.ainsert_one(collection, {"_id": "huge", "blob": "x" * 17_000_000})


async def test_a_mixed_type_array_round_trips(
    mongo: MongoDBService, collection: str
) -> None:
    values = [1, "two", 3.5, True, None, {"four": 4}, []]
    await mongo.ainsert_one(collection, {"_id": "m", "values": values})
    found = await mongo.afind_one(collection, {"_id": "m"})
    assert found is not None and found["values"] == values


async def test_a_stored_pydantic_model_round_trips_through_model_dump(
    mongo: MongoDBService, collection: str
) -> None:
    """How the repository stores models: dumped straight in, with no DTO."""
    from base.models.view_models.vector_search import VectorRecord

    record = VectorRecord(id="a", vector=[0.1, 0.2], metadata={"text": "héllo 🎉"})
    await mongo.ainsert_one(collection, {"_id": "r", **record.model_dump()})
    found = await mongo.afind_one(collection, {"_id": "r"})
    assert found is not None
    assert VectorRecord(**{k: v for k, v in found.items() if k != "_id"}) == record


# Filters and projections


async def test_a_filter_matching_nothing_returns_an_empty_list(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"i": 1}, {"i": 2}])
    assert await mongo.afind_many(collection, {"i": 99}) == []


async def test_a_filter_on_a_collection_that_does_not_exist_returns_nothing(
    mongo: MongoDBService, collection: str
) -> None:
    assert await mongo.afind_many(collection, {"anything": True}) == []


async def test_an_empty_filter_matches_everything(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"i": 1}, {"i": 2}])
    assert len(await mongo.afind_many(collection, {})) == 2


async def test_a_projection_can_exclude_the_id(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a", "n": 1, "hidden": True})
    found = await mongo.afind_one(
        collection, {"_id": "a"}, projection={"_id": 0, "n": 1}
    )
    assert found == {"n": 1}


async def test_a_projection_keeps_the_id_unless_told_otherwise(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a", "n": 1, "hidden": True})
    found = await mongo.afind_one(collection, {"_id": "a"}, projection={"n": 1})
    assert found == {"_id": "a", "n": 1}


async def test_a_projection_of_a_missing_field_simply_omits_it(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a", "n": 1})
    found = await mongo.afind_one(
        collection, {"_id": "a"}, projection={"_id": 0, "absent": 1}
    )
    assert found == {}


async def test_a_projection_reaches_a_nested_field(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_one(collection, {"_id": "a", "outer": {"keep": 1, "drop": 2}})
    found = await mongo.afind_one(
        collection, {"_id": "a"}, projection={"_id": 0, "outer.keep": 1}
    )
    assert found == {"outer": {"keep": 1}}


async def test_distinct_of_a_field_nothing_has_returns_nothing(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"i": 1}])
    assert await mongo.adistinct(collection, "absent") == []


async def test_distinct_counts_a_null_as_a_value(
    mongo: MongoDBService, collection: str
) -> None:
    await mongo.ainsert_many(collection, [{"c": "fr"}, {"c": None}])
    assert sorted(str(v) for v in await mongo.adistinct(collection, "c")) == [
        "None",
        "fr",
    ]
