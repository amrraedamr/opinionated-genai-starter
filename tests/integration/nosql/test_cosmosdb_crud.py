"""Every Cosmos write and read against the emulator the compose file starts.

The adapter is thin, and what a real account adds over a fake is everything
the service decides: that `SELECT VALUE c` is what returns a document rather
than a row wrapping one, that an upsert is how this adapter inserts, and that
Cosmos hands back `_rid`, `_etag` and `_ts` alongside the fields that were
written.

One database holds the whole run and is deleted at the end of it; each test
names its container from `unique` and deletes that too, so reruns and
parallel runs never collide. Every container here is partitioned on `/id`,
which is the convention this service assumes — `test_cosmosdb_errors.py`
covers what happens to a container partitioned on anything else.
"""

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from azure.cosmos import CosmosClient, PartitionKey

from base.services.configuration.mixins import CosmosSettingsMixin
from base.services.nosql.cosmosdb import CosmosDBService
from tests.conftest import MakeSettings

pytestmark = pytest.mark.integration

# The emulator's well-known credentials, published by Microsoft. Not a secret.
ACCOUNT_URI = "http://127.0.0.1:8081/"
ACCOUNT_KEY = "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw=="


@pytest.fixture(scope="session")
def database() -> Iterator[str]:
    """One database for the whole run, deleted when it ends."""
    name = f"it_{uuid.uuid4().hex[:10]}"
    client = CosmosClient(ACCOUNT_URI, credential=ACCOUNT_KEY)
    client.create_database(name)
    yield name
    client.delete_database(name)
    client.close()


@pytest.fixture
async def cosmos(
    make_settings: MakeSettings, database: str
) -> AsyncIterator[CosmosDBService]:
    """A service pointed at the emulator, with both clients closed afterwards."""
    make_settings(
        CosmosSettingsMixin,
        azure_cosmos_account_uri=ACCOUNT_URI,
        azure_cosmos_account_key=ACCOUNT_KEY,
        azure_cosmos_database_name=database,
    )
    service = CosmosDBService()
    yield service
    await service.adispose()
    service.dispose()


def _container(
    cosmos: CosmosDBService, database: str, name: str, path: str
) -> Iterator[str]:
    """The database proxy is fetched twice: a test may close the client between."""
    cosmos._client.get_database(database).create_container(
        name, PartitionKey(path=path)
    )
    yield name
    cosmos._client.get_database(database).delete_container(name)


@pytest.fixture
def container(cosmos: CosmosDBService, database: str, unique: str) -> Iterator[str]:
    """A container partitioned on `/id`, deleted when the test ends."""
    yield from _container(cosmos, database, f"t_{unique}", "/id")


@pytest.fixture
def tenant_container(
    cosmos: CosmosDBService, database: str, unique: str
) -> Iterator[str]:
    """A container partitioned on `/pk`, which is not what this service assumes."""
    yield from _container(cosmos, database, f"p_{unique}", "/pk")


def fields(document: dict[str, Any] | None) -> dict[str, Any]:
    """A document without the bookkeeping Cosmos adds to every one of them."""
    assert document is not None
    return {k: v for k, v in document.items() if not k.startswith("_")}


# Insert


async def test_an_item_comes_back_with_the_fields_it_went_in_with(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "n": 1, "tag": "x"})
    found = await cosmos.afind_one(container, {"id": "a"})
    assert fields(found) == {"id": "a", "n": 1, "tag": "x"}


async def test_an_insert_returns_the_id_the_item_carried(
    cosmos: CosmosDBService, container: str
) -> None:
    assert await cosmos.ainsert_one(container, {"id": "chosen"}) == "chosen"


async def test_an_insert_is_an_upsert_so_the_same_id_twice_overwrites(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "n": 1})
    await cosmos.ainsert_one(container, {"id": "a", "n": 2})
    assert fields(await cosmos.afind_one(container, {"id": "a"})) == {"id": "a", "n": 2}
    assert await cosmos.acount(container) == 1


async def test_an_upsert_replaces_rather_than_merges(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "gone": True})
    await cosmos.ainsert_one(container, {"id": "a", "n": 1})
    assert "gone" not in fields(await cosmos.afind_one(container, {"id": "a"}))


async def test_insert_many_returns_one_id_per_document(
    cosmos: CosmosDBService, container: str
) -> None:
    ids = await cosmos.ainsert_many(container, [{"id": f"i{i}"} for i in range(3)])
    assert ids == ["i0", "i1", "i2"]


async def test_an_empty_batch_writes_nothing_and_says_so(
    cosmos: CosmosDBService, container: str
) -> None:
    assert await cosmos.ainsert_many(container, []) == []
    assert await cosmos.acount(container) == 0


def test_an_insert_returns_the_id_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    assert cosmos.insert_one(container, {"id": "a", "n": 1}) == "a"
    assert fields(cosmos.find_one(container, {"id": "a"})) == {"id": "a", "n": 1}


def test_insert_many_returns_one_id_per_document_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    assert cosmos.insert_many(container, [{"id": "a"}, {"id": "b"}]) == ["a", "b"]


# Find


async def test_find_one_reads_an_item_by_its_id(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "n": 1})
    assert fields(await cosmos.afind_one(container, {"id": "a"})) == {"id": "a", "n": 1}


async def test_find_one_returns_none_when_the_item_is_not_there(
    cosmos: CosmosDBService, container: str
) -> None:
    assert await cosmos.afind_one(container, {"id": "missing"}) is None


async def test_find_one_narrows_on_a_field_that_is_not_the_id(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(container, [{"id": "a", "n": 1}, {"id": "b", "n": 2}])
    assert fields(await cosmos.afind_one(container, {"n": 2})) == {"id": "b", "n": 2}


async def test_find_one_returns_none_when_the_query_matched_nothing(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "n": 1})
    assert await cosmos.afind_one(container, {"n": 99}) is None


async def test_find_one_applies_the_projection(
    cosmos: CosmosDBService, container: str
) -> None:
    """A point read cannot project, so a projection takes the query path."""
    await cosmos.ainsert_one(container, {"id": "a", "n": 1, "hidden": True})
    found = await cosmos.afind_one(container, {"id": "a"}, projection={"n": 1})
    assert fields(found) == {"n": 1}


async def test_find_many_returns_documents_rather_than_rows_wrapping_them(
    cosmos: CosmosDBService, container: str
) -> None:
    """`SELECT c FROM c` answers `{"c": {...}}`; only `VALUE` unwraps it."""
    await cosmos.ainsert_one(container, {"id": "a", "n": 1})
    found = await cosmos.afind_many(container, {})
    assert fields(found[0]) == {"id": "a", "n": 1}


async def test_find_many_returns_every_match(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(
        container, [{"id": f"i{i}", "g": i % 2} for i in range(5)]
    )
    assert len(await cosmos.afind_many(container, {"g": 0})) == 3


async def test_find_many_returns_nothing_when_the_filter_matches_nothing(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "n": 1})
    assert await cosmos.afind_many(container, {"n": 99}) == []


async def test_find_many_with_an_empty_filter_returns_everything(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(container, [{"id": f"i{i}"} for i in range(4)])
    assert len(await cosmos.afind_many(container, {})) == 4


async def test_find_many_sorts_in_the_direction_it_was_given(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(container, [{"id": f"i{i}", "n": i} for i in range(3)])
    found = await cosmos.afind_many(container, {}, sort=[("n", -1)])
    assert [d["n"] for d in found] == [2, 1, 0]


async def test_find_many_skips_and_limits(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(container, [{"id": f"i{i}", "n": i} for i in range(5)])
    found = await cosmos.afind_many(container, {}, sort=[("n", 1)], skip=1, limit=2)
    assert [d["n"] for d in found] == [1, 2]


async def test_a_skip_without_a_limit_returns_the_rest(
    cosmos: CosmosDBService, container: str
) -> None:
    """The borrowed `LIMIT 2147483647` has to be a limit Cosmos accepts."""
    await cosmos.ainsert_many(container, [{"id": f"i{i}", "n": i} for i in range(4)])
    found = await cosmos.afind_many(container, {}, sort=[("n", 1)], skip=1)
    assert [d["n"] for d in found] == [1, 2, 3]


async def test_a_limit_without_a_skip_returns_the_first_rows(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(container, [{"id": f"i{i}", "n": i} for i in range(4)])
    found = await cosmos.afind_many(container, {}, sort=[("n", 1)], limit=2)
    assert [d["n"] for d in found] == [0, 1]


async def test_find_many_applies_the_projection(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "n": 1, "hidden": True})
    found = await cosmos.afind_many(container, {}, projection={"id": 1, "n": 1})
    assert fields(found[0]) == {"id": "a", "n": 1}


def test_find_many_returns_documents_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_many(container, [{"id": "a", "n": 1}, {"id": "b", "n": 2}])
    found = cosmos.find_many(container, {}, sort=[("n", 1)])
    assert [fields(d) for d in found] == [{"id": "a", "n": 1}, {"id": "b", "n": 2}]


def test_find_one_reads_an_item_by_its_id_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_one(container, {"id": "a", "n": 1})
    assert fields(cosmos.find_one(container, {"id": "a"})) == {"id": "a", "n": 1}


# Update


async def test_an_update_changes_only_the_fields_it_names(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "n": 1, "keep": True})
    assert await cosmos.aupdate_one(container, {"id": "a"}, {"$set": {"n": 2}}) == 1
    assert fields(await cosmos.afind_one(container, {"id": "a"})) == {
        "id": "a",
        "n": 2,
        "keep": True,
    }


async def test_an_update_without_an_operator_is_treated_as_a_set(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "n": 1, "keep": True})
    await cosmos.aupdate_one(container, {"id": "a"}, {"n": 5})
    found = await cosmos.afind_one(container, {"id": "a"})
    assert found is not None and found["n"] == 5 and found["keep"] is True


async def test_an_update_matching_nothing_changes_nothing(
    cosmos: CosmosDBService, container: str
) -> None:
    assert await cosmos.aupdate_one(container, {"id": "gone"}, {"$set": {"n": 1}}) == 0
    assert await cosmos.acount(container) == 0


async def test_an_update_can_insert_what_it_did_not_find(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.aupdate_one(container, {"id": "a"}, {"$set": {"n": 1}}, upsert=True)
    assert fields(await cosmos.afind_one(container, {"id": "a"})) == {"id": "a", "n": 1}


async def test_an_update_matching_one_of_several_changes_only_that_one(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(container, [{"id": "a", "g": 1}, {"id": "b", "g": 1}])
    assert await cosmos.aupdate_one(container, {"g": 1}, {"$set": {"n": 5}}) == 1
    assert await cosmos.acount(container, {"n": 5}) == 1


async def test_update_many_changes_every_match(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(container, [{"id": f"i{i}", "g": 1} for i in range(3)])
    assert await cosmos.aupdate_many(container, {"g": 1}, {"$set": {"n": 5}}) == 3
    assert await cosmos.acount(container, {"n": 5}) == 3


async def test_update_many_matching_nothing_changes_nothing(
    cosmos: CosmosDBService, container: str
) -> None:
    assert await cosmos.aupdate_many(container, {"g": 9}, {"$set": {"n": 1}}) == 0


async def test_update_many_with_upsert_inserts_a_single_document(
    cosmos: CosmosDBService, container: str
) -> None:
    assert (
        await cosmos.aupdate_many(
            container, {"id": "a"}, {"$set": {"n": 1}}, upsert=True
        )
        == 1
    )
    assert await cosmos.acount(container) == 1


def test_an_update_changes_only_the_fields_it_names_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_one(container, {"id": "a", "n": 1, "keep": True})
    assert cosmos.update_one(container, {"id": "a"}, {"$set": {"n": 2}}) == 1
    assert fields(cosmos.find_one(container, {"id": "a"})) == {
        "id": "a",
        "n": 2,
        "keep": True,
    }


def test_update_many_changes_every_match_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_many(container, [{"id": "a", "g": 1}, {"id": "b", "g": 1}])
    assert cosmos.update_many(container, {"g": 1}, {"$set": {"n": 5}}) == 2


# Replace


async def test_a_replacement_swaps_the_whole_document(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "old": True, "n": 1})
    assert await cosmos.areplace_one(container, {"id": "a"}, {"id": "a", "new": 1}) == 1
    assert fields(await cosmos.afind_one(container, {"id": "a"})) == {
        "id": "a",
        "new": 1,
    }


async def test_a_replacement_without_an_id_keeps_the_one_it_replaced(
    cosmos: CosmosDBService, container: str
) -> None:
    """Without it Cosmos refuses the write: a body must carry an id."""
    await cosmos.ainsert_one(container, {"id": "a", "old": True})
    assert await cosmos.areplace_one(container, {"old": True}, {"n": 1}) == 1
    assert fields(await cosmos.afind_one(container, {"id": "a"})) == {"id": "a", "n": 1}


async def test_a_replacement_matching_nothing_writes_nothing(
    cosmos: CosmosDBService, container: str
) -> None:
    assert await cosmos.areplace_one(container, {"id": "gone"}, {"n": 1}) == 0
    assert await cosmos.acount(container) == 0


async def test_a_replacement_can_insert_what_it_did_not_find(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.areplace_one(container, {"id": "a"}, {"n": 1}, upsert=True)
    assert fields(await cosmos.afind_one(container, {"id": "a"})) == {"id": "a", "n": 1}


def test_a_replacement_swaps_the_whole_document_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_one(container, {"id": "a", "old": True})
    assert cosmos.replace_one(container, {"id": "a"}, {"id": "a", "new": 1}) == 1
    assert fields(cosmos.find_one(container, {"id": "a"})) == {"id": "a", "new": 1}


def test_a_replacement_without_an_id_keeps_the_one_it_replaced_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_one(container, {"id": "a", "old": True})
    assert cosmos.replace_one(container, {"old": True}, {"n": 1}) == 1
    assert fields(cosmos.find_one(container, {"id": "a"})) == {"id": "a", "n": 1}


# Delete


async def test_a_delete_removes_the_item(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a"})
    assert await cosmos.adelete_one(container, {"id": "a"}) == 1
    assert await cosmos.afind_one(container, {"id": "a"}) is None


async def test_a_delete_matching_nothing_removes_nothing(
    cosmos: CosmosDBService, container: str
) -> None:
    assert await cosmos.adelete_one(container, {"id": "gone"}) == 0


async def test_a_delete_matching_one_of_several_removes_only_that_one(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(container, [{"id": "a", "g": 1}, {"id": "b", "g": 1}])
    assert await cosmos.adelete_one(container, {"g": 1}) == 1
    assert await cosmos.acount(container) == 1


async def test_delete_many_removes_every_match(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(container, [{"id": f"i{i}", "g": 1} for i in range(4)])
    assert await cosmos.adelete_many(container, {"g": 1}) == 4
    assert await cosmos.acount(container) == 0


async def test_delete_many_with_an_empty_filter_empties_the_container(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_many(container, [{"id": "a"}, {"id": "b"}])
    assert await cosmos.adelete_many(container, {}) == 2
    assert await cosmos.acount(container) == 0


async def test_delete_many_matching_nothing_removes_nothing(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a"})
    assert await cosmos.adelete_many(container, {"g": 9}) == 0
    assert await cosmos.acount(container) == 1


def test_a_delete_removes_the_item_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_one(container, {"id": "a"})
    assert cosmos.delete_one(container, {"id": "a"}) == 1
    assert cosmos.find_one(container, {"id": "a"}) is None


def test_delete_many_removes_every_match_on_the_sync_path(
    cosmos: CosmosDBService, container: str
) -> None:
    cosmos.insert_many(container, [{"id": "a", "g": 1}, {"id": "b", "g": 1}])
    assert cosmos.delete_many(container, {"g": 1}) == 2


# What the store adds, and what survives it


async def test_cosmos_adds_its_own_bookkeeping_to_every_item(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a"})
    found = await cosmos.afind_one(container, {"id": "a"})
    assert found is not None
    assert {"_rid", "_etag", "_ts"} <= set(found)


async def test_an_item_read_back_can_be_written_again_unchanged(
    cosmos: CosmosDBService, container: str
) -> None:
    """The update and delete paths all write back what a read handed them."""
    await cosmos.ainsert_one(container, {"id": "a", "n": 1})
    found = await cosmos.afind_one(container, {"id": "a"})
    assert found is not None
    assert await cosmos.ainsert_one(container, found) == "a"


async def test_unicode_keys_and_values_survive_the_round_trip(
    cosmos: CosmosDBService, container: str
) -> None:
    document = {"id": "u", "名前": "テスト", "ключ": "значение", "clé": "valeur"}
    await cosmos.ainsert_one(container, document)
    assert fields(await cosmos.afind_one(container, {"id": "u"})) == document


async def test_emoji_survive_the_round_trip(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "e", "text": "ship it 🚀🙈🎉"})
    found = await cosmos.afind_one(container, {"id": "e"})
    assert found is not None and found["text"] == "ship it 🚀🙈🎉"


async def test_a_unicode_value_can_be_filtered_on(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "u", "名前": "テスト"})
    assert await cosmos.acount(container, {"名前": "テスト"}) == 1


async def test_a_deeply_nested_document_round_trips(
    cosmos: CosmosDBService, container: str
) -> None:
    document = {"id": "n", "a": {"b": {"c": {"d": [1, {"e": "deep"}]}}}}
    await cosmos.ainsert_one(container, document)
    assert fields(await cosmos.afind_one(container, {"id": "n"})) == document


async def test_a_null_is_stored_as_null_rather_than_dropped(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "z", "maybe": None})
    found = await cosmos.afind_one(container, {"id": "z"})
    assert found is not None and "maybe" in found and found["maybe"] is None


async def test_an_empty_array_round_trips(
    cosmos: CosmosDBService, container: str
) -> None:
    await cosmos.ainsert_one(container, {"id": "a", "tags": []})
    found = await cosmos.afind_one(container, {"id": "a"})
    assert found is not None and found["tags"] == []


async def test_a_mixed_type_array_round_trips(
    cosmos: CosmosDBService, container: str
) -> None:
    values = [1, "two", 3.5, True, None, {"four": 4}, []]
    await cosmos.ainsert_one(container, {"id": "m", "values": values})
    found = await cosmos.afind_one(container, {"id": "m"})
    assert found is not None and found["values"] == values


async def test_a_stored_pydantic_model_round_trips_through_model_dump(
    cosmos: CosmosDBService, container: str
) -> None:
    """How the repository stores models: dumped straight in, with no DTO."""
    from base.models.view_models.vector_search import VectorRecord

    record = VectorRecord(id="r", vector=[0.1, 0.2], metadata={"text": "héllo 🎉"})
    await cosmos.ainsert_one(container, record.model_dump())
    found = await cosmos.afind_one(container, {"id": "r"})
    assert VectorRecord(**fields(found)) == record


# Health


async def test_the_async_health_check_reaches_the_emulator(
    cosmos: CosmosDBService,
) -> None:
    assert await cosmos.ahealth_check() is True


def test_the_health_check_reaches_the_emulator(cosmos: CosmosDBService) -> None:
    assert cosmos.health_check() is True
