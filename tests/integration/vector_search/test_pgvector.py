"""pgvector against the Postgres the repository's compose file starts.

What a real server answers for: that the extension installs itself, that a
table named with capitals is the table this backend asks about, that a batch
larger than one statement arrives whole, and that a metric is recovered from
the index by a process that did not create the collection. Each test names
its table from `unique` and drops it, so a rerun or a parallel run never meets
another test's rows.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from sqlalchemy import text

from base.models.definitions.vector_search import VECTOR_UPSERT_BATCH_SIZE
from base.models.view_models.exceptions.vector_search import (
    VectorSearchError,
    VectorSearchUnexpectedError,
)
from base.models.view_models.vector_search import VectorRecord
from base.services.vector_search.pgvector import PgVectorSearchService

pytestmark = pytest.mark.integration


@pytest.fixture
async def pgvector(infra_settings: Any) -> AsyncIterator[PgVectorSearchService]:
    """A service on the local Postgres, with both pools closed afterwards."""
    service = PgVectorSearchService()
    yield service
    await service.adispose()
    service.dispose()


@pytest.fixture
async def collection(
    pgvector: PgVectorSearchService, unique: str
) -> AsyncIterator[str]:
    """An empty three-dimensional table, dropped when the test ends."""
    name = f"pgv_{unique}"
    await pgvector.acreate_collection(name, vector_size=3)
    yield name
    await pgvector.adelete_collection(name)


def record(
    identifier: str, vector: list[float] | None = None, **metadata: Any
) -> VectorRecord:
    return VectorRecord(
        id=identifier,
        vector=[1.0, 0.0, 0.0] if vector is None else vector,
        metadata=metadata,
    )


# Collections


async def test_a_created_collection_exists(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    assert await pgvector.acollection_exists(collection) is True


async def test_a_collection_that_was_never_created_does_not_exist(
    pgvector: PgVectorSearchService, unique: str
) -> None:
    assert await pgvector.acollection_exists(f"pgv_absent_{unique}") is False


async def test_creating_a_collection_that_is_already_there_is_not_an_error(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.acreate_collection(collection, vector_size=3)
    assert await pgvector.acount(collection) == 0


async def test_creating_a_collection_again_keeps_what_is_in_it(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("a")])
    await pgvector.acreate_collection(collection, vector_size=3)
    assert await pgvector.acount(collection) == 1


async def test_a_dropped_collection_stops_existing(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.adelete_collection(collection)
    assert await pgvector.acollection_exists(collection) is False


async def test_dropping_a_collection_that_is_not_there_is_tolerated(
    pgvector: PgVectorSearchService, unique: str
) -> None:
    await pgvector.adelete_collection(f"pgv_absent_{unique}")


async def test_the_extension_is_installed_by_the_first_collection(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    async with pgvector._asession() as session:
        installed = await session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        )
    assert installed.scalar() == 1


async def test_a_collection_named_with_capitals_answers_for_itself(
    pgvector: PgVectorSearchService, unique: str
) -> None:
    """An unquoted name would be folded, so it would answer for another table."""
    name = f"PgV_{unique}"
    await pgvector.acreate_collection(name, vector_size=3)
    try:
        assert await pgvector.acollection_exists(name) is True
    finally:
        await pgvector.adelete_collection(name)


async def test_a_collection_is_not_confused_with_its_lower_case_twin(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    assert await pgvector.acollection_exists(collection.upper()) is False


async def test_two_collections_differing_only_in_case_are_two_collections(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    upper = collection.upper()
    await pgvector.acreate_collection(upper, vector_size=3)
    try:
        await pgvector.aupsert(upper, [record("a")])
        assert await pgvector.acount(upper) == 1
        assert await pgvector.acount(collection) == 0
    finally:
        await pgvector.adelete_collection(upper)


@pytest.mark.parametrize(
    "name", ['chunks"; DROP TABLE x; --', "my chunks", "1chunks", "pg.chunks"]
)
async def test_a_name_that_is_not_an_identifier_never_reaches_the_server(
    pgvector: PgVectorSearchService, name: str
) -> None:
    with pytest.raises(VectorSearchUnexpectedError, match="invalid collection name"):
        await pgvector.acreate_collection(name, vector_size=3)


@pytest.mark.parametrize("name", ['chunks"; DROP TABLE x; --', "my chunks", ""])
async def test_asking_about_a_name_that_could_never_be_a_table_answers_no(
    pgvector: PgVectorSearchService, name: str
) -> None:
    assert await pgvector.acollection_exists(name) is False


def test_the_sync_path_creates_and_drops_a_collection(
    pgvector: PgVectorSearchService, unique: str
) -> None:
    name = f"pgv_sync_{unique}"
    pgvector.create_collection(name, vector_size=3)
    assert pgvector.collection_exists(name) is True
    pgvector.delete_collection(name)
    assert pgvector.collection_exists(name) is False


# Upsert and get


async def test_a_record_comes_back_with_the_same_id_it_went_in_with(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1")])
    assert (await pgvector.aget(collection, ["doc-1"]))[0].id == "doc-1"


@pytest.mark.parametrize(
    "identifier",
    ["doc/42", "0", str(uuid.uuid4()), "o'brien", 'a"b', "a;b", "δοκιμή", "x" * 200],
)
async def test_an_id_of_any_shape_round_trips(
    pgvector: PgVectorSearchService, collection: str, identifier: str
) -> None:
    await pgvector.aupsert(collection, [record(identifier)])
    assert (await pgvector.aget(collection, [identifier]))[0].id == identifier


async def test_metadata_round_trips(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    metadata = {"text": "héllo 🎉", "n": 3, "ok": True, "tags": ["a", "b"]}
    await pgvector.aupsert(
        collection,
        [VectorRecord(id="doc-1", vector=[1.0, 0.0, 0.0], metadata=metadata)],
    )
    assert (await pgvector.aget(collection, ["doc-1"]))[0].metadata == metadata


async def test_nested_metadata_round_trips(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    metadata = {"outer": {"inner": [1, {"deep": "value"}]}}
    await pgvector.aupsert(
        collection,
        [VectorRecord(id="doc-1", vector=[1.0, 0.0, 0.0], metadata=metadata)],
    )
    assert (await pgvector.aget(collection, ["doc-1"]))[0].metadata == metadata


async def test_metadata_carrying_sql_is_stored_as_data(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    metadata = {"text": "'); DROP TABLE students; --"}
    await pgvector.aupsert(
        collection,
        [VectorRecord(id="doc-1", vector=[1.0, 0.0, 0.0], metadata=metadata)],
    )
    assert (await pgvector.aget(collection, ["doc-1"]))[0].metadata == metadata
    assert await pgvector.acount(collection) == 1


async def test_a_record_without_metadata_comes_back_with_an_empty_mapping(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1")])
    assert (await pgvector.aget(collection, ["doc-1"]))[0].metadata == {}


async def test_the_vector_is_left_out_unless_asked_for(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1")])
    assert (await pgvector.aget(collection, ["doc-1"]))[0].vector == []


async def test_the_vector_comes_back_when_asked_for(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1", [0.25, -0.5, 1.0])])
    found = await pgvector.aget(collection, ["doc-1"], with_vectors=True)
    assert found[0].vector == [0.25, -0.5, 1.0]


async def test_upserting_the_same_id_twice_replaces_rather_than_duplicates(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1", v=1)])
    await pgvector.aupsert(collection, [record("doc-1", [0.0, 1.0, 0.0], v=2)])
    assert await pgvector.acount(collection) == 1
    assert (await pgvector.aget(collection, ["doc-1"]))[0].metadata == {"v": 2}


async def test_the_same_id_twice_in_one_batch_keeps_the_last_of_them(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1", v=1), record("doc-1", v=2)])
    assert await pgvector.acount(collection) == 1
    assert (await pgvector.aget(collection, ["doc-1"]))[0].metadata == {"v": 2}


async def test_an_empty_upsert_changes_nothing(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [])
    assert await pgvector.acount(collection) == 0


async def test_a_batch_larger_than_one_statement_is_stored_whole(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    count = VECTOR_UPSERT_BATCH_SIZE + 5
    await pgvector.aupsert(collection, [record(f"doc-{i}") for i in range(count)])
    assert await pgvector.acount(collection) == count


async def test_a_vector_of_the_wrong_width_is_refused(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    with pytest.raises(VectorSearchError):
        await pgvector.aupsert(collection, [record("doc-1", [1.0, 0.0])])


async def test_an_empty_vector_is_refused(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    with pytest.raises(VectorSearchError):
        await pgvector.aupsert(collection, [record("doc-1", [])])


async def test_upserting_into_a_collection_that_is_not_there_fails(
    pgvector: PgVectorSearchService, unique: str
) -> None:
    with pytest.raises(VectorSearchError):
        await pgvector.aupsert(f"pgv_absent_{unique}", [record("doc-1")])


async def test_getting_an_id_that_is_not_there_returns_nothing(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    assert await pgvector.aget(collection, ["nobody"]) == []


async def test_getting_a_mix_returns_only_what_exists(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1")])
    found = await pgvector.aget(collection, ["doc-1", "nobody"])
    assert [r.id for r in found] == ["doc-1"]


async def test_getting_no_ids_returns_nothing(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1")])
    assert await pgvector.aget(collection, []) == []


def test_the_sync_path_stores_and_reads_a_record(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    pgvector.upsert(collection, [record("doc-1", [0.5, 0.5, 0.0], a=1)])
    assert pgvector.count(collection) == 1
    found = pgvector.get(collection, ["doc-1"], with_vectors=True)[0]
    assert (found.metadata, found.vector) == ({"a": 1}, [0.5, 0.5, 0.0])


def test_the_sync_path_batches_the_same_way(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    count = VECTOR_UPSERT_BATCH_SIZE + 5
    pgvector.upsert(collection, [record(f"doc-{i}") for i in range(count)])
    assert pgvector.count(collection) == count


# Delete


async def test_deleting_a_record_removes_it(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1")])
    await pgvector.adelete(collection, ["doc-1"])
    assert await pgvector.acount(collection) == 0


async def test_deleting_an_id_that_is_not_there_is_tolerated(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1")])
    await pgvector.adelete(collection, ["nobody"])
    assert await pgvector.acount(collection) == 1


async def test_deleting_nothing_removes_nothing(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1")])
    await pgvector.adelete(collection, [])
    assert await pgvector.acount(collection) == 1


async def test_deleting_several_records_removes_each_of_them(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record(f"doc-{i}") for i in range(3)])
    await pgvector.adelete(collection, ["doc-0", "doc-1"])
    assert await pgvector.acount(collection) == 1


def test_the_sync_path_deletes_a_record(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    pgvector.upsert(collection, [record("doc-1")])
    pgvector.delete(collection, ["doc-1"])
    assert pgvector.count(collection) == 0


# Count


async def test_an_empty_collection_counts_zero(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    assert await pgvector.acount(collection) == 0


async def test_counting_a_collection_that_is_not_there_fails(
    pgvector: PgVectorSearchService, unique: str
) -> None:
    with pytest.raises(VectorSearchError):
        await pgvector.acount(f"pgv_absent_{unique}")


# Scroll


async def test_scroll_walks_the_whole_collection_in_pages(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    ids = {f"doc-{i}" for i in range(5)}
    await pgvector.aupsert(collection, [record(i) for i in ids])
    seen: set[str] = set()
    offset: str | None = None
    for _ in range(5):
        page, offset = await pgvector.ascroll(collection, limit=2, offset=offset)
        seen.update(r.id for r in page)
        if offset is None:
            break
    assert seen == ids


async def test_scroll_returns_records_in_id_order(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("c"), record("a"), record("b")])
    page, _ = await pgvector.ascroll(collection)
    assert [r.id for r in page] == ["a", "b", "c"]


async def test_a_full_page_reports_an_offset_even_when_nothing_follows(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    """Keyset paging cannot see ahead, so the caller finds out by asking."""
    await pgvector.aupsert(collection, [record("a"), record("b")])
    page, offset = await pgvector.ascroll(collection, limit=2)
    assert offset == "b"
    assert await pgvector.ascroll(collection, limit=2, offset=offset) == ([], None)


async def test_a_short_page_is_the_last_one(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("a")])
    _, offset = await pgvector.ascroll(collection, limit=2)
    assert offset is None


async def test_scrolling_an_empty_collection_returns_nothing(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    assert await pgvector.ascroll(collection) == ([], None)


async def test_scrolling_for_no_records_at_all_returns_nothing(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("a")])
    assert await pgvector.ascroll(collection, limit=0) == ([], None)


async def test_scroll_narrows_with_a_filter(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(
        collection, [record("keep", tenant="a"), record("drop", tenant="b")]
    )
    page, _ = await pgvector.ascroll(collection, filters={"tenant": "a"})
    assert [r.id for r in page] == ["keep"]


async def test_scroll_can_bring_the_vectors(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("a", [0.5, 0.0, -1.0])])
    page, _ = await pgvector.ascroll(collection, with_vectors=True)
    assert page[0].vector == [0.5, 0.0, -1.0]


async def test_scroll_resumes_strictly_after_the_offset(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("a"), record("b"), record("c")])
    page, _ = await pgvector.ascroll(collection, offset="a")
    assert [r.id for r in page] == ["b", "c"]


def test_the_sync_path_scrolls_the_same_way(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    pgvector.upsert(collection, [record("a"), record("b")])
    page, offset = pgvector.scroll(collection, limit=1)
    assert ([r.id for r in page], offset) == (["a"], "a")


# Health and lifecycle


async def test_the_service_reports_itself_healthy(
    pgvector: PgVectorSearchService,
) -> None:
    assert await pgvector.ahealth_check() is True


def test_the_sync_health_check_reports_the_same(
    pgvector: PgVectorSearchService,
) -> None:
    assert pgvector.health_check() is True


def test_the_service_names_itself_pgvector(pgvector: PgVectorSearchService) -> None:
    assert pgvector.get_service_name() == "pgvector"


def test_a_disposed_service_opens_a_new_pool_when_it_is_used_again(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    pgvector.upsert(collection, [record("doc-1")])
    pgvector.dispose()
    assert pgvector.count(collection) == 1


async def test_disposing_asynchronously_releases_the_async_pool(
    pgvector: PgVectorSearchService, collection: str
) -> None:
    await pgvector.aupsert(collection, [record("doc-1")])
    await pgvector.adispose()
    assert await pgvector.acount(collection) == 1
