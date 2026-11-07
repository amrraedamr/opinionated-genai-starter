"""What a real Qdrant accepts as a point id, and what a point carries.

Over gRPC a point id must be a UUID or an unsigned integer. `VectorRecord.id`
is typed `str`, so a UUID string is the only id a caller of this service can
actually use: a numeric string is parsed as a UUID and refused, and so is any
other word. That is pinned here rather than assumed, together with the
`metadata` to `payload` rename the adapter owns.
"""

import uuid

import pytest

from base.models.view_models.exceptions.vector_search import (
    VectorSearchError,
    VectorSearchUnexpectedError,
)
from base.models.view_models.vector_search import VectorRecord
from base.services.vector_search.qdrant import QdrantVectorSearchService

pytestmark = pytest.mark.integration


async def test_a_uuid_is_accepted_as_a_point_id(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    await qdrant.aupsert(
        collection, [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0])]
    )
    assert await qdrant.acount(collection) == 1


@pytest.mark.parametrize("bad", ["v1", "12345", "", "not-a-uuid", "0"])
async def test_an_id_that_is_not_a_uuid_is_refused(
    qdrant: QdrantVectorSearchService, collection: str, bad: str
) -> None:
    """Pins the limit: with a string-only id field, only UUIDs are reachable."""
    with pytest.raises(VectorSearchError) as caught:
        await qdrant.aupsert(collection, [VectorRecord(id=bad, vector=[1.0, 0.0, 0.0])])
    assert "UUID" in str(caught.value)


async def test_an_invalid_id_surfaces_as_an_unexpected_error(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    """gRPC raises its own error type, which the Qdrant handler does not know."""
    with pytest.raises(VectorSearchUnexpectedError):
        await qdrant.aupsert(
            collection, [VectorRecord(id="v1", vector=[1.0, 0.0, 0.0])]
        )


async def test_an_invalid_id_leaves_the_collection_untouched(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    with pytest.raises(VectorSearchError):
        await qdrant.aupsert(
            collection, [VectorRecord(id="v1", vector=[1.0, 0.0, 0.0])]
        )
    assert await qdrant.acount(collection) == 0


async def test_one_bad_id_rejects_the_whole_batch(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    with pytest.raises(VectorSearchError):
        await qdrant.aupsert(
            collection,
            [
                VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0]),
                VectorRecord(id="v1", vector=[0.0, 1.0, 0.0]),
            ],
        )
    assert await qdrant.acount(collection) == 0


# Records and metadata


async def test_metadata_round_trips_as_qdrants_payload(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    metadata = {"text": "héllo 🎉", "n": 3, "ok": True, "tags": ["a", "b"]}
    await qdrant.aupsert(
        collection,
        [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0], metadata=metadata)],
    )
    record = (await qdrant.aget(collection, [point_id]))[0]
    assert record.metadata == metadata


async def test_a_record_comes_back_with_the_same_id_it_went_in_with(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    await qdrant.aupsert(
        collection, [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0])]
    )
    assert (await qdrant.aget(collection, [point_id]))[0].id == point_id


async def test_a_record_without_metadata_comes_back_with_an_empty_mapping(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    await qdrant.aupsert(
        collection, [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0])]
    )
    assert (await qdrant.aget(collection, [point_id]))[0].metadata == {}


async def test_the_vector_is_left_out_unless_asked_for(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    await qdrant.aupsert(
        collection, [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0])]
    )
    assert (await qdrant.aget(collection, [point_id]))[0].vector == []


async def test_the_vector_comes_back_when_asked_for(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    await qdrant.aupsert(
        collection, [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0])]
    )
    record = (await qdrant.aget(collection, [point_id], with_vectors=True))[0]
    assert record.vector == [1.0, 0.0, 0.0]


async def test_getting_an_id_that_is_not_there_returns_nothing(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    assert await qdrant.aget(collection, [str(uuid.uuid4())]) == []


async def test_getting_a_mix_returns_only_what_exists(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    await qdrant.aupsert(
        collection, [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0])]
    )
    found = await qdrant.aget(collection, [point_id, str(uuid.uuid4())])
    assert [r.id for r in found] == [point_id]


async def test_nested_metadata_round_trips(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    metadata = {"outer": {"inner": [1, {"deep": "value"}]}}
    await qdrant.aupsert(
        collection,
        [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0], metadata=metadata)],
    )
    assert (await qdrant.aget(collection, [point_id]))[0].metadata == metadata


# Upsert semantics


async def test_upserting_the_same_id_twice_replaces_rather_than_duplicates(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    await qdrant.aupsert(
        collection,
        [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0], metadata={"v": 1})],
    )
    await qdrant.aupsert(
        collection,
        [VectorRecord(id=point_id, vector=[0.0, 1.0, 0.0], metadata={"v": 2})],
    )
    assert await qdrant.acount(collection) == 1
    assert (await qdrant.aget(collection, [point_id]))[0].metadata == {"v": 2}


async def test_an_empty_upsert_changes_nothing(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await qdrant.aupsert(collection, [])
    assert await qdrant.acount(collection) == 0


async def test_a_batch_larger_than_one_request_is_stored_whole(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    from base.models.definitions.vector_search import VECTOR_UPSERT_BATCH_SIZE

    records = [
        VectorRecord(id=str(uuid.uuid4()), vector=[1.0, 0.0, 0.0])
        for _ in range(VECTOR_UPSERT_BATCH_SIZE + 5)
    ]
    await qdrant.aupsert(collection, records)
    assert await qdrant.acount(collection) == VECTOR_UPSERT_BATCH_SIZE + 5


# Delete


async def test_deleting_a_point_removes_it(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    await qdrant.aupsert(
        collection, [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0])]
    )
    await qdrant.adelete(collection, [point_id])
    assert await qdrant.acount(collection) == 0


async def test_deleting_an_id_that_is_not_there_is_tolerated(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    await qdrant.aupsert(
        collection, [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0])]
    )
    await qdrant.adelete(collection, [str(uuid.uuid4())])
    assert await qdrant.acount(collection) == 1


async def test_deleting_several_points_removes_each_of_them(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    ids = [str(uuid.uuid4()) for _ in range(3)]
    await qdrant.aupsert(
        collection, [VectorRecord(id=i, vector=[1.0, 0.0, 0.0]) for i in ids]
    )
    await qdrant.adelete(collection, ids[:2])
    assert await qdrant.acount(collection) == 1


# Sync counterparts


def test_the_sync_path_stores_and_reads_a_point(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    qdrant.upsert(
        collection,
        [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0], metadata={"a": 1})],
    )
    assert qdrant.count(collection) == 1
    assert qdrant.get(collection, [point_id])[0].metadata == {"a": 1}


def test_the_sync_path_deletes_a_point(
    qdrant: QdrantVectorSearchService, collection: str, point_id: str
) -> None:
    qdrant.upsert(collection, [VectorRecord(id=point_id, vector=[1.0, 0.0, 0.0])])
    qdrant.delete(collection, [point_id])
    assert qdrant.count(collection) == 0


def test_the_sync_path_refuses_the_same_bad_ids(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    with pytest.raises(VectorSearchError):
        qdrant.upsert(collection, [VectorRecord(id="v1", vector=[1.0, 0.0, 0.0])])


# Scroll


async def test_scroll_walks_the_whole_collection_in_pages(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    ids = {str(uuid.uuid4()) for _ in range(5)}
    await qdrant.aupsert(
        collection, [VectorRecord(id=i, vector=[1.0, 0.0, 0.0]) for i in ids]
    )
    seen: set[str] = set()
    offset: str | None = None
    for _ in range(5):
        records, offset = await qdrant.ascroll(collection, limit=2, offset=offset)
        seen.update(r.id for r in records)
        if offset is None:
            break
    assert seen == ids


async def test_scroll_narrows_with_a_filter(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    keep, drop = str(uuid.uuid4()), str(uuid.uuid4())
    await qdrant.aupsert(
        collection,
        [
            VectorRecord(id=keep, vector=[1.0, 0.0, 0.0], metadata={"tenant": "a"}),
            VectorRecord(id=drop, vector=[0.0, 1.0, 0.0], metadata={"tenant": "b"}),
        ],
    )
    records, _ = await qdrant.ascroll(collection, filters={"tenant": "a"})
    assert [r.id for r in records] == [keep]
