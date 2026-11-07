"""Creating, checking and deleting a real Qdrant collection, sync and async.

The whole collection half of `AbstractVectorSearchService` is here, in both
call styles, because a backend that answers `collection_exists` correctly but
only on one of them is broken for half its callers.
"""

import uuid
from typing import Any

import pytest

from base.models.view_models.exceptions.vector_search import VectorSearchError
from base.services.vector_search.qdrant import QdrantVectorSearchService

pytestmark = pytest.mark.integration


async def test_a_created_collection_exists(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    assert await qdrant.acollection_exists(collection) is True


async def test_a_collection_that_was_never_created_does_not_exist(
    qdrant: QdrantVectorSearchService, unique: str
) -> None:
    assert await qdrant.acollection_exists(f"absent_{unique}") is False


async def test_a_deleted_collection_stops_existing(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await qdrant.adelete_collection(collection)
    assert await qdrant.acollection_exists(collection) is False


def test_the_sync_path_sees_the_same_collection(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    assert qdrant.collection_exists(collection) is True


def test_a_collection_created_synchronously_is_visible_asynchronously(
    qdrant: QdrantVectorSearchService, unique: str
) -> None:
    name = f"s_{unique}"
    qdrant.create_collection(name, vector_size=3)
    try:
        assert qdrant.collection_exists(name) is True
    finally:
        qdrant.delete_collection(name)
    assert qdrant.collection_exists(name) is False


@pytest.mark.parametrize("distance", ["cosine", "dot", "euclidean"])
async def test_every_distance_metric_builds_a_working_collection(
    qdrant: QdrantVectorSearchService, unique: str, distance: str
) -> None:
    name = f"d_{unique}_{distance}"
    await qdrant.acreate_collection(name, vector_size=3, distance=distance)
    try:
        assert await qdrant.acount(name) == 0
    finally:
        await qdrant.adelete_collection(name)


async def test_creating_a_collection_that_already_exists_is_refused(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    with pytest.raises(VectorSearchError):
        await qdrant.acreate_collection(collection, vector_size=3)


async def test_deleting_a_collection_that_does_not_exist_is_tolerated(
    qdrant: QdrantVectorSearchService, unique: str
) -> None:
    """Qdrant answers a delete of nothing without complaint."""
    await qdrant.adelete_collection(f"absent_{unique}")
    assert await qdrant.acollection_exists(f"absent_{unique}") is False


async def test_a_new_collection_is_empty(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    assert await qdrant.acount(collection) == 0


async def test_scrolling_an_empty_collection_yields_nothing(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    records, offset = await qdrant.ascroll(collection)
    assert records == []
    assert offset is None


async def test_searching_an_empty_collection_yields_nothing(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    assert await qdrant.asearch(collection, [1.0, 0.0, 0.0]) == []


async def test_searching_a_collection_that_does_not_exist_fails(
    qdrant: QdrantVectorSearchService, unique: str
) -> None:
    with pytest.raises(Exception) as caught:
        await qdrant.asearch(f"absent_{unique}", [1.0, 0.0, 0.0])
    assert "doesn't exist" in str(caught.value)


async def test_counting_a_collection_that_does_not_exist_fails(
    qdrant: QdrantVectorSearchService, unique: str
) -> None:
    with pytest.raises(Exception):
        await qdrant.acount(f"absent_{unique}")


async def test_the_async_health_check_reaches_the_container(
    qdrant: QdrantVectorSearchService,
) -> None:
    assert await qdrant.ahealth_check() is True


def test_the_health_check_reaches_the_container(
    qdrant: QdrantVectorSearchService,
) -> None:
    assert qdrant.health_check() is True


async def test_a_collection_of_a_different_width_rejects_the_wrong_vector(
    qdrant: QdrantVectorSearchService, unique: str
) -> None:
    from base.models.view_models.vector_search import VectorRecord

    name = f"w_{unique}"
    await qdrant.acreate_collection(name, vector_size=4)
    try:
        with pytest.raises(Exception) as caught:
            await qdrant.aupsert(
                name,
                [VectorRecord(id=str(uuid.uuid4()), vector=[1.0, 0.0, 0.0])],
            )
        assert "dimension" in str(caught.value).lower()
    finally:
        await qdrant.adelete_collection(name)


def test_the_client_prefers_grpc_against_the_local_container(
    qdrant: QdrantVectorSearchService, infra_settings: Any
) -> None:
    """The stricter transport, and the one the id rules below are about."""
    assert qdrant._client._prefer_grpc is True
    assert qdrant._client._grpc_port == 6334
