"""Similarity search against a real Qdrant: order, limits, filters and failures.

Score ordering and metadata filtering are the two things a caller relies on
and neither can be proved without a server, so they are checked against real
cosine scores here. The failure cases — a query of the wrong width, a
collection that is not there — are pinned as the errors they actually raise.
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

NORTH = [1.0, 0.0, 0.0]
EAST = [0.0, 1.0, 0.0]
UP = [0.0, 0.0, 1.0]


async def seed(qdrant: QdrantVectorSearchService, collection: str) -> dict[str, str]:
    """Three orthogonal vectors, each tagged with its own name."""
    ids = {name: str(uuid.uuid4()) for name in ("north", "east", "up")}
    await qdrant.aupsert(
        collection,
        [
            VectorRecord(
                id=ids["north"], vector=NORTH, metadata={"name": "north", "axis": "x"}
            ),
            VectorRecord(
                id=ids["east"], vector=EAST, metadata={"name": "east", "axis": "y"}
            ),
            VectorRecord(id=ids["up"], vector=UP, metadata={"name": "up", "axis": "z"}),
        ],
    )
    return ids


async def test_the_nearest_vector_comes_first(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    ids = await seed(qdrant, collection)
    results = await qdrant.asearch(collection, NORTH)
    assert results[0].id == ids["north"]


async def test_an_exact_match_scores_one_under_cosine(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    assert await qdrant.asearch(collection, NORTH) != []
    assert (await qdrant.asearch(collection, NORTH))[0].score == pytest.approx(1.0)


async def test_results_arrive_in_descending_score_order(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    scores = [r.score for r in await qdrant.asearch(collection, [0.9, 0.4, 0.1])]
    assert scores == sorted(scores, reverse=True)


async def test_the_limit_caps_the_number_of_results(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    assert len(await qdrant.asearch(collection, NORTH, limit=2)) == 2


async def test_a_limit_larger_than_the_collection_returns_everything(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    assert len(await qdrant.asearch(collection, NORTH, limit=50)) == 3


async def test_a_limit_of_one_returns_only_the_nearest(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    ids = await seed(qdrant, collection)
    results = await qdrant.asearch(collection, EAST, limit=1)
    assert [r.id for r in results] == [ids["east"]]


async def test_a_result_carries_its_metadata(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    assert (await qdrant.asearch(collection, NORTH))[0].metadata["name"] == "north"


async def test_a_result_carries_its_vector(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    assert (await qdrant.asearch(collection, NORTH))[0].vector == NORTH


# Filters


async def test_a_metadata_filter_narrows_the_results(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    ids = await seed(qdrant, collection)
    results = await qdrant.asearch(collection, NORTH, filters={"name": "east"})
    assert [r.id for r in results] == [ids["east"]]


async def test_a_list_filter_matches_any_of_its_values(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    results = await qdrant.asearch(collection, NORTH, filters={"axis": ["x", "y"]})
    assert {r.metadata["axis"] for r in results} == {"x", "y"}


async def test_two_filters_must_both_hold(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    results = await qdrant.asearch(
        collection, NORTH, filters={"name": "north", "axis": "y"}
    )
    assert results == []


async def test_a_filter_matching_nothing_returns_nothing(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    assert await qdrant.asearch(collection, NORTH, filters={"name": "absent"}) == []


async def test_a_filter_on_a_field_nothing_has_returns_nothing(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    assert await qdrant.asearch(collection, NORTH, filters={"absent": "x"}) == []


# Thresholds


async def test_a_score_threshold_drops_the_distant_vectors(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    ids = await seed(qdrant, collection)
    results = await qdrant.asearch(collection, NORTH, score_threshold=0.9)
    assert [r.id for r in results] == [ids["north"]]


async def test_a_threshold_above_every_score_returns_nothing(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    assert await qdrant.asearch(collection, NORTH, score_threshold=1.1) == []


# Failures


async def test_a_query_vector_of_the_wrong_width_is_refused(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    with pytest.raises(VectorSearchError) as caught:
        await qdrant.asearch(collection, [1.0, 0.0])
    assert "dimension" in str(caught.value).lower()


async def test_a_query_vector_that_is_too_long_is_refused(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    with pytest.raises(VectorSearchError):
        await qdrant.asearch(collection, [1.0, 0.0, 0.0, 0.0])


async def test_an_empty_query_vector_is_refused(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    with pytest.raises(VectorSearchError):
        await qdrant.asearch(collection, [])


async def test_searching_a_collection_that_does_not_exist_names_it(
    qdrant: QdrantVectorSearchService, unique: str
) -> None:
    with pytest.raises(VectorSearchUnexpectedError) as caught:
        await qdrant.asearch(f"absent_{unique}", NORTH)
    assert f"absent_{unique}" in str(caught.value)


async def test_searching_a_deleted_collection_fails(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    await seed(qdrant, collection)
    await qdrant.adelete_collection(collection)
    with pytest.raises(VectorSearchError):
        await qdrant.asearch(collection, NORTH)


# Sync counterpart


def test_the_sync_search_orders_results_the_same_way(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    identifier = str(uuid.uuid4())
    qdrant.upsert(
        collection,
        [
            VectorRecord(id=identifier, vector=NORTH, metadata={"name": "north"}),
            VectorRecord(id=str(uuid.uuid4()), vector=EAST, metadata={"name": "east"}),
        ],
    )
    results = qdrant.search(collection, NORTH, limit=1)
    assert [r.id for r in results] == [identifier]


def test_the_sync_search_applies_a_metadata_filter(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    wanted = str(uuid.uuid4())
    qdrant.upsert(
        collection,
        [
            VectorRecord(
                id=str(uuid.uuid4()), vector=NORTH, metadata={"name": "north"}
            ),
            VectorRecord(id=wanted, vector=EAST, metadata={"name": "east"}),
        ],
    )
    assert [
        r.id for r in qdrant.search(collection, NORTH, filters={"name": "east"})
    ] == [wanted]


def test_the_sync_search_refuses_the_wrong_width(
    qdrant: QdrantVectorSearchService, collection: str
) -> None:
    with pytest.raises(VectorSearchError):
        qdrant.search(collection, [1.0])
