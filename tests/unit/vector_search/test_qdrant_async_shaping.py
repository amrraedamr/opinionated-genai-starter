"""The async half of the Qdrant adapter, which uses a client of its own.

Each async method goes through `async_client` rather than `sync_client`, so a
sync-only test leaves half the surface unproved. The translations are shared,
so what is checked here is that every async method reaches the async client
with the same request its sync twin builds.
"""

from typing import Any

import pytest

from base.models.definitions.vector_search import VECTOR_UPSERT_BATCH_SIZE
from base.models.view_models.vector_search import VectorRecord
from base.services.configuration.mixins import QdrantSettingsMixin
from base.services.vector_search.qdrant import QdrantVectorSearchService
from tests.conftest import MakeSettings
from tests.unit.vector_search.test_qdrant_shaping import (
    CountResponse,
    Point,
    QueryResponse,
)


class AsyncRecordingClient:
    """Stands in for `AsyncQdrantClient`, keeping every request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.points: list[Point] = []
        self.scored: list[Point] = []
        self.next_offset: Any = None
        self.exists: bool = True

    def call(self, method: str) -> dict[str, Any]:
        for name, kwargs in reversed(self.calls):
            if name == method:
                return kwargs
        raise AssertionError(f"{method} was never called")

    def every(self, method: str) -> list[dict[str, Any]]:
        return [kwargs for name, kwargs in self.calls if name == method]

    async def create_collection(self, **kwargs: Any) -> None:
        self.calls.append(("create_collection", kwargs))

    async def delete_collection(self, **kwargs: Any) -> None:
        self.calls.append(("delete_collection", kwargs))

    async def collection_exists(self, **kwargs: Any) -> bool:
        self.calls.append(("collection_exists", kwargs))
        return self.exists

    async def upsert(self, **kwargs: Any) -> None:
        self.calls.append(("upsert", kwargs))

    async def query_points(self, **kwargs: Any) -> QueryResponse:
        self.calls.append(("query_points", kwargs))
        return QueryResponse(self.scored)

    async def retrieve(self, **kwargs: Any) -> list[Point]:
        self.calls.append(("retrieve", kwargs))
        return self.points

    async def delete(self, **kwargs: Any) -> None:
        self.calls.append(("delete", kwargs))

    async def count(self, **kwargs: Any) -> CountResponse:
        self.calls.append(("count", kwargs))
        return CountResponse(len(self.points))

    async def scroll(self, **kwargs: Any) -> tuple[list[Point], Any]:
        self.calls.append(("scroll", kwargs))
        return self.points, self.next_offset

    async def get_collections(self) -> list[Any]:
        self.calls.append(("get_collections", {}))
        return []


@pytest.fixture
def service(make_settings: MakeSettings) -> QdrantVectorSearchService:
    make_settings(QdrantSettingsMixin, qdrant_url="http://localhost:6333")
    return QdrantVectorSearchService()


@pytest.fixture
def client(service: QdrantVectorSearchService) -> AsyncRecordingClient:
    recorder = AsyncRecordingClient()
    service._client._async_client = recorder  # type: ignore[assignment]
    return recorder


async def test_creating_a_collection_sends_the_size_and_the_metric(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    await service.acreate_collection("chunks", vector_size=1536, distance="dot")
    config = client.call("create_collection")["vectors_config"]
    assert config.size == 1536
    assert config.distance == "Dot"


async def test_deleting_a_collection_names_it(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    await service.adelete_collection("chunks")
    assert client.call("delete_collection") == {"collection_name": "chunks"}


async def test_collection_exists_answers_with_what_qdrant_said(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    client.exists = False
    assert await service.acollection_exists("chunks") is False


async def test_a_record_becomes_a_point_with_its_metadata_as_the_payload(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    await service.aupsert(
        "chunks", [VectorRecord(id="a", vector=[0.1], metadata={"text": "hi"})]
    )
    point = client.call("upsert")["points"][0]
    assert (point.id, point.vector, point.payload) == (
        QdrantVectorSearchService._point_id("a"),
        [0.1],
        {"text": "hi", "_source_id": "a"},
    )


async def test_an_empty_upsert_sends_nothing(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    await service.aupsert("chunks", [])
    assert client.every("upsert") == []


async def test_a_batch_over_the_limit_is_split(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    records = [
        VectorRecord(id=str(i), vector=[0.1])
        for i in range(VECTOR_UPSERT_BATCH_SIZE + 1)
    ]
    await service.aupsert("chunks", records)
    assert [len(call["points"]) for call in client.every("upsert")] == [
        VECTOR_UPSERT_BATCH_SIZE,
        1,
    ]


async def test_a_search_asks_for_the_payload_and_the_vector(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    await service.asearch("chunks", [0.1], limit=3, score_threshold=0.5)
    sent = client.call("query_points")
    assert sent["with_payload"] is True
    assert sent["with_vectors"] is True
    assert sent["limit"] == 3
    assert sent["score_threshold"] == 0.5


async def test_a_search_translates_its_filters(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    await service.asearch("chunks", [0.1], filters={"tag": ["a", "b"]})
    built = client.call("query_points")["query_filter"]
    assert built.must[0].match.any == ["a", "b"]


async def test_a_scored_point_becomes_a_result(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    client.scored = [Point("a", [0.1], {"text": "hi"}, score=0.9)]
    result = (await service.asearch("chunks", [0.1]))[0]
    assert (result.id, result.score, result.vector) == ("a", 0.9, [0.1])
    assert result.metadata == {"text": "hi"}


async def test_get_asks_for_the_ids_it_was_given(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    client.points = [Point("a", [0.1], {"text": "hi", "_source_id": "a"})]
    record = (await service.aget("chunks", ["a"], with_vectors=True))[0]
    assert client.call("retrieve")["ids"] == [QdrantVectorSearchService._point_id("a")]
    assert (record.id, record.vector, record.metadata) == ("a", [0.1], {"text": "hi"})


async def test_delete_sends_the_ids_as_a_point_selector(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    await service.adelete("chunks", ["a", "b"])
    assert client.call("delete")["points_selector"].points == [
        QdrantVectorSearchService._point_id("a"),
        QdrantVectorSearchService._point_id("b"),
    ]


async def test_count_reads_the_number_off_the_response(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    client.points = [Point("a"), Point("b"), Point("c")]
    assert await service.acount("chunks") == 3


async def test_scroll_passes_the_offset_through_and_reports_the_next_one(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    client.next_offset = 9
    _, offset = await service.ascroll("chunks", limit=4, offset="3")
    sent = client.call("scroll")
    assert (sent["limit"], sent["offset"]) == (4, "3")
    assert offset == "9"


async def test_scroll_reports_no_next_offset_at_the_end(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    client.next_offset = None
    assert (await service.ascroll("chunks"))[1] is None


async def test_the_async_health_check_reaches_the_async_client(
    service: QdrantVectorSearchService, client: AsyncRecordingClient
) -> None:
    assert await service.ahealth_check() is True
    assert client.every("get_collections") == [{}]
