"""What the Qdrant adapter builds, before any of it reaches a server.

The filter translation, the distance mapping, the batching and the shape of a
point are all decisions made in this process, so they are pinned here against
a recorder rather than a container. `metadata` becomes Qdrant's `payload` on
the way down and comes back as `metadata` again, which is the one rename the
adapter owns.
"""

from typing import Any

import pytest
from qdrant_client.models import Distance, FieldCondition, MatchAny, MatchValue

from base.models.definitions.vector_search import VECTOR_UPSERT_BATCH_SIZE
from base.models.view_models.vector_search import VectorRecord
from base.services.configuration.mixins import QdrantSettingsMixin
from base.services.vector_search.qdrant import QdrantVectorSearchService
from tests.conftest import MakeSettings


class Point:
    """What Qdrant hands back for a retrieved or scored point."""

    def __init__(
        self,
        identifier: Any,
        vector: Any = None,
        payload: dict[str, Any] | None = None,
        score: float = 0.0,
    ) -> None:
        self.id = identifier
        self.vector = vector
        self.payload = payload
        self.score = score


class QueryResponse:
    def __init__(self, points: list[Point]) -> None:
        self.points = points


class CountResponse:
    def __init__(self, count: int) -> None:
        self.count = count


class RecordingClient:
    """Stands in for the Qdrant SDK client, keeping every request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.points: list[Point] = []
        self.scored: list[Point] = []
        self.next_offset: Any = None
        self.exists: bool = True

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))

    def call(self, method: str) -> dict[str, Any]:
        for name, kwargs in reversed(self.calls):
            if name == method:
                return kwargs
        raise AssertionError(f"{method} was never called")

    def every(self, method: str) -> list[dict[str, Any]]:
        return [kwargs for name, kwargs in self.calls if name == method]

    def create_collection(self, **kwargs: Any) -> None:
        self._record("create_collection", **kwargs)

    def delete_collection(self, **kwargs: Any) -> None:
        self._record("delete_collection", **kwargs)

    def collection_exists(self, **kwargs: Any) -> bool:
        self._record("collection_exists", **kwargs)
        return self.exists

    def upsert(self, **kwargs: Any) -> None:
        self._record("upsert", **kwargs)

    def query_points(self, **kwargs: Any) -> QueryResponse:
        self._record("query_points", **kwargs)
        return QueryResponse(self.scored)

    def retrieve(self, **kwargs: Any) -> list[Point]:
        self._record("retrieve", **kwargs)
        return self.points

    def delete(self, **kwargs: Any) -> None:
        self._record("delete", **kwargs)

    def count(self, **kwargs: Any) -> CountResponse:
        self._record("count", **kwargs)
        return CountResponse(len(self.points))

    def scroll(self, **kwargs: Any) -> tuple[list[Point], Any]:
        self._record("scroll", **kwargs)
        return self.points, self.next_offset

    def get_collections(self) -> list[Any]:
        self._record("get_collections")
        return []


@pytest.fixture
def service(make_settings: MakeSettings) -> QdrantVectorSearchService:
    make_settings(QdrantSettingsMixin, qdrant_url="http://localhost:6333")
    return QdrantVectorSearchService()


@pytest.fixture
def client(service: QdrantVectorSearchService) -> RecordingClient:
    recorder = RecordingClient()
    service._client._sync_client = recorder  # type: ignore[assignment]
    return recorder


# Filter translation


def test_no_filter_sends_no_filter(service: QdrantVectorSearchService) -> None:
    assert service._build_filter(None) is None
    assert service._build_filter({}) is None


def test_a_scalar_becomes_a_match_on_that_value(
    service: QdrantVectorSearchService,
) -> None:
    built = service._build_filter({"tenant": "acme"})
    assert built is not None
    condition = built.must[0]
    assert isinstance(condition, FieldCondition)
    assert condition.key == "tenant"
    assert condition.match == MatchValue(value="acme")


def test_a_list_becomes_a_match_on_any_of_its_values(
    service: QdrantVectorSearchService,
) -> None:
    built = service._build_filter({"tag": ["a", "b"]})
    assert built is not None
    assert built.must[0].match == MatchAny(any=["a", "b"])


def test_several_fields_are_all_required(
    service: QdrantVectorSearchService,
) -> None:
    built = service._build_filter({"tenant": "acme", "year": 2026})
    assert built is not None
    assert [c.key for c in built.must] == ["tenant", "year"]


def test_a_numeric_value_keeps_its_type(service: QdrantVectorSearchService) -> None:
    built = service._build_filter({"year": 2026})
    assert built is not None
    assert built.must[0].match == MatchValue(value=2026)


# Collection management


@pytest.mark.parametrize(
    ("named", "expected"),
    [
        ("cosine", Distance.COSINE),
        ("dot", Distance.DOT),
        ("euclidean", Distance.EUCLID),
    ],
)
def test_each_distance_name_maps_to_the_qdrant_metric(
    service: QdrantVectorSearchService,
    client: RecordingClient,
    named: str,
    expected: Distance,
) -> None:
    service.create_collection("chunks", vector_size=8, distance=named)
    assert client.call("create_collection")["vectors_config"].distance == expected


def test_an_unknown_distance_falls_back_to_cosine(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.create_collection("chunks", vector_size=8, distance="manhattan")
    assert (
        client.call("create_collection")["vectors_config"].distance == Distance.COSINE
    )


def test_the_vector_size_reaches_the_collection_configuration(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.create_collection("chunks", vector_size=1536)
    assert client.call("create_collection")["vectors_config"].size == 1536


def test_extra_options_are_passed_through_to_qdrant(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.create_collection("chunks", vector_size=8, on_disk_payload=True)
    assert client.call("create_collection")["on_disk_payload"] is True


def test_deleting_a_collection_names_it(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.delete_collection("chunks")
    assert client.call("delete_collection") == {"collection_name": "chunks"}


def test_collection_exists_answers_with_what_qdrant_said(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    client.exists = False
    assert service.collection_exists("chunks") is False


# Upsert


def test_a_record_becomes_a_point_with_its_metadata_as_the_payload(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.upsert(
        "chunks", [VectorRecord(id="a", vector=[0.1, 0.2], metadata={"text": "hi"})]
    )
    point = client.call("upsert")["points"][0]
    assert point.id == QdrantVectorSearchService._point_id("a")
    assert point.vector == [0.1, 0.2]
    assert point.payload == {"text": "hi", "_source_id": "a"}


def test_an_empty_upsert_sends_nothing(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.upsert("chunks", [])
    assert client.every("upsert") == []


def test_a_batch_at_the_limit_is_one_request(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    records = [
        VectorRecord(id=str(i), vector=[0.1]) for i in range(VECTOR_UPSERT_BATCH_SIZE)
    ]
    service.upsert("chunks", records)
    assert len(client.every("upsert")) == 1


def test_a_batch_over_the_limit_is_split(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    records = [
        VectorRecord(id=str(i), vector=[0.1])
        for i in range(VECTOR_UPSERT_BATCH_SIZE + 1)
    ]
    service.upsert("chunks", records)
    sent = client.every("upsert")
    assert [len(call["points"]) for call in sent] == [VECTOR_UPSERT_BATCH_SIZE, 1]


# Search


def test_a_search_asks_for_the_payload_and_the_vector(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.search("chunks", [0.1, 0.2])
    sent = client.call("query_points")
    assert sent["with_payload"] is True
    assert sent["with_vectors"] is True


def test_a_search_sends_the_query_vector_and_the_limit(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.search("chunks", [0.1, 0.2], limit=3)
    sent = client.call("query_points")
    assert sent["query"] == [0.1, 0.2]
    assert sent["limit"] == 3


def test_a_search_without_filters_sends_no_filter(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.search("chunks", [0.1])
    assert client.call("query_points")["query_filter"] is None


def test_a_search_translates_its_filters(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.search("chunks", [0.1], filters={"tenant": "acme"})
    built = client.call("query_points")["query_filter"]
    assert built.must[0].key == "tenant"


def test_a_score_threshold_is_pushed_down_to_qdrant(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.search("chunks", [0.1], score_threshold=0.75)
    assert client.call("query_points")["score_threshold"] == 0.75


def test_a_scored_point_becomes_a_result_with_its_metadata(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    client.scored = [Point("a", [0.1], {"text": "hi"}, score=0.9)]
    result = service.search("chunks", [0.1])[0]
    assert (result.id, result.score, result.vector) == ("a", 0.9, [0.1])
    assert result.metadata == {"text": "hi"}


def test_a_point_without_a_payload_comes_back_with_empty_metadata(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    client.scored = [Point("a", [0.1], None, score=0.5)]
    assert service.search("chunks", [0.1])[0].metadata == {}


def test_a_numeric_point_id_is_reported_as_a_string(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    """Qdrant allows unsigned integer ids; the shared record only has strings."""
    client.scored = [Point(42, [0.1], {}, score=0.5)]
    assert service.search("chunks", [0.1])[0].id == "42"


# Get, delete, count and scroll


def test_get_asks_for_the_ids_it_was_given(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.get("chunks", ["a", "b"])
    sent = client.call("retrieve")
    assert sent["ids"] == [
        QdrantVectorSearchService._point_id("a"),
        QdrantVectorSearchService._point_id("b"),
    ]
    assert sent["with_vectors"] is False


def test_get_can_ask_for_the_vectors_too(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.get("chunks", ["a"], with_vectors=True)
    assert client.call("retrieve")["with_vectors"] is True


def test_a_retrieved_point_becomes_a_record(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    client.points = [Point("a", [0.1], {"text": "hi"})]
    record = service.get("chunks", ["a"], with_vectors=True)[0]
    assert (record.id, record.vector, record.metadata) == ("a", [0.1], {"text": "hi"})


def test_delete_sends_the_ids_as_a_point_selector(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.delete("chunks", ["a", "b"])
    assert client.call("delete")["points_selector"].points == [
        QdrantVectorSearchService._point_id("a"),
        QdrantVectorSearchService._point_id("b"),
    ]


def test_count_reads_the_number_off_the_response(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    client.points = [Point("a"), Point("b")]
    assert service.count("chunks") == 2


def test_scroll_passes_the_offset_through(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.scroll("chunks", limit=5, offset="7")
    sent = client.call("scroll")
    assert (sent["limit"], sent["offset"]) == (5, "7")


def test_scroll_reports_the_next_offset_as_a_string(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    client.next_offset = 7
    _, offset = service.scroll("chunks")
    assert offset == "7"


def test_scroll_reports_no_next_offset_at_the_end(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    client.next_offset = None
    _, offset = service.scroll("chunks")
    assert offset is None


def test_scroll_translates_its_filters(
    service: QdrantVectorSearchService, client: RecordingClient
) -> None:
    service.scroll("chunks", filters={"tenant": "acme"})
    assert client.call("scroll")["scroll_filter"].must[0].key == "tenant"


# Vector extraction


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ([0.1, 0.2], [0.1, 0.2]),
        ([1, 2], [1, 2]),
        (None, []),
        ([], []),
        ({"dense": [0.1]}, []),
        ([[0.1], [0.2]], []),
        ("not a vector", []),
    ],
)
def test_only_a_flat_unnamed_vector_is_extracted(
    service: QdrantVectorSearchService, given: Any, expected: list[float]
) -> None:
    """Named and multi-vectors are outside what the shared record can hold."""
    assert service._extract_vector(given) == expected


def test_the_service_names_itself_qdrant(service: QdrantVectorSearchService) -> None:
    assert service.get_service_name() == "Qdrant"
