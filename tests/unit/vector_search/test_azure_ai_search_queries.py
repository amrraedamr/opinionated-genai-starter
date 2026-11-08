"""The requests Azure AI Search is sent, and what comes back out of them.

Every call here is a translation that has no server to check it: an OData
filter string, a `VectorizedQuery`, a document whose metadata has been
flattened to JSON. The adapter sends `search_text=None` on a similarity
search, so the backend's own hybrid ranking is never asked for — that is
pinned rather than assumed.
"""

import json
from typing import Any

import pytest

from base.models.definitions.vector_search import VECTOR_UPSERT_BATCH_SIZE
from base.models.view_models.vector_search import VectorRecord
from base.services.configuration.mixins import AzureSearchSettingsMixin
from base.services.vector_search import SupportsLexicalSearch
from base.services.vector_search.azure_ai_search import AzureAISearchService


class SearchPage:
    """An iterable of documents that can also report a total count."""

    def __init__(
        self, documents: list[dict[str, Any]], total: int | None = None
    ) -> None:
        self.documents = documents
        self.total = total

    def __iter__(self) -> Any:
        return iter(self.documents)

    def get_count(self) -> int | None:
        return self.total


class RecordingSearchClient:
    """Stands in for `SearchClient`, keeping every request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.documents: list[dict[str, Any]] = []
        self.total: int | None = None
        self.count_value: int = 0
        self.missing: set[str] = set()

    def _record(self, method: str, payload: Any) -> None:
        self.calls.append((method, payload))

    def call(self, method: str) -> Any:
        for name, payload in reversed(self.calls):
            if name == method:
                return payload
        raise AssertionError(f"{method} was never called")

    def every(self, method: str) -> list[Any]:
        return [payload for name, payload in self.calls if name == method]

    def search(self, **kwargs: Any) -> SearchPage:
        self._record("search", kwargs)
        return SearchPage(self.documents, self.total)

    def get_document(self, key: str) -> dict[str, Any]:
        self._record("get_document", key)
        from azure.core.exceptions import ResourceNotFoundError

        if key in self.missing:
            raise ResourceNotFoundError("no such document")
        return {"id": key, "vector": [0.1], "metadata": '{"text": "hi"}'}

    def merge_or_upload_documents(self, documents: list[dict[str, Any]]) -> None:
        self._record("merge_or_upload_documents", documents)

    def delete_documents(self, documents: list[dict[str, Any]]) -> None:
        self._record("delete_documents", documents)

    def get_document_count(self) -> int:
        self._record("get_document_count", None)
        return self.count_value


class RecordingIndexClient:
    """Stands in for `SearchIndexClient`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.names: list[str] = []

    def create_or_update_index(self, index: Any) -> Any:
        self.calls.append(("create_or_update_index", index))
        return index

    def delete_index(self, name: str) -> None:
        self.calls.append(("delete_index", name))

    def list_index_names(self) -> list[str]:
        self.calls.append(("list_index_names", None))
        return list(self.names)

    def call(self, method: str) -> Any:
        for name, payload in reversed(self.calls):
            if name == method:
                return payload
        raise AssertionError(f"{method} was never called")


@pytest.fixture
def service(make_settings: Any) -> AzureAISearchService:
    make_settings(
        AzureSearchSettingsMixin,
        azure_search_endpoint="https://account.search.windows.net",
        azure_search_api_key="the-key",
    )
    return AzureAISearchService()


@pytest.fixture
def search_client(
    service: AzureAISearchService, monkeypatch: pytest.MonkeyPatch
) -> RecordingSearchClient:
    recorder = RecordingSearchClient()
    monkeypatch.setattr(
        service._client, "get_search_client", lambda index_name: recorder
    )
    return recorder


@pytest.fixture
def index_client(service: AzureAISearchService) -> RecordingIndexClient:
    recorder = RecordingIndexClient()
    service._client._sync_index_client = recorder  # type: ignore[assignment]
    return recorder


# Index lifecycle


def test_creating_a_collection_sends_the_index_definition(
    service: AzureAISearchService, index_client: RecordingIndexClient
) -> None:
    service.create_collection("chunks", vector_size=8)
    assert index_client.call("create_or_update_index").name == "chunks"


def test_creating_a_collection_is_an_upsert(
    service: AzureAISearchService, index_client: RecordingIndexClient
) -> None:
    """Azure has no create-only call here, so a rerun updates rather than fails."""
    service.create_collection("chunks", vector_size=8)
    service.create_collection("chunks", vector_size=8)
    assert len(index_client.calls) == 2


def test_deleting_a_collection_names_the_index(
    service: AzureAISearchService, index_client: RecordingIndexClient
) -> None:
    service.delete_collection("chunks")
    assert index_client.call("delete_index") == "chunks"


def test_collection_exists_looks_the_name_up_in_the_listing(
    service: AzureAISearchService, index_client: RecordingIndexClient
) -> None:
    index_client.names = ["chunks", "docs"]
    assert service.collection_exists("chunks") is True
    assert service.collection_exists("missing") is False


# Upsert


def test_a_record_becomes_a_document_with_metadata_as_json(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.upsert(
        "chunks", [VectorRecord(id="a", vector=[0.1], metadata={"text": "hi"})]
    )
    document = search_client.call("merge_or_upload_documents")[0]
    assert document == {"id": "a", "vector": [0.1], "metadata": '{"text": "hi"}'}


def test_empty_metadata_becomes_an_empty_json_object(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.upsert("chunks", [VectorRecord(id="a", vector=[0.1])])
    assert search_client.call("merge_or_upload_documents")[0]["metadata"] == "{}"


def test_non_ascii_metadata_survives_the_json_round_trip(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.upsert(
        "chunks", [VectorRecord(id="a", vector=[0.1], metadata={"t": "héllo 🎉"})]
    )
    written = search_client.call("merge_or_upload_documents")[0]["metadata"]
    assert json.loads(written) == {"t": "héllo 🎉"}


def test_an_empty_upsert_sends_nothing(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.upsert("chunks", [])
    assert search_client.every("merge_or_upload_documents") == []


def test_a_batch_over_the_limit_is_split(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    records = [
        VectorRecord(id=str(i), vector=[0.1])
        for i in range(VECTOR_UPSERT_BATCH_SIZE + 2)
    ]
    service.upsert("chunks", records)
    sent = search_client.every("merge_or_upload_documents")
    assert [len(batch) for batch in sent] == [VECTOR_UPSERT_BATCH_SIZE, 2]


# Search


def test_a_search_sends_a_vectorized_query_over_the_vector_field(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.search("chunks", [0.1, 0.2], limit=4)
    query = search_client.call("search")["vector_queries"][0]
    assert query.vector == [0.1, 0.2]
    assert query.k_nearest_neighbors == 4
    assert query.fields == "vector"


def test_a_search_caps_the_result_count_as_well_as_the_neighbours(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.search("chunks", [0.1], limit=4)
    assert search_client.call("search")["top"] == 4


def test_a_similarity_search_never_asks_for_lexical_ranking(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    """Azure would fuse text and vector hits; the adapter does not use that."""
    service.search("chunks", [0.1])
    assert search_client.call("search")["search_text"] is None


def test_the_backend_is_not_offered_as_a_lexical_one(
    service: AzureAISearchService,
) -> None:
    """The guard for the test above: callers are told there is no text search."""
    assert not isinstance(service, SupportsLexicalSearch)


def test_a_search_without_filters_sends_no_filter(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.search("chunks", [0.1])
    assert search_client.call("search")["filter"] is None


def test_a_search_sends_an_odata_filter(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.search("chunks", [0.1], filters={"tenant": "acme"})
    assert search_client.call("search")["filter"] == "tenant eq 'acme'"


def test_a_scored_document_becomes_a_result(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    search_client.documents = [
        {"id": "a", "vector": [0.1], "metadata": '{"text": "hi"}', "@search.score": 0.9}
    ]
    result = service.search("chunks", [0.1])[0]
    assert (result.id, result.score, result.vector) == ("a", 0.9, [0.1])
    assert result.metadata == {"text": "hi"}


def test_a_document_without_a_score_is_read_as_zero(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    search_client.documents = [{"id": "a"}]
    assert service.search("chunks", [0.1])[0].score == 0.0


def test_the_score_threshold_is_applied_here_rather_than_by_azure(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    """Azure has no threshold parameter, so low hits are dropped on arrival."""
    search_client.documents = [
        {"id": "a", "@search.score": 0.9},
        {"id": "b", "@search.score": 0.1},
    ]
    results = service.search("chunks", [0.1], score_threshold=0.5)
    assert [r.id for r in results] == ["a"]
    assert "score_threshold" not in search_client.call("search")


def test_a_threshold_that_excludes_everything_returns_nothing(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    search_client.documents = [{"id": "a", "@search.score": 0.1}]
    assert service.search("chunks", [0.1], score_threshold=0.5) == []


def test_an_empty_index_returns_no_results(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    search_client.documents = []
    assert service.search("chunks", [0.1]) == []


# Get, delete and count


def test_get_reads_one_document_per_id(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    records = service.get("chunks", ["a", "b"])
    assert [r.id for r in records] == ["a", "b"]
    assert search_client.every("get_document") == ["a", "b"]


def test_get_skips_an_id_that_is_not_there(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    search_client.missing = {"b"}
    assert [r.id for r in service.get("chunks", ["a", "b"])] == ["a"]


def test_get_leaves_the_vector_out_unless_asked(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    assert service.get("chunks", ["a"])[0].vector == []


def test_get_returns_the_vector_when_asked(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    assert service.get("chunks", ["a"], with_vectors=True)[0].vector == [0.1]


def test_delete_sends_one_key_document_per_id(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.delete("chunks", ["a", "b"])
    assert search_client.call("delete_documents") == [{"id": "a"}, {"id": "b"}]


def test_delete_batches_at_the_shared_limit(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.delete("chunks", [str(i) for i in range(VECTOR_UPSERT_BATCH_SIZE + 3)])
    sent = search_client.every("delete_documents")
    assert [len(batch) for batch in sent] == [VECTOR_UPSERT_BATCH_SIZE, 3]


def test_deleting_nothing_sends_nothing(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.delete("chunks", [])
    assert search_client.every("delete_documents") == []


def test_count_asks_azure_for_the_document_count(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    search_client.count_value = 12
    assert service.count("chunks") == 12


# Scroll


def test_scroll_matches_everything_and_asks_for_the_total(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.scroll("chunks", limit=5)
    sent = search_client.call("search")
    assert sent["search_text"] == "*"
    assert sent["top"] == 5
    assert sent["include_total_count"] is True


def test_scroll_turns_the_offset_into_a_skip(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.scroll("chunks", offset="20")
    assert search_client.call("search")["skip"] == 20


def test_scroll_starts_at_zero_without_an_offset(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    service.scroll("chunks")
    assert search_client.call("search")["skip"] == 0


def test_scroll_reports_the_next_offset_while_rows_remain(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    search_client.documents = [{"id": "a"}, {"id": "b"}]
    search_client.total = 10
    _, offset = service.scroll("chunks", limit=2)
    assert offset == "2"


def test_scroll_reports_no_next_offset_at_the_end(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    search_client.documents = [{"id": "a"}]
    search_client.total = 1
    _, offset = service.scroll("chunks")
    assert offset is None


def test_scroll_without_a_total_reports_no_next_offset(
    service: AzureAISearchService, search_client: RecordingSearchClient
) -> None:
    search_client.documents = [{"id": "a"}]
    search_client.total = None
    _, offset = service.scroll("chunks")
    assert offset is None


# OData filters


def test_a_string_value_is_quoted(service: AzureAISearchService) -> None:
    assert service._build_filter({"tenant": "acme"}) == "tenant eq 'acme'"


def test_a_number_is_not_quoted(service: AzureAISearchService) -> None:
    assert service._build_filter({"year": 2026}) == "year eq 2026"


def test_a_float_is_not_quoted(service: AzureAISearchService) -> None:
    assert service._build_filter({"score": 0.5}) == "score eq 0.5"


def test_a_boolean_is_lower_cased_for_odata(service: AzureAISearchService) -> None:
    assert service._build_filter({"active": True}) == "active eq true"
    assert service._build_filter({"active": False}) == "active eq false"


def test_several_fields_are_joined_with_and(service: AzureAISearchService) -> None:
    assert (
        service._build_filter({"tenant": "acme", "year": 2026})
        == "tenant eq 'acme' and year eq 2026"
    )


def test_a_list_becomes_a_parenthesised_or(service: AzureAISearchService) -> None:
    assert service._build_filter({"tag": ["a", "b"]}) == "(tag eq 'a' or tag eq 'b')"


def test_no_filter_sends_no_filter(service: AzureAISearchService) -> None:
    assert service._build_filter(None) is None
    assert service._build_filter({}) is None


def test_a_boolean_is_rendered_the_way_odata_spells_one(
    service: AzureAISearchService,
) -> None:
    """`True` is Python; Azure rejects it. The list branch used to emit it."""
    assert service._build_filter({"active": True}) == "active eq true"
    assert service._build_filter({"active": [True, False]}) == (
        "(active eq true or active eq false)"
    )


def test_a_none_value_narrows_rather_than_disappears(
    service: AzureAISearchService,
) -> None:
    """A dropped condition widens the query, which is the dangerous direction.

    Asking for `tenant is null` and being handed every tenant is worse than
    an error, because it looks like a result.
    """
    assert service._build_filter({"tenant": None}) == "tenant eq null"


def test_a_none_alongside_a_real_condition_keeps_both(
    service: AzureAISearchService,
) -> None:
    assert service._build_filter({"tenant": "acme", "deleted": None}) == (
        "tenant eq 'acme' and deleted eq null"
    )


def test_a_quote_in_a_value_is_escaped(service: AzureAISearchService) -> None:
    """OData doubles a quote inside a literal; leaving it broke the filter."""
    assert service._build_filter({"name": "O'Hara"}) == "name eq 'O''Hara'"
    assert service._build_filter({"name": ["O'Hara"]}) == "(name eq 'O''Hara')"


def test_a_type_the_builder_cannot_render_is_refused(
    service: AzureAISearchService,
) -> None:
    """Silently dropping it returned everything instead of nothing."""
    with pytest.raises(TypeError):
        service._build_filter({"when": object()})


# Document translation


def test_unparseable_metadata_is_read_as_empty(
    service: AzureAISearchService,
) -> None:
    record = service._document_to_record({"id": "a", "metadata": "not json"})
    assert record.metadata == {}


def test_metadata_that_is_not_a_string_is_read_as_empty(
    service: AzureAISearchService,
) -> None:
    record = service._document_to_record({"id": "a", "metadata": {"already": "a dict"}})
    assert record.metadata == {}


def test_a_document_without_an_id_becomes_an_empty_id(
    service: AzureAISearchService,
) -> None:
    assert service._document_to_record({}).id == ""


def test_a_result_keeps_the_vector_whenever_the_document_carried_one(
    service: AzureAISearchService,
) -> None:
    result = service._document_to_search_result({"id": "a", "vector": [0.1]}, 0.5)
    assert result.vector == [0.1]


def test_a_result_without_a_vector_reports_none(
    service: AzureAISearchService,
) -> None:
    assert service._document_to_search_result({"id": "a"}, 0.5).vector is None
