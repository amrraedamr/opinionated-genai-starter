"""The async half of the Azure AI Search adapter, which differs from the sync half.

The async SDK awaits `search` and then iterates it, and `get_count` is a
coroutine there rather than a plain method. Those differences are where a
sync-only test would miss a break, so each async path is exercised against an
async recorder of its own.
"""

from typing import Any

import pytest

from base.models.definitions.vector_search import VECTOR_UPSERT_BATCH_SIZE
from base.models.view_models.vector_search import VectorRecord
from base.services.configuration.mixins import AzureSearchSettingsMixin
from base.services.vector_search.azure_ai_search import AzureAISearchService


class AsyncSearchPage:
    """An async iterable of documents that also reports a total count."""

    def __init__(self, documents: list[dict[str, Any]], total: int | None) -> None:
        self.documents = documents
        self.total = total

    def __aiter__(self) -> Any:
        async def iterator() -> Any:
            for document in self.documents:
                yield document

        return iterator()

    async def get_count(self) -> int | None:
        return self.total


class AsyncRecordingSearchClient:
    """Stands in for the async `SearchClient`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.documents: list[dict[str, Any]] = []
        self.total: int | None = None
        self.count_value: int = 0
        self.missing: set[str] = set()

    def call(self, method: str) -> Any:
        for name, payload in reversed(self.calls):
            if name == method:
                return payload
        raise AssertionError(f"{method} was never called")

    def every(self, method: str) -> list[Any]:
        return [payload for name, payload in self.calls if name == method]

    async def search(self, **kwargs: Any) -> AsyncSearchPage:
        self.calls.append(("search", kwargs))
        return AsyncSearchPage(self.documents, self.total)

    async def get_document(self, key: str) -> dict[str, Any]:
        self.calls.append(("get_document", key))
        from azure.core.exceptions import ResourceNotFoundError

        if key in self.missing:
            raise ResourceNotFoundError("no such document")
        return {"id": key, "vector": [0.1], "metadata": '{"text": "hi"}'}

    async def merge_or_upload_documents(self, documents: list[dict[str, Any]]) -> None:
        self.calls.append(("merge_or_upload_documents", documents))

    async def delete_documents(self, documents: list[dict[str, Any]]) -> None:
        self.calls.append(("delete_documents", documents))

    async def get_document_count(self) -> int:
        self.calls.append(("get_document_count", None))
        return self.count_value


class AsyncRecordingIndexClient:
    """Stands in for the async `SearchIndexClient`."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.names: list[str] = []

    async def create_or_update_index(self, index: Any) -> Any:
        self.calls.append(("create_or_update_index", index))
        return index

    async def delete_index(self, name: str) -> None:
        self.calls.append(("delete_index", name))

    def list_index_names(self) -> Any:
        self.calls.append(("list_index_names", None))
        names = list(self.names)

        async def iterator() -> Any:
            for name in names:
                yield name

        return iterator()

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
) -> AsyncRecordingSearchClient:
    recorder = AsyncRecordingSearchClient()
    monkeypatch.setattr(
        service._client, "get_async_search_client", lambda index_name: recorder
    )
    return recorder


@pytest.fixture
def index_client(service: AzureAISearchService) -> AsyncRecordingIndexClient:
    recorder = AsyncRecordingIndexClient()
    service._client._async_index_client = recorder  # type: ignore[assignment]
    return recorder


async def test_creating_a_collection_sends_the_index_definition(
    service: AzureAISearchService, index_client: AsyncRecordingIndexClient
) -> None:
    await service.acreate_collection("chunks", vector_size=1536)
    index = index_client.call("create_or_update_index")
    assert index.name == "chunks"
    assert index.fields[1].vector_search_dimensions == 1536


async def test_deleting_a_collection_names_the_index(
    service: AzureAISearchService, index_client: AsyncRecordingIndexClient
) -> None:
    await service.adelete_collection("chunks")
    assert index_client.call("delete_index") == "chunks"


async def test_collection_exists_walks_the_async_listing(
    service: AzureAISearchService, index_client: AsyncRecordingIndexClient
) -> None:
    index_client.names = ["chunks"]
    assert await service.acollection_exists("chunks") is True
    assert await service.acollection_exists("missing") is False


async def test_a_record_becomes_a_document_with_metadata_as_json(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    await service.aupsert(
        "chunks", [VectorRecord(id="a", vector=[0.1], metadata={"text": "hi"})]
    )
    assert search_client.call("merge_or_upload_documents") == [
        {"id": "a", "vector": [0.1], "metadata": '{"text": "hi"}'}
    ]


async def test_a_batch_over_the_limit_is_split(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    records = [
        VectorRecord(id=str(i), vector=[0.1])
        for i in range(VECTOR_UPSERT_BATCH_SIZE + 2)
    ]
    await service.aupsert("chunks", records)
    sent = search_client.every("merge_or_upload_documents")
    assert [len(batch) for batch in sent] == [VECTOR_UPSERT_BATCH_SIZE, 2]


async def test_a_search_sends_a_vectorized_query_and_no_search_text(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    await service.asearch("chunks", [0.1, 0.2], limit=4)
    sent = search_client.call("search")
    assert sent["search_text"] is None
    assert sent["vector_queries"][0].vector == [0.1, 0.2]
    assert sent["top"] == 4


async def test_a_search_sends_an_odata_filter(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    await service.asearch("chunks", [0.1], filters={"tenant": "acme"})
    assert search_client.call("search")["filter"] == "tenant eq 'acme'"


async def test_a_scored_document_becomes_a_result(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    search_client.documents = [
        {"id": "a", "vector": [0.1], "metadata": '{"text": "hi"}', "@search.score": 0.9}
    ]
    result = (await service.asearch("chunks", [0.1]))[0]
    assert (result.id, result.score) == ("a", 0.9)
    assert result.metadata == {"text": "hi"}


async def test_the_score_threshold_drops_low_hits_on_arrival(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    search_client.documents = [
        {"id": "a", "@search.score": 0.9},
        {"id": "b", "@search.score": 0.1},
    ]
    results = await service.asearch("chunks", [0.1], score_threshold=0.5)
    assert [r.id for r in results] == ["a"]


async def test_an_empty_index_returns_no_results(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    assert await service.asearch("chunks", [0.1]) == []


async def test_get_reads_one_document_per_id(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    records = await service.aget("chunks", ["a", "b"])
    assert [r.id for r in records] == ["a", "b"]


async def test_get_skips_an_id_that_is_not_there(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    search_client.missing = {"b"}
    assert [r.id for r in await service.aget("chunks", ["a", "b"])] == ["a"]


async def test_get_returns_the_vector_only_when_asked(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    assert (await service.aget("chunks", ["a"]))[0].vector == []
    assert (await service.aget("chunks", ["a"], with_vectors=True))[0].vector == [0.1]


async def test_delete_sends_one_key_document_per_id(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    await service.adelete("chunks", ["a", "b"])
    assert search_client.call("delete_documents") == [{"id": "a"}, {"id": "b"}]


async def test_count_asks_azure_for_the_document_count(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    search_client.count_value = 12
    assert await service.acount("chunks") == 12


async def test_scroll_matches_everything_and_asks_for_the_total(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    await service.ascroll("chunks", limit=5)
    sent = search_client.call("search")
    assert sent["search_text"] == "*"
    assert sent["include_total_count"] is True
    assert sent["skip"] == 0


async def test_scroll_reports_the_next_offset_while_rows_remain(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    search_client.documents = [{"id": "a"}, {"id": "b"}]
    search_client.total = 10
    _, offset = await service.ascroll("chunks", limit=2)
    assert offset == "2"


async def test_scroll_reports_no_next_offset_at_the_end(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    search_client.documents = [{"id": "a"}]
    search_client.total = 1
    _, offset = await service.ascroll("chunks")
    assert offset is None


async def test_scroll_resumes_from_the_offset_it_was_handed(
    service: AzureAISearchService, search_client: AsyncRecordingSearchClient
) -> None:
    search_client.documents = [{"id": "c"}]
    search_client.total = 10
    _, offset = await service.ascroll("chunks", limit=1, offset="5")
    assert search_client.call("search")["skip"] == 5
    assert offset == "6"
