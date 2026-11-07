"""Pins batching, dimensions and the empty-input shortcut.

Splitting a list to fit a provider's ceiling lives in the abstract service
rather than in each adapter, because an adapter that reimplemented it could
return vectors out of order. The reranker's shortcut is the same idea from the
other end: an empty query or an empty document list never reaches a provider,
so nothing is billed for a call that could not have produced an answer.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage
from openai.types.embedding import Embedding

from base.models.definitions.model import ProviderType
from base.models.view_models.embedding import EmbeddingBatch
from base.models.view_models.exceptions.embedding import EmbeddingConfigError
from base.models.view_models.exceptions.rerank import RerankConfigError
from base.models.view_models.model import EmbeddingConfigurations
from base.models.view_models.rerank import RerankBatch, RerankResult
from base.models.view_models.usage import ProviderUsage
from base.services.reranking import AbstractRerankService
from base.services.reranking.cohere import CohereRerankService
from base.services.text_embedding import AbstractTextEmbeddingService
from base.services.text_embedding.openai import OpenAIEmbeddingService


class RecordingEmbedder(AbstractTextEmbeddingService):
    """An embedder that keeps the batches it was handed."""

    def __init__(self, batch_size: int = 2, dimensions: int | None = 3) -> None:
        self.batches: list[list[str]] = []
        self._size = batch_size
        self._dimensions = dimensions

    @property
    def model(self) -> str:
        return "recorder"

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    @property
    def batch_size(self) -> int:
        return self._size

    def get_service_name(self) -> str:
        return "recorder"

    def _embed(self, texts: list[str]) -> EmbeddingBatch:
        self.batches.append(list(texts))
        return EmbeddingBatch(
            vectors=[[float(len(text))] for text in texts],
            usage=ProviderUsage(input_tokens=len(texts)),
        )

    async def _aembed(self, texts: list[str]) -> EmbeddingBatch:
        return self._embed(texts)


class RecordingReranker(AbstractRerankService):
    """A reranker that keeps what it was asked, and reverses the order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int | None]] = []

    @property
    def model(self) -> str:
        return "recorder"

    def get_service_name(self) -> str:
        return "recorder"

    def _rerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> RerankBatch:
        self.calls.append((query, list(documents), top_n))
        ordered = list(reversed(list(enumerate(documents))))[: top_n or len(documents)]
        return RerankBatch(
            results=[
                RerankResult(index=index, score=1.0 / (position + 1), document=text)
                for position, (index, text) in enumerate(ordered)
            ],
            usage=ProviderUsage(input_tokens=len(documents)),
        )

    async def _arerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> RerankBatch:
        return self._rerank(query, documents, top_n)


def test_a_list_shorter_than_the_batch_size_is_one_request() -> None:
    embedder = RecordingEmbedder(batch_size=10)

    embedder.embed_documents(["a", "b"])

    assert embedder.batches == [["a", "b"]]


def test_a_list_longer_than_the_batch_size_is_split_in_order() -> None:
    embedder = RecordingEmbedder(batch_size=2)

    vectors = embedder.embed_documents(["a", "bb", "ccc", "dddd", "e"])

    assert embedder.batches == [["a", "bb"], ["ccc", "dddd"], ["e"]]
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [1.0]]


async def test_the_async_surface_batches_the_same_way() -> None:
    embedder = RecordingEmbedder(batch_size=2)

    vectors = await embedder.aembed_documents(["a", "bb", "ccc"])

    assert embedder.batches == [["a", "bb"], ["ccc"]]
    assert len(vectors) == 3


def test_an_empty_list_never_reaches_the_provider() -> None:
    embedder = RecordingEmbedder()

    assert embedder.embed_documents([]) == []
    assert embedder.batches == []


async def test_an_empty_list_never_reaches_the_async_provider() -> None:
    embedder = RecordingEmbedder()

    assert await embedder.aembed_documents([]) == []
    assert embedder.batches == []


def test_a_single_query_is_embedded_as_a_batch_of_one() -> None:
    embedder = RecordingEmbedder()

    assert embedder.embed_query("abc") == [3.0]
    assert embedder.batches == [["abc"]]


async def test_a_single_async_query_is_embedded_as_a_batch_of_one() -> None:
    embedder = RecordingEmbedder()

    assert await embedder.aembed_query("abcd") == [4.0]


def test_a_batch_size_of_zero_still_makes_progress() -> None:
    embedder = RecordingEmbedder(batch_size=0)

    embedder.embed_documents(["a", "b"])

    assert embedder.batches == [["a"], ["b"]]


def test_a_failing_batch_stops_the_whole_call() -> None:
    class Failing(RecordingEmbedder):
        def _embed(self, texts: list[str]) -> EmbeddingBatch:
            super()._embed(texts)
            raise RuntimeError("provider said no")

    embedder = Failing(batch_size=2)

    with pytest.raises(RuntimeError, match="provider said no"):
        embedder.embed_documents(["a", "b", "c"])
    assert embedder.batches == [["a", "b"]]


def test_the_configured_dimensions_are_sent_when_the_deployment_pins_one() -> None:
    service = object.__new__(OpenAIEmbeddingService)
    service._model = "text-embedding-3-small"
    service._dimensions = 256

    assert service._request_kwargs(["a"]) == {
        "input": ["a"],
        "model": "text-embedding-3-small",
        "dimensions": 256,
    }


def test_no_dimensions_means_the_argument_is_left_off_entirely() -> None:
    service = object.__new__(OpenAIEmbeddingService)
    service._model = "text-embedding-3-small"
    service._dimensions = None

    assert "dimensions" not in service._request_kwargs(["a"])


def test_a_configured_embedding_service_reports_its_knobs() -> None:
    service = OpenAIEmbeddingService(
        EmbeddingConfigurations(
            api_host="http://localhost:1",
            api_key="k",
            model_name="text-embedding-3-small",
            api_type=ProviderType.OPENAI,
            batch_size=7,
            dimensions=256,
        )
    )

    assert (service.model, service.batch_size, service.dimensions) == (
        "text-embedding-3-small",
        7,
        256,
    )
    assert service.get_service_name() == "openai"


def test_an_unset_batch_size_falls_back_to_the_documented_default() -> None:
    service = OpenAIEmbeddingService(
        EmbeddingConfigurations(
            api_host="http://localhost:2",
            api_key="k",
            model_name="text-embedding-3-large",
            api_type=ProviderType.OPENAI,
        )
    )

    assert service.batch_size == OpenAIEmbeddingService.DEFAULT_BATCH_SIZE


def test_an_embedding_model_name_is_required() -> None:
    with pytest.raises(EmbeddingConfigError, match="model_name is required"):
        OpenAIEmbeddingService(
            EmbeddingConfigurations(api_host="http://localhost:3", api_key="k")
        )


def test_vectors_are_returned_in_the_order_they_were_asked_for() -> None:
    """The API does not promise response order, so the adapter sorts by index."""
    response = CreateEmbeddingResponse(
        data=[
            Embedding(embedding=[0.3], index=2, object="embedding"),
            Embedding(embedding=[0.1], index=0, object="embedding"),
            Embedding(embedding=[0.2], index=1, object="embedding"),
        ],
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=7, total_tokens=7),
    )

    batch = OpenAIEmbeddingService._to_batch(response)

    assert batch.vectors == [[0.1], [0.2], [0.3]]
    assert batch.usage is not None
    assert batch.usage.input_tokens == 7


def test_an_embedding_response_without_usage_reports_none() -> None:
    response = CreateEmbeddingResponse(
        data=[Embedding(embedding=[0.1], index=0, object="embedding")],
        model="text-embedding-3-small",
        object="list",
        usage=Usage(prompt_tokens=0, total_tokens=0),
    )
    response.usage = None  # type: ignore[assignment]

    assert OpenAIEmbeddingService._to_batch(response).usage is None


def test_an_empty_document_list_never_reaches_the_reranker() -> None:
    reranker = RecordingReranker()

    assert reranker.rerank("query", []) == []
    assert reranker.calls == []


async def test_an_empty_document_list_never_reaches_the_async_reranker() -> None:
    reranker = RecordingReranker()

    assert await reranker.arerank("query", []) == []
    assert reranker.calls == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_a_blank_query_never_reaches_the_reranker(query: str) -> None:
    reranker = RecordingReranker()

    assert reranker.rerank(query, ["a", "b"]) == []
    assert reranker.calls == []


def test_a_reranked_result_keeps_its_place_in_the_original_list() -> None:
    reranker = RecordingReranker()

    results = reranker.rerank("query", ["zero", "one", "two"])

    assert [item.index for item in results] == [2, 1, 0]
    assert [item.document for item in results] == ["two", "one", "zero"]


async def test_the_async_reranker_behaves_the_same_way() -> None:
    reranker = RecordingReranker()

    results = await reranker.arerank("query", ["zero", "one"], top_n=1)

    assert [item.index for item in results] == [1]
    assert reranker.calls == [("query", ["zero", "one"], 1)]


def test_a_failing_reranker_propagates_rather_than_returning_nothing() -> None:
    class Failing(RecordingReranker):
        def _rerank(
            self, query: str, documents: list[str], top_n: int | None
        ) -> RerankBatch:
            raise RuntimeError("provider said no")

    with pytest.raises(RuntimeError, match="provider said no"):
        Failing().rerank("query", ["a"])


def test_cohere_results_are_mapped_back_to_the_documents_they_scored() -> None:
    response = SimpleNamespace(
        results=[
            SimpleNamespace(index=1, relevance_score=0.9),
            SimpleNamespace(index=0, relevance_score=0.1),
        ],
        meta=SimpleNamespace(
            tokens=SimpleNamespace(input_tokens=5, output_tokens=0),
            cached_tokens=2,
        ),
    )

    batch = CohereRerankService._to_batch(response, ["zero", "one"])

    assert [(item.index, item.document) for item in batch.results] == [
        (1, "one"),
        (0, "zero"),
    ]
    assert batch.usage is not None
    assert (batch.usage.input_tokens, batch.usage.cached_input_tokens) == (5, 2)


def test_a_cohere_reply_without_metadata_reports_no_usage() -> None:
    response = SimpleNamespace(results=[], meta=None)

    assert CohereRerankService._to_batch(response, []).usage is None


def test_a_cohere_rerank_model_is_required() -> None:
    config: Any = SimpleNamespace(cohere_rerank_model="", cohere_api_key="k")

    with pytest.raises(RerankConfigError, match="cohere_rerank_model is required"):
        CohereRerankService.__init__(object.__new__(CohereRerankService), config)


def test_a_configured_cohere_service_reports_its_model_and_name() -> None:
    config: Any = SimpleNamespace(
        cohere_rerank_model="rerank-v4.0-fast", cohere_api_key="k"
    )
    service = object.__new__(CohereRerankService)
    CohereRerankService.__init__(service, config)

    assert service.model == "rerank-v4.0-fast"
    assert service.get_service_name() == "cohere-rerank"


def test_a_cohere_client_is_not_built_until_it_is_needed() -> None:
    config: Any = SimpleNamespace(
        cohere_rerank_model="rerank-v4.0-fast", cohere_api_key="k"
    )
    service = object.__new__(CohereRerankService)
    CohereRerankService.__init__(service, config)

    assert service._sync_client is None
    assert service._async_client is None
