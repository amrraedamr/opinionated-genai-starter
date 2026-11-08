"""Which backends can search text as well as vectors, and what happens when none can.

`SupportsLexicalSearch` is deliberately off `AbstractVectorSearchService`: a
backend without full-text should not be made to fake one. Callers ask with
`isinstance` and fall back to dense-only retrieval, so the answer each backend
gives to that question is the whole contract.
"""

from typing import Any

import pytest

from base.models.view_models.vector_search import VectorSearchResult
from base.services.configuration.mixins import (
    AzureSearchSettingsMixin,
    QdrantSettingsMixin,
)
from base.services.vector_search import SupportsLexicalSearch
from base.services.vector_search.azure_ai_search import AzureAISearchService
from base.services.vector_search.pgvector import PgVectorSearchService
from base.services.vector_search.qdrant import QdrantVectorSearchService


class LexicalStub:
    """The smallest thing that satisfies the protocol."""

    async def alexical_search(
        self,
        collection: str,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        return [VectorSearchResult(id="a", score=1.0)]


class DenseOnlyStub:
    """A backend with vectors and no text index."""

    async def asearch(self, collection: str, vector: list[float]) -> list[Any]:
        return []


def test_the_protocol_asks_for_one_method() -> None:
    assert "alexical_search" in SupportsLexicalSearch.__protocol_attrs__


def test_pgvector_satisfies_the_protocol() -> None:
    """Postgres has `to_tsvector`, so it is the backend that does both."""
    assert issubclass(PgVectorSearchService, SupportsLexicalSearch)


def test_qdrant_does_not_satisfy_the_protocol() -> None:
    assert not issubclass(QdrantVectorSearchService, SupportsLexicalSearch)


def test_azure_ai_search_satisfies_the_protocol() -> None:
    assert issubclass(AzureAISearchService, SupportsLexicalSearch)


def test_a_qdrant_instance_is_not_a_lexical_backend(make_settings: Any) -> None:
    make_settings(QdrantSettingsMixin, qdrant_url="http://localhost:6333")
    assert not isinstance(QdrantVectorSearchService(), SupportsLexicalSearch)


def test_an_azure_instance_is_a_lexical_backend(make_settings: Any) -> None:
    make_settings(
        AzureSearchSettingsMixin,
        azure_search_endpoint="https://x.search.windows.net",
        azure_search_api_key="k",
    )
    assert isinstance(AzureAISearchService(), SupportsLexicalSearch)


def test_anything_with_the_method_satisfies_the_protocol() -> None:
    """Structural, not nominal: no registration or base class is needed."""
    assert isinstance(LexicalStub(), SupportsLexicalSearch)


def test_a_dense_only_backend_does_not() -> None:
    assert not isinstance(DenseOnlyStub(), SupportsLexicalSearch)


def test_retrieval_finds_the_lexical_half_of_a_backend_that_has_one(
    make_settings: Any,
) -> None:
    from base.services.rag.retrieval import RagRetrievalService

    make_settings()
    service = RagRetrievalService(
        vector_service=LexicalStub(),  # type: ignore[arg-type]
        collection="chunks",
        embedding_service=object(),  # type: ignore[arg-type]
    )
    assert service._lexical is not None


def test_retrieval_falls_back_to_dense_only_for_a_backend_without_one(
    make_settings: Any,
) -> None:
    from base.services.rag.retrieval import RagRetrievalService

    make_settings(QdrantSettingsMixin, qdrant_url="http://localhost:6333")
    service = RagRetrievalService(
        vector_service=QdrantVectorSearchService(),
        collection="chunks",
        embedding_service=object(),  # type: ignore[arg-type]
    )
    assert service._lexical is None


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
async def test_retrieval_of_a_blank_question_searches_nothing(
    make_settings: Any, query: str
) -> None:
    from base.services.rag.retrieval import RagRetrievalService

    make_settings()
    service = RagRetrievalService(
        vector_service=DenseOnlyStub(),  # type: ignore[arg-type]
        collection="chunks",
        embedding_service=object(),  # type: ignore[arg-type]
    )
    assert await service.asearch(query) == []
