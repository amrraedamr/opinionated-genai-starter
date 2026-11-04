"""Finding the chunks that answer a question."""

from typing import Any

from base.models.view_models.rag import RetrievedChunk
from base.models.view_models.vector_search import VectorSearchResult
from base.services.configuration import get_settings
from base.services.rag.ingestion import (
    CHUNK_INDEX_FIELD,
    DOCUMENT_ID_FIELD,
    TEXT_FIELD,
)
from base.services.reranking import AbstractRerankService
from base.services.text_embedding import AbstractTextEmbeddingService
from base.services.text_embedding.openai import OpenAIEmbeddingService
from base.services.vector_search import (
    AbstractVectorSearchService,
    SupportsLexicalSearch,
)
from base.tools.fusion import reciprocal_rank_fusion


class RagRetrievalService:
    """Embeds a question and returns the chunks nearest to it.

    The text itself is stored alongside each vector at ingestion time, so a
    search answers with the passage rather than an id the caller then has to
    look up somewhere else.
    """

    def __init__(
        self,
        vector_service: AbstractVectorSearchService,
        collection: str,
        reranker: AbstractRerankService | None = None,
        overfetch: int | None = None,
        embedding_service: AbstractTextEmbeddingService | None = None,
    ) -> None:
        self._vector_service = vector_service
        self._collection = collection
        self._reranker = reranker
        self._overfetch = max(overfetch or self._configured_overfetch(), 1)
        # No-argument default on purpose: an embedding service is a
        # KeyedSingleton keyed on its arguments, so passing an explicit
        # None would resolve a different instance than a bare one.
        self._embedding_service: AbstractTextEmbeddingService = (
            embedding_service or OpenAIEmbeddingService()
        )

    @staticmethod
    def _configured_overfetch() -> int:
        """`rerank_overfetch` from settings, for services that declare it."""
        return int(getattr(get_settings(), "rerank_overfetch", 4))

    @property
    def _lexical(self) -> SupportsLexicalSearch | None:
        """The backend's full-text search, when it has one."""
        service = self._vector_service
        return service if isinstance(service, SupportsLexicalSearch) else None

    async def asearch(
        self,
        query: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """The `limit` chunks closest to `query`, best first.

        Over-fetches when a reranker or a lexical backend is available, since
        both improve on the order vector proximity alone produces.
        """
        if not query.strip():
            return []

        fetch = limit * self._overfetch if (self._reranker or self._lexical) else limit
        vector = await self._embedding_service.aembed_query(query)
        results = await self._vector_service.asearch(
            self._collection,
            vector=vector,
            limit=fetch,
            filters=filters,
            score_threshold=score_threshold,
        )
        results = await self._afuse(query, results, fetch, filters)
        results = await self._arerank(query, results, limit)
        return [
            RetrievedChunk(
                text=str(result.metadata.get(TEXT_FIELD, "")),
                score=result.score,
                document_id=str(result.metadata.get(DOCUMENT_ID_FIELD, "")),
                chunk_index=int(result.metadata.get(CHUNK_INDEX_FIELD, -1)),
                metadata=result.metadata,
            )
            for result in results
        ]

    async def _afuse(
        self,
        query: str,
        dense: list[VectorSearchResult],
        limit: int,
        filters: dict[str, Any] | None,
    ) -> list[VectorSearchResult]:
        """Fuse dense and lexical hits by rank, when the backend has both.

        Scores from the two are on different scales, so they are merged by
        position rather than value.
        """
        lexical_service = self._lexical
        if lexical_service is None:
            return dense
        lexical = await lexical_service.alexical_search(
            self._collection, query=query, limit=limit, filters=filters
        )
        if not lexical:
            return dense
        by_id = {result.id: result for result in [*lexical, *dense]}
        fused = reciprocal_rank_fusion(
            [[r.id for r in dense], [r.id for r in lexical]], limit=limit
        )
        return [
            by_id[identifier].model_copy(update={"score": score})
            for identifier, score in fused
            if identifier in by_id
        ]

    async def _arerank(
        self, query: str, results: list[VectorSearchResult], limit: int
    ) -> list[VectorSearchResult]:
        """Reorder with the cross-encoder, when one was supplied."""
        if self._reranker is None:
            return results[:limit]
        documents = [str(r.metadata.get(TEXT_FIELD, "")) for r in results]
        reranked = await self._reranker.arerank(query, documents, top_n=limit)
        if not reranked:
            return results[:limit]
        return [
            results[item.index].model_copy(update={"score": item.score})
            for item in reranked
        ]

    async def acontext(
        self,
        query: str,
        limit: int = 5,
        separator: str = "\n\n---\n\n",
        **kwargs: Any,
    ) -> str:
        """The same chunks, joined into one block ready to drop into a prompt."""
        chunks = await self.asearch(query, limit=limit, **kwargs)
        return separator.join(chunk.text for chunk in chunks if chunk.text)
