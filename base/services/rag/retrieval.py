"""Finding the chunks that answer a question."""

from typing import Any

from base.models.view_models.rag import RetrievedChunk
from base.services.rag.ingestion import (
    CHUNK_INDEX_FIELD,
    DOCUMENT_ID_FIELD,
    TEXT_FIELD,
)
from base.services.text_embedding.openai_embedding import OpenAITextEmbeddingService
from base.services.vector_search import AbstractVectorSearchService


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
    ) -> None:
        self._vector_service = vector_service
        self._collection = collection
        self._embedding_service: OpenAITextEmbeddingService = (
            OpenAITextEmbeddingService()
        )

    async def asearch(
        self,
        query: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """The `limit` chunks closest to `query`, best first."""
        if not query.strip():
            return []

        vector = await self._embedding_service.aembed_query(query)
        results = await self._vector_service.asearch(
            self._collection,
            vector=vector,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )
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
