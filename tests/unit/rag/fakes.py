"""In-process stand-ins for the backends RAG talks to.

The vector store is a real dictionary rather than a mock: ingestion's whole
job is which records it writes and which stale ones it removes, and that is
only observable against something that actually stores them.
"""

from __future__ import annotations

from typing import Any

from base.models.view_models.embedding import EmbeddingBatch
from base.models.view_models.rerank import RerankBatch, RerankResult
from base.models.view_models.vector_search import VectorRecord, VectorSearchResult
from base.services.reranking import AbstractRerankService
from base.services.text_embedding import AbstractTextEmbeddingService
from base.services.vector_search import AbstractVectorSearchService


class FakeEmbeddings(AbstractTextEmbeddingService):
    """Deterministic vectors, plus the batches it was asked for."""

    def __init__(self, width: int = 3, batch_size: int = 100) -> None:
        self._width = width
        self._batch_size = batch_size
        self.batches: list[list[str]] = []

    @property
    def model(self) -> str:
        return "fake-embedding"

    @property
    def dimensions(self) -> int | None:
        return self._width

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def get_service_name(self) -> str:
        return "FakeEmbeddings"

    def _vector(self, text: str) -> list[float]:
        seed = sum(ord(ch) * (position + 1) for position, ch in enumerate(text))
        return [float(seed % 9973 + offset) for offset in range(self._width)]

    def _embed(self, texts: list[str]) -> EmbeddingBatch:
        self.batches.append(list(texts))
        return EmbeddingBatch(vectors=[self._vector(t) for t in texts])

    async def _aembed(self, texts: list[str]) -> EmbeddingBatch:
        return self._embed(texts)

    @property
    def embedded(self) -> list[str]:
        return [text for batch in self.batches for text in batch]


class FakeVectorStore(AbstractVectorSearchService):
    """A dictionary pretending to be a vector database."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, VectorRecord]] = {}
        self.created: list[tuple[str, int, str]] = []
        self.searches: list[dict[str, Any]] = []
        self.deleted: list[tuple[str, list[str]]] = []
        self.scroll_page_size: int | None = None

    # Collections

    def create_collection(
        self, collection: str, vector_size: int, distance: str = "cosine", **kwargs: Any
    ) -> None:
        self.created.append((collection, vector_size, distance))
        self.collections.setdefault(collection, {})

    async def acreate_collection(
        self, collection: str, vector_size: int, distance: str = "cosine", **kwargs: Any
    ) -> None:
        self.create_collection(collection, vector_size, distance, **kwargs)

    def delete_collection(self, collection: str) -> None:
        self.collections.pop(collection, None)

    async def adelete_collection(self, collection: str) -> None:
        self.delete_collection(collection)

    def collection_exists(self, collection: str) -> bool:
        return collection in self.collections

    async def acollection_exists(self, collection: str) -> bool:
        return self.collection_exists(collection)

    # Upsert

    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        bucket = self.collections.setdefault(collection, {})
        for record in records:
            bucket[record.id] = record

    async def aupsert(self, collection: str, records: list[VectorRecord]) -> None:
        self.upsert(collection, records)

    # Search

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        self.searches.append(
            {
                "collection": collection,
                "vector": vector,
                "limit": limit,
                "filters": filters,
                "score_threshold": score_threshold,
            }
        )
        matches = self._matching(collection, filters)
        return [
            VectorSearchResult(
                id=record.id, score=1.0 - index / 100, metadata=record.metadata
            )
            for index, record in enumerate(matches[:limit])
        ]

    async def asearch(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        return self.search(collection, vector, limit, filters, score_threshold)

    # Get / delete / count

    def get(
        self, collection: str, ids: list[str], with_vectors: bool = False
    ) -> list[VectorRecord]:
        bucket = self.collections.get(collection, {})
        return [bucket[i] for i in ids if i in bucket]

    async def aget(
        self, collection: str, ids: list[str], with_vectors: bool = False
    ) -> list[VectorRecord]:
        return self.get(collection, ids, with_vectors)

    def delete(self, collection: str, ids: list[str]) -> None:
        self.deleted.append((collection, list(ids)))
        bucket = self.collections.get(collection, {})
        for identifier in ids:
            bucket.pop(identifier, None)

    async def adelete(self, collection: str, ids: list[str]) -> None:
        self.delete(collection, ids)

    def count(self, collection: str) -> int:
        return len(self.collections.get(collection, {}))

    async def acount(self, collection: str) -> int:
        return self.count(collection)

    # Scroll

    def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[VectorRecord], str | None]:
        self.scroll_page_size = limit
        matches = self._matching(collection, filters)
        start = int(offset) if offset else 0
        page = matches[start : start + limit]
        next_offset = start + limit if start + limit < len(matches) else None
        return page, str(next_offset) if next_offset is not None else None

    async def ascroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[VectorRecord], str | None]:
        return self.scroll(collection, limit, offset, filters, with_vectors)

    # Lifecycle

    def health_check(self) -> bool:
        return True

    async def ahealth_check(self) -> bool:
        return True

    def get_service_name(self) -> str:
        return "FakeVectorStore"

    def dispose(self) -> None:
        return None

    async def adispose(self) -> None:
        return None

    # Internals

    def _matching(
        self, collection: str, filters: dict[str, Any] | None
    ) -> list[VectorRecord]:
        records = list(self.collections.get(collection, {}).values())
        if not filters:
            return records
        return [
            record
            for record in records
            if all(record.metadata.get(k) == v for k, v in filters.items())
        ]


class LexicalVectorStore(FakeVectorStore):
    """A backend that also answers full-text queries."""

    def __init__(self, lexical: list[VectorSearchResult] | None = None) -> None:
        super().__init__()
        self._lexical = lexical or []
        self.lexical_queries: list[str] = []

    async def alexical_search(
        self,
        collection: str,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        self.lexical_queries.append(query)
        return list(self._lexical[:limit])


class ReversingReranker(AbstractRerankService):
    """Puts the passages back in the opposite order, so the effect is visible."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int | None]] = []

    @property
    def model(self) -> str:
        return "fake-reranker"

    def get_service_name(self) -> str:
        return "ReversingReranker"

    def _rerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> RerankBatch:
        self.calls.append((query, list(documents), top_n))
        ordered = list(reversed(list(enumerate(documents))))[: top_n or len(documents)]
        return RerankBatch(
            results=[
                RerankResult(index=index, score=1.0 - position / 100, document=document)
                for position, (index, document) in enumerate(ordered)
            ]
        )

    async def _arerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> RerankBatch:
        return self._rerank(query, documents, top_n)
