"""Chunking, ingestion and retrieval, end to end against an in-memory store."""

from __future__ import annotations

from typing import Any

import pytest

from base.models.view_models.rag import Document
from base.models.view_models.vector_search import VectorRecord, VectorSearchResult
from base.services.rag import (
    ChunkingConfig,
    IngestionConfig,
    RagIngestionService,
    RagRetrievalService,
    TextChunker,
)
from base.services.vector_search import AbstractVectorSearchService


class InMemoryVectorSearch(AbstractVectorSearchService):
    """Enough of the vector contract to exercise the pipeline without a server."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, VectorRecord]] = {}
        self.vector_size: int | None = None

    # Collections

    def create_collection(self, collection, vector_size, distance="cosine", **kwargs):
        self.collections.setdefault(collection, {})
        self.vector_size = vector_size

    async def acreate_collection(
        self, collection, vector_size, distance="cosine", **kwargs
    ):
        self.create_collection(collection, vector_size, distance, **kwargs)

    def delete_collection(self, collection):
        self.collections.pop(collection, None)

    async def adelete_collection(self, collection):
        self.delete_collection(collection)

    def collection_exists(self, collection):
        return collection in self.collections

    async def acollection_exists(self, collection):
        return self.collection_exists(collection)

    # Records

    def upsert(self, collection, records):
        store = self.collections.setdefault(collection, {})
        for record in records:
            store[record.id] = record

    async def aupsert(self, collection, records):
        self.upsert(collection, records)

    def delete(self, collection, ids):
        store = self.collections.get(collection, {})
        for identifier in ids:
            store.pop(identifier, None)

    async def adelete(self, collection, ids):
        self.delete(collection, ids)

    def get(self, collection, ids, with_vectors=False):
        store = self.collections.get(collection, {})
        return [store[i] for i in ids if i in store]

    async def aget(self, collection, ids, with_vectors=False):
        return self.get(collection, ids, with_vectors)

    def count(self, collection):
        return len(self.collections.get(collection, {}))

    async def acount(self, collection):
        return self.count(collection)

    # Query

    def _matching(self, collection, filters):
        records = list(self.collections.get(collection, {}).values())
        if not filters:
            return records
        return [
            record
            for record in records
            if all(record.metadata.get(k) == v for k, v in filters.items())
        ]

    def search(
        self, collection, vector, limit=10, filters=None, score_threshold=None
    ):
        scored = [
            VectorSearchResult(
                id=record.id,
                score=_similarity(vector, record.vector),
                metadata=record.metadata,
            )
            for record in self._matching(collection, filters)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        if score_threshold is not None:
            scored = [r for r in scored if r.score >= score_threshold]
        return scored[:limit]

    async def asearch(
        self, collection, vector, limit=10, filters=None, score_threshold=None
    ):
        return self.search(collection, vector, limit, filters, score_threshold)

    def scroll(self, collection, limit=100, offset=None, filters=None, with_vectors=False):
        records = self._matching(collection, filters)
        start = int(offset or 0)
        page = records[start : start + limit]
        following = start + limit
        return page, (str(following) if following < len(records) else None)

    async def ascroll(
        self, collection, limit=100, offset=None, filters=None, with_vectors=False
    ):
        return self.scroll(collection, limit, offset, filters, with_vectors)

    def health_check(self):
        return True

    async def ahealth_check(self):
        return True

    # Lifecycle

    def get_service_name(self):
        return "In-memory vectors"

    def dispose(self):
        self.collections.clear()

    async def adispose(self):
        self.dispose()


def _similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


class FakeEmbeddings:
    """Deterministic embeddings: a word-count vector over a fixed vocabulary."""

    VOCABULARY = ("leave", "holiday", "expenses", "laptop", "penguin")

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        counts = [float(lowered.count(word)) for word in self.VOCABULARY]
        return counts or [0.0] * len(self.VOCABULARY)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def vectors() -> InMemoryVectorSearch:
    return InMemoryVectorSearch()


@pytest.fixture
def ingestion(vectors) -> RagIngestionService:
    service = RagIngestionService(
        vectors,
        IngestionConfig(
            collection="handbook",
            chunking=ChunkingConfig(max_tokens=32, overlap_tokens=4),
        ),
    )
    service._embedding_service = FakeEmbeddings()
    return service


@pytest.fixture
def retrieval(vectors) -> RagRetrievalService:
    service = RagRetrievalService(vectors, "handbook")
    service._embedding_service = FakeEmbeddings()
    return service


def _document(identifier: str, text: str, **metadata: Any) -> Document:
    return Document(id=identifier, text=text, metadata=metadata)


class TestChunking:
    def test_short_text_is_one_chunk(self):
        chunks = TextChunker(ChunkingConfig(max_tokens=100, overlap_tokens=0)).split(
            _document("d", "A short paragraph.")
        )
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].token_count > 0

    def test_paragraphs_are_packed_up_to_the_budget(self):
        text = "\n\n".join(f"Paragraph number {n} about leave." for n in range(10))
        chunks = TextChunker(ChunkingConfig(max_tokens=20, overlap_tokens=0)).split(
            _document("d", text)
        )
        assert len(chunks) > 1
        assert all(chunk.token_count <= 20 for chunk in chunks)

    def test_no_chunk_exceeds_the_budget_even_without_boundaries(self):
        chunker = TextChunker(ChunkingConfig(max_tokens=16, overlap_tokens=0))
        chunks = chunker.split(_document("d", "word " * 200))
        assert chunks
        assert all(chunk.token_count <= 16 for chunk in chunks)

    def test_overlap_repeats_the_previous_tail(self):
        text = "\n\n".join(f"Sentence {n} about holiday policy." for n in range(8))
        plain = TextChunker(ChunkingConfig(max_tokens=24, overlap_tokens=0)).split(
            _document("d", text)
        )
        overlapped = TextChunker(ChunkingConfig(max_tokens=24, overlap_tokens=8)).split(
            _document("d", text)
        )
        assert len(plain) == len(overlapped)
        assert overlapped[1].token_count > plain[1].token_count

    def test_metadata_travels_onto_every_chunk(self):
        chunks = TextChunker(ChunkingConfig(max_tokens=16, overlap_tokens=0)).split(
            _document("d", "word " * 100, source="handbook.pdf")
        )
        assert all(chunk.metadata["source"] == "handbook.pdf" for chunk in chunks)

    def test_empty_text_produces_nothing(self):
        assert TextChunker().split(_document("d", "   \n\n  ")) == []

    def test_overlap_must_be_smaller_than_the_budget(self):
        with pytest.raises(ValueError):
            TextChunker(ChunkingConfig(max_tokens=10, overlap_tokens=10))


class TestIngestion:
    async def test_it_creates_the_collection_and_writes_chunks(
        self, ingestion, vectors
    ):
        result = await ingestion.aingest(
            [_document("leave-policy", "How to request leave. " * 20)]
        )

        assert result.documents == 1
        assert result.chunks > 1
        assert result.collection == "handbook"
        assert vectors.count("handbook") == result.chunks
        assert vectors.vector_size == len(FakeEmbeddings.VOCABULARY)

    async def test_every_chunk_keeps_its_text_and_provenance(
        self, ingestion, vectors
    ):
        await ingestion.aingest(
            [_document("leave-policy", "Leave is requested in advance.", team="ops")]
        )

        record = next(iter(vectors.collections["handbook"].values()))
        assert record.metadata["document_id"] == "leave-policy"
        assert record.metadata["chunk_index"] == 0
        assert "Leave" in record.metadata["text"]
        assert record.metadata["team"] == "ops"

    async def test_ingesting_nothing_is_not_an_error(self, ingestion):
        result = await ingestion.aingest([])
        assert result.chunks == 0

    async def test_an_empty_document_is_skipped(self, ingestion):
        result = await ingestion.aingest([_document("blank", "   ")])
        assert result.documents == 0
        assert result.chunks == 0

    async def test_reingesting_replaces_rather_than_duplicates(
        self, ingestion, vectors
    ):
        await ingestion.aingest([_document("policy", "Leave policy. " * 20)])
        first = vectors.count("handbook")

        await ingestion.aingest([_document("policy", "Leave policy. " * 20)])

        assert vectors.count("handbook") == first

    async def test_a_shorter_revision_drops_the_stale_chunks(
        self, ingestion, vectors
    ):
        await ingestion.aingest([_document("policy", "Leave policy. " * 40)])
        assert vectors.count("handbook") > 1

        await ingestion.aingest([_document("policy", "Leave policy.")])

        assert vectors.count("handbook") == 1

    async def test_deleting_a_document_removes_every_chunk(self, ingestion, vectors):
        await ingestion.aingest(
            [
                _document("keep", "Expenses are reimbursed. " * 10),
                _document("drop", "Laptop policy. " * 10),
            ]
        )

        removed = await ingestion.adelete_document("drop")

        assert removed > 0
        remaining = {
            record.metadata["document_id"]
            for record in vectors.collections["handbook"].values()
        }
        assert remaining == {"keep"}

    async def test_deleting_an_unknown_document_is_a_no_op(self, ingestion):
        assert await ingestion.adelete_document("never-ingested") == 0


class TestRetrieval:
    async def test_it_finds_the_relevant_document(self, ingestion, retrieval):
        await ingestion.aingest(
            [
                _document("leave", "Leave leave leave requests go to your manager."),
                _document("laptop", "Laptop laptop laptop replacements go to IT."),
            ]
        )

        results = await retrieval.asearch("leave", limit=1)

        assert len(results) == 1
        assert results[0].document_id == "leave"
        assert results[0].score > 0

    async def test_a_result_carries_its_text_and_position(self, ingestion, retrieval):
        await ingestion.aingest([_document("leave", "Leave requests go to a manager.")])

        result = (await retrieval.asearch("leave"))[0]

        assert "manager" in result.text
        assert result.document_id == "leave"
        assert result.chunk_index == 0

    async def test_filters_narrow_the_search(self, ingestion, retrieval):
        await ingestion.aingest(
            [
                _document("a", "Leave policy for ops.", team="ops"),
                _document("b", "Leave policy for sales.", team="sales"),
            ]
        )

        results = await retrieval.asearch("leave", filters={"team": "sales"})

        assert {r.document_id for r in results} == {"b"}

    async def test_a_blank_query_searches_for_nothing(self, retrieval):
        assert await retrieval.asearch("   ") == []

    async def test_context_joins_the_passages(self, ingestion, retrieval):
        await ingestion.aingest(
            [
                _document("a", "Leave is approved by your manager."),
                _document("b", "Leave carries over each year."),
            ]
        )

        context = await retrieval.acontext("leave", limit=2)

        assert "manager" in context
        assert "carries over" in context
        assert "---" in context
