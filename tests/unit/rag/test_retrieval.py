"""`RagRetrievalService` turns a question into passages, not into ids.

The text is stored beside the vector at ingestion time, so a search answers
with the passage. What is worth pinning is the over-fetch: it only applies
when something downstream can improve on vector order, because fetching five
times as much for nothing is paid for on every query.
"""

from __future__ import annotations

from typing import Any

import pytest

from base.models.view_models.vector_search import VectorSearchResult
from base.services.rag.ingestion import (
    CHUNK_INDEX_FIELD,
    DOCUMENT_ID_FIELD,
    TEXT_FIELD,
)
from base.services.rag.retrieval import RagRetrievalService
from tests.conftest import MakeSettings
from tests.unit.rag.fakes import (
    FakeEmbeddings,
    FakeVectorStore,
    LexicalVectorStore,
    ReversingReranker,
)

COLLECTION = "handbook"


def hit(identifier: str, text: str, document: str = "doc", index: int = 0) -> Any:
    from base.models.view_models.vector_search import VectorRecord

    return VectorRecord(
        id=identifier,
        vector=[0.0, 0.0, 0.0],
        metadata={
            TEXT_FIELD: text,
            DOCUMENT_ID_FIELD: document,
            CHUNK_INDEX_FIELD: index,
        },
    )


def populated(count: int = 10) -> FakeVectorStore:
    store = FakeVectorStore()
    store.upsert(
        COLLECTION,
        [hit(f"doc:{n}", f"Passage {n}.", index=n) for n in range(count)],
    )
    return store


def retrieval(
    store: FakeVectorStore,
    *,
    reranker: ReversingReranker | None = None,
    overfetch: int | None = 3,
) -> RagRetrievalService:
    return RagRetrievalService(
        store,
        COLLECTION,
        reranker=reranker,
        overfetch=overfetch,
        embedding_service=FakeEmbeddings(),
    )


async def test_a_blank_query_never_reaches_a_backend() -> None:
    store = populated()

    assert await retrieval(store).asearch("   ") == []
    assert store.searches == []


async def test_results_come_back_as_passages_with_their_provenance() -> None:
    chunks = await retrieval(populated(3)).asearch("anything", limit=3)

    assert [c.text for c in chunks] == ["Passage 0.", "Passage 1.", "Passage 2."]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.document_id == "doc" for c in chunks)
    assert chunks[0].score > chunks[1].score


async def test_the_query_is_embedded_and_the_vector_is_what_is_searched() -> None:
    store = populated(3)
    embeddings = FakeEmbeddings()
    service = RagRetrievalService(
        store, COLLECTION, overfetch=1, embedding_service=embeddings
    )

    await service.asearch("how do I request leave?")

    assert embeddings.embedded == ["how do I request leave?"]
    assert store.searches[0]["vector"] == embeddings._vector("how do I request leave?")
    assert store.searches[0]["collection"] == COLLECTION


async def test_without_a_reranker_or_lexical_backend_only_the_limit_is_fetched() -> (
    None
):
    store = populated()

    await retrieval(store, overfetch=5).asearch("anything", limit=2)

    assert store.searches[0]["limit"] == 2


async def test_a_reranker_makes_the_search_over_fetch() -> None:
    store = populated()

    await retrieval(store, reranker=ReversingReranker(), overfetch=4).asearch(
        "anything", limit=2
    )

    assert store.searches[0]["limit"] == 8


async def test_a_lexical_backend_makes_the_search_over_fetch() -> None:
    store = LexicalVectorStore()
    store.upsert(COLLECTION, [hit("doc:0", "Passage 0.")])
    service = RagRetrievalService(
        store, COLLECTION, overfetch=3, embedding_service=FakeEmbeddings()
    )

    await service.asearch("anything", limit=2)

    assert store.searches[0]["limit"] == 6


async def test_filters_and_a_threshold_are_passed_through_to_the_backend() -> None:
    store = populated(3)

    await retrieval(store, overfetch=1).asearch(
        "anything", limit=2, filters={"team": "people"}, score_threshold=0.4
    )

    assert store.searches[0]["filters"] == {"team": "people"}
    assert store.searches[0]["score_threshold"] == 0.4


async def test_a_reranker_reorders_the_results_and_restates_their_scores() -> None:
    reranker = ReversingReranker()

    chunks = await retrieval(populated(4), reranker=reranker, overfetch=1).asearch(
        "anything", limit=3
    )

    assert [c.text for c in chunks] == ["Passage 2.", "Passage 1.", "Passage 0."]
    assert chunks[0].score == 1.0
    assert reranker.calls[0][0] == "anything"
    assert reranker.calls[0][2] == 3


async def test_a_reranker_that_says_nothing_leaves_the_vector_order_alone() -> None:
    class SilentReranker(ReversingReranker):
        def _rerank(self, query: str, documents: list[str], top_n: int | None) -> Any:
            from base.models.view_models.rerank import RerankBatch

            return RerankBatch(results=[])

    chunks = await retrieval(
        populated(4), reranker=SilentReranker(), overfetch=1
    ).asearch("anything", limit=2)

    assert [c.text for c in chunks] == ["Passage 0.", "Passage 1."]


async def test_lexical_hits_are_fused_with_dense_ones_by_rank() -> None:
    store = LexicalVectorStore(
        lexical=[
            VectorSearchResult(
                id="doc:2",
                score=99.0,
                metadata={
                    TEXT_FIELD: "Passage 2.",
                    DOCUMENT_ID_FIELD: "doc",
                    CHUNK_INDEX_FIELD: 2,
                },
            )
        ]
    )
    store.upsert(
        COLLECTION, [hit(f"doc:{n}", f"Passage {n}.", index=n) for n in range(3)]
    )
    service = RagRetrievalService(
        store, COLLECTION, overfetch=1, embedding_service=FakeEmbeddings()
    )

    chunks = await service.asearch("leave policy", limit=3)

    assert store.lexical_queries == ["leave policy"]
    assert [c.text for c in chunks][0] == "Passage 2."
    assert chunks[0].score < 1.0


async def test_a_lexical_backend_with_no_hits_falls_back_to_dense_order() -> None:
    store = LexicalVectorStore(lexical=[])
    store.upsert(
        COLLECTION, [hit(f"doc:{n}", f"Passage {n}.", index=n) for n in range(3)]
    )
    service = RagRetrievalService(
        store, COLLECTION, overfetch=1, embedding_service=FakeEmbeddings()
    )

    chunks = await service.asearch("anything", limit=3)

    assert [c.text for c in chunks] == ["Passage 0.", "Passage 1.", "Passage 2."]


async def test_context_joins_the_passages_into_one_block() -> None:
    context = await retrieval(populated(3), overfetch=1).acontext("anything", limit=3)

    assert context == "Passage 0.\n\n---\n\nPassage 1.\n\n---\n\nPassage 2."


async def test_context_takes_the_separator_the_caller_wants() -> None:
    context = await retrieval(populated(2), overfetch=1).acontext(
        "anything", limit=2, separator=" | "
    )

    assert context == "Passage 0. | Passage 1."


async def test_context_from_an_empty_search_is_an_empty_string() -> None:
    assert await retrieval(FakeVectorStore(), overfetch=1).acontext("anything") == ""


async def test_a_record_with_no_stored_text_is_skipped_by_context() -> None:
    from base.models.view_models.vector_search import VectorRecord

    store = FakeVectorStore()
    store.upsert(
        COLLECTION,
        [
            VectorRecord(id="doc:0", vector=[0.0], metadata={}),
            hit("doc:1", "Real passage.", index=1),
        ],
    )

    assert await retrieval(store, overfetch=1).acontext("anything") == "Real passage."


async def test_a_missing_chunk_index_reads_back_as_minus_one() -> None:
    from base.models.view_models.vector_search import VectorRecord

    store = FakeVectorStore()
    store.upsert(
        COLLECTION,
        [VectorRecord(id="doc:0", vector=[0.0], metadata={TEXT_FIELD: "Loose."})],
    )

    chunks = await retrieval(store, overfetch=1).asearch("anything")

    assert chunks[0].chunk_index == -1
    assert chunks[0].document_id == ""


def test_the_overfetch_falls_back_to_settings_when_the_caller_names_none(
    make_settings: MakeSettings,
) -> None:
    from base.services.configuration.mixins import RedisSettingsMixin

    make_settings(RedisSettingsMixin, redis_url="redis://x:1/0")
    service = RagRetrievalService(
        FakeVectorStore(), COLLECTION, embedding_service=FakeEmbeddings()
    )

    assert service._overfetch == 4


def test_an_overfetch_below_one_is_raised_to_one() -> None:
    service = RagRetrievalService(
        FakeVectorStore(),
        COLLECTION,
        overfetch=-5,
        embedding_service=FakeEmbeddings(),
    )

    assert service._overfetch == 1


def test_retrieval_without_an_explicit_overfetch_needs_configured_settings() -> None:
    with pytest.raises(RuntimeError, match="Settings not configured"):
        RagRetrievalService(
            FakeVectorStore(), COLLECTION, embedding_service=FakeEmbeddings()
        )


def test_a_backend_without_full_text_is_not_treated_as_lexical() -> None:
    assert retrieval(FakeVectorStore())._lexical is None


def test_a_backend_with_full_text_is_recognised_structurally() -> None:
    store = LexicalVectorStore()
    assert retrieval(store)._lexical is store
