"""`RagIngestionService` is the pipeline from a document to rows in a collection.

Re-ingesting replaces rather than appends, which is the behaviour worth
pinning: chunk ids are derived from the document id, so a shorter second
version has to take its leftovers with it or they come back as stale context.
"""

from __future__ import annotations

from base.models.view_models.rag import ChunkingConfig, Document, IngestionConfig
from base.services.rag.ingestion import (
    CHUNK_INDEX_FIELD,
    DOCUMENT_ID_FIELD,
    TEXT_FIELD,
    RagIngestionService,
)
from tests.unit.rag.fakes import FakeEmbeddings, FakeVectorStore

COLLECTION = "handbook"


def service(
    store: FakeVectorStore,
    embeddings: FakeEmbeddings,
    *,
    max_tokens: int = 200,
    distance: str = "cosine",
) -> RagIngestionService:
    return RagIngestionService(
        store,
        IngestionConfig(
            collection=COLLECTION,
            distance=distance,
            chunking=ChunkingConfig(max_tokens=max_tokens, overlap_tokens=0),
        ),
        embeddings,
    )


def paragraphs(count: int) -> str:
    return "\n\n".join(f"Paragraph {n} of the handbook." for n in range(count))


async def test_ingesting_nothing_touches_no_backend() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()

    result = await service(store, embeddings).aingest([])

    assert (result.documents, result.chunks, result.collection) == (0, 0, COLLECTION)
    assert embeddings.batches == []
    assert store.collections == {}


async def test_documents_with_no_text_are_ingested_as_nothing() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()

    result = await service(store, embeddings).aingest(
        [Document(id="a", text=""), Document(id="b", text="   ")]
    )

    assert (result.documents, result.chunks) == (0, 0)
    assert embeddings.batches == []
    assert store.collections == {}


async def test_one_document_becomes_one_record_per_chunk() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()

    result = await service(store, embeddings, max_tokens=12).aingest(
        [Document(id="onboarding", text=paragraphs(3))]
    )

    assert result.documents == 1
    assert result.chunks == 3
    assert sorted(store.collections[COLLECTION]) == [
        "onboarding:0",
        "onboarding:1",
        "onboarding:2",
    ]


async def test_the_collection_is_created_at_the_width_of_the_first_vector() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings(width=5)

    await service(store, embeddings, distance="dot").aingest(
        [Document(id="a", text="Some text.")]
    )

    assert store.created == [(COLLECTION, 5, "dot")]


async def test_an_existing_collection_is_not_created_again() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()
    await store.acreate_collection(COLLECTION, vector_size=3)
    store.created.clear()

    await service(store, embeddings).aingest([Document(id="a", text="Some text.")])

    assert store.created == []


async def test_every_chunk_is_embedded_once_in_document_order() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()

    await service(store, embeddings, max_tokens=12).aingest(
        [
            Document(id="first", text=paragraphs(2)),
            Document(id="second", text=paragraphs(1)),
        ]
    )

    assert embeddings.embedded == [
        "Paragraph 0 of the handbook.",
        "Paragraph 1 of the handbook.",
        "Paragraph 0 of the handbook.",
    ]


async def test_embedding_is_batched_at_the_providers_ceiling() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings(batch_size=2)

    await service(store, embeddings, max_tokens=12).aingest(
        [Document(id="a", text=paragraphs(5))]
    )

    assert [len(batch) for batch in embeddings.batches] == [2, 2, 1]


async def test_a_record_carries_the_text_its_source_and_its_position() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()

    await service(store, embeddings, max_tokens=12).aingest(
        [Document(id="handbook", text=paragraphs(2), metadata={"team": "people"})]
    )

    record = store.collections[COLLECTION]["handbook:1"]
    assert record.metadata[TEXT_FIELD] == "Paragraph 1 of the handbook."
    assert record.metadata[DOCUMENT_ID_FIELD] == "handbook"
    assert record.metadata[CHUNK_INDEX_FIELD] == 1
    assert record.metadata["team"] == "people"


async def test_each_record_carries_the_vector_for_its_own_chunk() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()

    await service(store, embeddings, max_tokens=12).aingest(
        [Document(id="a", text=paragraphs(3))]
    )

    records = store.collections[COLLECTION]
    assert len(records) == 3
    assert all(
        record.vector == embeddings._vector(record.metadata[TEXT_FIELD])
        for record in records.values()
    )
    assert records["a:0"].vector != records["a:1"].vector


async def test_re_ingesting_a_shorter_document_removes_its_leftovers() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()
    pipeline = service(store, embeddings, max_tokens=12)
    await pipeline.aingest([Document(id="a", text=paragraphs(3))])

    result = await pipeline.aingest([Document(id="a", text=paragraphs(1))])

    assert result.chunks == 1
    assert sorted(store.collections[COLLECTION]) == ["a:0"]


async def test_re_ingesting_leaves_another_documents_chunks_alone() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()
    pipeline = service(store, embeddings, max_tokens=12)
    await pipeline.aingest(
        [Document(id="a", text=paragraphs(3)), Document(id="b", text=paragraphs(2))]
    )

    await pipeline.aingest([Document(id="a", text=paragraphs(1))])

    assert sorted(store.collections[COLLECTION]) == ["a:0", "b:0", "b:1"]


async def test_re_ingesting_the_same_length_overwrites_in_place() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()
    pipeline = service(store, embeddings, max_tokens=12)
    await pipeline.aingest([Document(id="a", text="First version.")])

    await pipeline.aingest([Document(id="a", text="Second version, longer.")])

    assert list(store.collections[COLLECTION]) == ["a:0"]
    assert (
        store.collections[COLLECTION]["a:0"].metadata[TEXT_FIELD]
        == "Second version, longer."
    )


async def test_stale_chunks_are_scrolled_for_by_document_id() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()
    pipeline = service(store, embeddings, max_tokens=12)
    await pipeline.aingest(
        [Document(id="a", text=paragraphs(2)), Document(id="b", text=paragraphs(2))]
    )

    await pipeline.aingest([Document(id="a", text=paragraphs(1))])

    assert ("handbook", ["a:1"]) in store.deleted


async def test_deleting_a_document_removes_every_chunk_and_counts_them() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()
    pipeline = service(store, embeddings, max_tokens=12)
    await pipeline.aingest([Document(id="a", text=paragraphs(3))])

    removed = await pipeline.adelete_document("a")

    assert removed == 3
    assert store.collections[COLLECTION] == {}


async def test_deleting_an_unknown_document_removes_nothing() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()
    pipeline = service(store, embeddings)
    await pipeline.aingest([Document(id="a", text="text")])
    store.deleted.clear()

    assert await pipeline.adelete_document("nobody") == 0
    assert store.deleted == []


async def test_deleting_before_the_collection_exists_is_not_an_error() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()

    assert await service(store, embeddings).adelete_document("a") == 0


async def test_a_mix_of_empty_and_real_documents_ingests_only_the_real_ones() -> None:
    store, embeddings = FakeVectorStore(), FakeEmbeddings()

    result = await service(store, embeddings, max_tokens=12).aingest(
        [
            Document(id="empty", text=""),
            Document(id="real", text=paragraphs(2)),
        ]
    )

    assert result.documents == 1
    assert result.chunks == 2
    assert sorted(store.collections[COLLECTION]) == ["real:0", "real:1"]
