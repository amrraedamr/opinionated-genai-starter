"""Getting documents into a vector collection, and keeping them current."""

from pydantic import Field

from base.models.view_models import BaseModel
from base.models.view_models.rag import Chunk, Document, IngestionResult
from base.models.view_models.vector_search import VectorRecord
from base.services.logging.console import ConsoleLoggerService
from base.services.rag.chunking import ChunkingConfig, TextChunker
from base.services.text_embedding.openai_embedding import OpenAITextEmbeddingService
from base.services.vector_search import AbstractVectorSearchService

DOCUMENT_ID_FIELD = "document_id"
CHUNK_INDEX_FIELD = "chunk_index"
TEXT_FIELD = "text"


class IngestionConfig(BaseModel):
    """Where documents go and how they are cut up on the way."""

    collection: str
    distance: str = "cosine"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class RagIngestionService:
    """Chunk, embed and upsert documents into a vector collection.

    The vector backend is a constructor argument rather than something built
    here: which store a collection lives in is the caller's choice, and it is
    the one thing `AbstractVectorSearchService` exists to let you swap.

    Re-ingesting a document replaces it. Chunk ids are derived from the
    document id, so the chunks that still exist are overwritten, and any left
    over from a longer previous version are deleted rather than left behind
    to be retrieved as stale context.
    """

    def __init__(
        self,
        vector_service: AbstractVectorSearchService,
        config: IngestionConfig,
    ) -> None:
        self._vector_service = vector_service
        self._config = config
        self._chunker: TextChunker = TextChunker(config.chunking)
        self._embedding_service: OpenAITextEmbeddingService = (
            OpenAITextEmbeddingService()
        )
        self._logger: ConsoleLoggerService = ConsoleLoggerService()

    async def aingest(self, documents: list[Document]) -> IngestionResult:
        """Index `documents`, replacing any earlier version of each."""
        if not documents:
            return IngestionResult(
                documents=0, chunks=0, collection=self._config.collection
            )

        chunked = [(document, self._chunker.split(document)) for document in documents]
        chunked = [(document, chunks) for document, chunks in chunked if chunks]
        if not chunked:
            self._logger.warning("Nothing to ingest: every document was empty")
            return IngestionResult(
                documents=0, chunks=0, collection=self._config.collection
            )

        texts = [chunk.text for _, chunks in chunked for chunk in chunks]
        vectors = await self._embedding_service.aembed_documents(texts)

        await self._aensure_collection(len(vectors[0]))

        records: list[VectorRecord] = []
        position = 0
        for document, chunks in chunked:
            for chunk in chunks:
                records.append(
                    self._record(document, chunk, vectors[position])
                )
                position += 1
            await self._adrop_stale_chunks(document.id, len(chunks))

        await self._vector_service.aupsert(self._config.collection, records)
        self._logger.info(
            f"Ingested {len(records)} chunks from {len(chunked)} document(s) "
            f"into {self._config.collection}"
        )
        return IngestionResult(
            documents=len(chunked),
            chunks=len(records),
            collection=self._config.collection,
        )

    async def adelete_document(self, document_id: str) -> int:
        """Remove every chunk of a document. Returns how many were removed."""
        ids = await self._achunk_ids(document_id)
        if ids:
            await self._vector_service.adelete(self._config.collection, ids)
        return len(ids)

    def _record(
        self, document: Document, chunk: Chunk, vector: list[float]
    ) -> VectorRecord:
        return VectorRecord(
            id=chunk.vector_id(document.id),
            vector=vector,
            metadata={
                **chunk.metadata,
                DOCUMENT_ID_FIELD: document.id,
                CHUNK_INDEX_FIELD: chunk.index,
                TEXT_FIELD: chunk.text,
            },
        )

    async def _aensure_collection(self, vector_size: int) -> None:
        if await self._vector_service.acollection_exists(self._config.collection):
            return
        await self._vector_service.acreate_collection(
            self._config.collection,
            vector_size=vector_size,
            distance=self._config.distance,
        )

    async def _adrop_stale_chunks(self, document_id: str, keeping: int) -> None:
        """Delete chunks left over from a longer previous version."""
        stale = [
            chunk_id
            for chunk_id in await self._achunk_ids(document_id)
            if self._index_of(chunk_id) >= keeping
        ]
        if stale:
            await self._vector_service.adelete(self._config.collection, stale)

    async def _achunk_ids(self, document_id: str) -> list[str]:
        if not await self._vector_service.acollection_exists(self._config.collection):
            return []

        ids: list[str] = []
        offset: str | None = None
        while True:
            records, offset = await self._vector_service.ascroll(
                self._config.collection,
                limit=256,
                offset=offset,
                filters={DOCUMENT_ID_FIELD: document_id},
            )
            ids.extend(record.id for record in records)
            if offset is None:
                return ids

    @staticmethod
    def _index_of(chunk_id: str) -> int:
        _, _, index = chunk_id.rpartition(":")
        return int(index) if index.isdigit() else -1
