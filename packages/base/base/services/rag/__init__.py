"""Retrieval-augmented generation: get documents in, get passages back out.

Ingestion chunks a document, embeds the chunks, and upserts them into a
vector collection along with the text itself. Retrieval embeds a question and
returns the nearest chunks, text included, ready to drop into a prompt.

    service = RagIngestionService(
        QdrantVectorSearchService(),
        IngestionConfig(collection="handbook"),
    )
    await service.aingest([Document(id="onboarding", text=...)])

    context = await RagRetrievalService(
        QdrantVectorSearchService(), "handbook"
    ).acontext("how do I request leave?")
"""

from base.models.view_models.rag import (
    Chunk,
    Document,
    IngestionResult,
    RetrievedChunk,
)
from base.services.rag.chunking import ChunkingConfig, TextChunker
from base.services.rag.ingestion import IngestionConfig, RagIngestionService
from base.services.rag.retrieval import RagRetrievalService

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "Document",
    "IngestionConfig",
    "IngestionResult",
    "RagIngestionService",
    "RagRetrievalService",
    "RetrievedChunk",
    "TextChunker",
]
