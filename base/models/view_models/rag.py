"""Canonical shapes for retrieval-augmented generation."""

from typing import Any

from pydantic import Field

from base.models.view_models import BaseModel


class Document(BaseModel):
    """A source document, before it is split for indexing."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Travels onto every chunk, so a result can be traced back to its source."""


class Chunk(BaseModel):
    """One indexable slice of a document."""

    text: str
    index: int
    """Position within the document, counting from zero."""

    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    def vector_id(self, document_id: str) -> str:
        """Stable id for this chunk, so re-ingesting a document overwrites it."""
        return f"{document_id}:{self.index}"


class RetrievedChunk(BaseModel):
    """A chunk that came back from a search, with the score that found it."""

    text: str
    score: float
    document_id: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """What an ingestion run wrote."""

    documents: int
    chunks: int
    collection: str
