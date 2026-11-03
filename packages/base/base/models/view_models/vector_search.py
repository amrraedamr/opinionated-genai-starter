from typing import Any

from pydantic import Field

from base.models.view_models import BaseModel


class VectorRecord(BaseModel):
    """A vector record for storage and retrieval."""

    id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(BaseModel):
    """A scored result from a vector similarity search."""

    id: str
    score: float
    vector: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
