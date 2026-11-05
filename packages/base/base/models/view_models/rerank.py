"""What a reranker says about the passages it was given."""

from base.models.view_models import BaseModel
from base.models.view_models.usage import ProviderUsage


class RerankResult(BaseModel):
    """A reranked passage, carrying its position in the original list."""

    index: int
    """Position in the list handed to the reranker, so callers can map the
    result back to whatever the text came from."""

    score: float
    document: str


class RerankBatch(BaseModel):
    """One request's reordered passages, and what it cost.

    Usage travels with the results so the abstraction can record the call
    without each adapter remembering to.
    """

    results: list[RerankResult]
    usage: ProviderUsage | None = None
