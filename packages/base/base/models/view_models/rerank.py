"""What a reranker says about one passage."""

from base.models.view_models import BaseModel


class RerankResult(BaseModel):
    """A reranked passage, carrying its position in the original list."""

    index: int
    """Position in the list handed to the reranker, so callers can map the
    result back to whatever the text came from."""

    score: float
    document: str
