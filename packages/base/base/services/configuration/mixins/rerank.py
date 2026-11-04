from pydantic import BaseModel


class RerankSettingsMixin(BaseModel):
    """Reranking knobs that hold whichever provider is behind them."""

    rerank_overfetch: int = 4
    """How many times `limit` to retrieve before reranking. A cross-encoder
    can only reorder what retrieval already returned, so it needs a wider
    candidate set than the caller asked for."""
