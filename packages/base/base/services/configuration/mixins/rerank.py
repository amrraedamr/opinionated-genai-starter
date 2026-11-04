from pydantic import BaseModel


class RerankSettingsMixin(BaseModel):
    """Reranking knobs that hold whichever provider is behind them."""

    rerank_overfetch: int = 4
    """How many times `limit` to retrieve before reranking. A cross-encoder
    can only reorder what retrieval already returned, so it needs a wider
    candidate set than the caller asked for."""


class CohereSettingsMixin(BaseModel):
    """Cohere. Requires the `rerank` extra."""

    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v4.0-fast"
