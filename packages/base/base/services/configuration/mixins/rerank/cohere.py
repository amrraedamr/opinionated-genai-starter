from pydantic import BaseModel


class CohereSettingsMixin(BaseModel):
    """Cohere, for reranking. Requires the `rerank` extra."""

    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v4.0-fast"
