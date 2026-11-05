from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsOpenAIEmbeddingConfig(Protocol):
    """Satisfied by `OpenAIEmbeddingSettingsMixin`."""

    openai_embedding_endpoint: str
    openai_embedding_model: str
    openai_embedding_api_key: str
    openai_embedding_dimensions: int
    openai_embedding_batch_size: int
