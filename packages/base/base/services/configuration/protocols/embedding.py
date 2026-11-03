from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsEmbeddingConfig(Protocol):
    """Satisfied by `EmbeddingSettingsMixin`."""

    azure_openai_api_version: str
    azure_openai_embedding_endpoint: str
    azure_openai_embedding_deployment_name: str
    azure_openai_embedding_api_key: str
    azure_openai_embedding_dimensions: int
    azure_openai_embedding_batch_size: int

    openai_embedding_endpoint: str
    openai_embedding_model: str
    openai_embedding_api_key: str
    openai_embedding_dimensions: int
    openai_embedding_batch_size: int
