from typing import Protocol, runtime_checkable

from base.services.configuration.protocols.azure_openai import (
    SupportsAzureOpenAIConfig,
)


@runtime_checkable
class SupportsAzureOpenAIEmbeddingConfig(SupportsAzureOpenAIConfig, Protocol):
    """Satisfied by `AzureOpenAIEmbeddingSettingsMixin`."""

    azure_openai_embedding_endpoint: str
    azure_openai_embedding_deployment_name: str
    azure_openai_embedding_api_key: str
    azure_openai_embedding_dimensions: int
    azure_openai_embedding_batch_size: int
