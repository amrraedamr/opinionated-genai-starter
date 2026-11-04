from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsAzureOpenAIConfig(Protocol):
    """Satisfied by `AzureOpenAISettingsMixin`."""

    azure_openai_api_version: str


@runtime_checkable
class SupportsAzureOpenAILLMConfig(SupportsAzureOpenAIConfig, Protocol):
    """Satisfied by `AzureOpenAILLMSettingsMixin`."""

    azure_openai_llm_endpoint: str
    azure_openai_llm_deployment_name: str
    azure_openai_llm_api_key: str


@runtime_checkable
class SupportsAzureOpenAIEmbeddingConfig(SupportsAzureOpenAIConfig, Protocol):
    """Satisfied by `AzureOpenAIEmbeddingSettingsMixin`."""

    azure_openai_embedding_endpoint: str
    azure_openai_embedding_deployment_name: str
    azure_openai_embedding_api_key: str
    azure_openai_embedding_dimensions: int
    azure_openai_embedding_batch_size: int
