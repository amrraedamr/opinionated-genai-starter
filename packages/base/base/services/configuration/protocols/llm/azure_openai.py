from typing import Protocol, runtime_checkable

from base.services.configuration.protocols.azure_openai import (
    SupportsAzureOpenAIConfig,
)


@runtime_checkable
class SupportsAzureOpenAILLMConfig(SupportsAzureOpenAIConfig, Protocol):
    """Satisfied by `AzureOpenAILLMSettingsMixin`."""

    azure_openai_llm_endpoint: str
    azure_openai_llm_deployment_name: str
    azure_openai_llm_api_key: str
