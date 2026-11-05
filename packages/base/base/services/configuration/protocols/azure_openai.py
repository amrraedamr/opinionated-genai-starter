from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsAzureOpenAIConfig(Protocol):
    """Satisfied by `AzureOpenAISettingsMixin`."""

    azure_openai_api_version: str
