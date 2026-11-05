from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsAzureSearchConfig(Protocol):
    """Satisfied by `AzureSearchSettingsMixin`."""

    azure_search_endpoint: str
    azure_search_vector_field: str
    azure_search_key_field: str
    azure_search_api_key: str
