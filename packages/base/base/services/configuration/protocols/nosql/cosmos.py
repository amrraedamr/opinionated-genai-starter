from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsCosmosConfig(Protocol):
    """Satisfied by `CosmosSettingsMixin`."""

    azure_cosmos_account_uri: str
    azure_cosmos_account_key: str
    azure_cosmos_database_name: str
