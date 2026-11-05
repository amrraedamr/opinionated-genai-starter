from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsAzureBlobConfig(Protocol):
    """Satisfied by `AzureBlobSettingsMixin`."""

    azure_blob_account_url: str
