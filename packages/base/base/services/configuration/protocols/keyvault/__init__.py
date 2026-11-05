"""Secret store settings protocols, one module per backend."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsKeyVaultConfig(Protocol):
    """Satisfied by `KeyVaultSettingsMixin`."""

    keyvault_url: str
    system_keyvault_tag: str
    model_keyvault_tag: str
