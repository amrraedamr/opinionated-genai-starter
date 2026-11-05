"""What a secret store does, in terms no vendor owns.

A vault holds named secrets and hands them back. Azure Key Vault, AWS Secrets
Manager and HashiCorp Vault differ in how a secret is addressed and what
metadata travels with it, not in what a caller wants from one.

Naming rules stay with the adapter. A vault decides what a legal secret name
looks like, and this starter's tag prefixes are applied where that rule lives.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import final

from base.models.view_models.keyvault import SecretInfo


class AbstractKeyVaultService(ABC):
    """Runtime secret storage.

    This is an application service, not the configuration system. Settings
    come from the environment; this is for secrets product code reaches for
    while running — a customer's key, a per-tenant credential.

    Reads and removals are safe to repeat, so an implementation puts them
    behind its retry policy. Writing a secret creates a new version, so an
    adapter decides whether repeating one is safe for its vault.
    """

    @abstractmethod
    def get_service_name(self) -> str: ...

    @abstractmethod
    def get_secret(self, secret_name: str) -> str | None:
        """The secret's value, or None when the vault has no such secret."""

    @abstractmethod
    async def aget_secret(self, secret_name: str) -> str | None:
        """The secret's value, or None when the vault has no such secret."""

    @final
    def get_secrets(self, secret_names: list[str]) -> dict[str, str | None]:
        """Several secrets by name, each fetched on its own.

        Fetching them one at a time is what keeps each name's retries its own.
        """
        return {name: self.get_secret(name) for name in secret_names}

    @abstractmethod
    async def aget_secrets(
        self, secret_names: list[str], concurrent: bool = True
    ) -> dict[str, str | None]:
        """Several secrets by name, concurrently unless asked otherwise.

        Whether concurrency is worth it depends on the client, so the adapter
        decides how to gather them.
        """

    @abstractmethod
    def upsert_secret(self, secret_key: str, secret_value: str) -> str:
        """Store a value and return the version the vault assigned it."""

    @abstractmethod
    async def aupsert_secret(self, secret_key: str, secret_value: str) -> str:
        """Store a value and return the version the vault assigned it."""

    @abstractmethod
    def remove_secret(self, secret_name: str) -> None:
        """Delete a secret."""

    @abstractmethod
    async def aremove_secret(self, secret_name: str) -> None:
        """Delete a secret."""

    @abstractmethod
    def list_secrets(self) -> Iterator[SecretInfo]:
        """Every secret the vault holds, without their values."""

    @abstractmethod
    def alist_secrets(self) -> AsyncIterator[SecretInfo]:
        """Every secret the vault holds, without their values."""
