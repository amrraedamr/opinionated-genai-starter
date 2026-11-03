import asyncio
from collections.abc import AsyncIterator, Iterator

from base.tools.errors import MissingExtraError

try:
    from azure.identity import DefaultAzureCredential
    from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
    from azure.keyvault.secrets import SecretClient, SecretProperties
    from azure.keyvault.secrets.aio import SecretClient as AsyncSecretClient
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "azure", "azure-keyvault-secrets", feature="AzureKeyVaultService"
    ) from exc
from typing_extensions import Self

from base.services.common.meta import Singleton
from base.services.configuration import get_settings
from base.tools import resources


class AzureKeyVaultService(metaclass=Singleton):
    """Azure Key Vault client for runtime secret get/set/list.

    This is an application service, not the config system. App settings live in
    `settings` (env / `.env`). Use this when product code needs vault secrets at
    runtime (customer keys, per-tenant secrets, etc.).

    Wiring Key Vault into Settings later (optional):
      - Build a `PydanticBaseSettingsSource` that calls `alist_secrets` /
        `aget_secrets`, strips `get_settings().system_keyvault_tag` + `-`, and returns
        a dict of Settings field names → values.
      - Register it in `Settings.settings_customise_sources`.
      - For multi-instance hot reload: after rotating secrets, publish
        `CacheChannels.KEYVAULT_RELOAD`; each process pops/rebuilds the
        `Settings` singleton (see `Settings` docstring). `format_secret_name`
        below is the naming convention that source should share.
    """

    def __init__(self) -> None:
        self._client: SecretClient | None = None
        self._async_client: AsyncSecretClient | None = None
        self._credential: DefaultAzureCredential | None = None
        self._async_credential: AsyncDefaultAzureCredential | None = None

    # Client lifecycle

    @property
    def client(self) -> SecretClient:
        if self._client is None:
            self._credential = DefaultAzureCredential()
            self._client = SecretClient(
                vault_url=get_settings().keyvault_url,
                credential=self._credential,
            )
        return self._client

    @property
    def async_client(self) -> AsyncSecretClient:
        if self._async_client is None:
            self._async_credential = AsyncDefaultAzureCredential()
            self._async_client = AsyncSecretClient(
                vault_url=get_settings().keyvault_url,
                credential=self._async_credential,
            )
        return self._async_client

    def close(self) -> None:
        resources.close_all(self._client, self._credential)
        self._client = None
        self._credential = None

    async def aclose(self) -> None:
        await resources.aclose_all(self._async_client, self._async_credential)
        self._async_client = None
        self._async_credential = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    # Secret operations

    def get_secret(self, secret_name: str) -> str | None:
        formatted: str = self.format_secret_name(secret_name)
        return self.client.get_secret(formatted).value

    async def aget_secret(self, secret_name: str) -> str | None:
        formatted: str = self.format_secret_name(secret_name)
        secret = await self.async_client.get_secret(formatted)
        return secret.value

    def get_secrets(self, secret_names: list[str]) -> dict[str, str | None]:
        return {name: self.get_secret(name) for name in secret_names}

    async def aget_secrets(
        self, secret_names: list[str], concurrent: bool = True
    ) -> dict[str, str | None]:
        if not concurrent:
            return {name: await self.aget_secret(name) for name in secret_names}

        results: list[str | None] = await asyncio.gather(
            *(self.aget_secret(name) for name in secret_names)
        )
        return dict(zip(secret_names, results, strict=True))

    def upsert_secret(
        self, secret_key: str, secret_value: str, format_name: bool = True
    ) -> str:
        if format_name:
            secret_key = self.format_secret_name(secret_key)
        secret = self.client.set_secret(secret_key, secret_value)
        assert secret.properties.version is not None, (
            "Azure returned no version after upsert"
        )
        return secret.properties.version

    async def aupsert_secret(
        self, secret_key: str, secret_value: str, format_name: bool = True
    ) -> str:
        if format_name:
            secret_key = self.format_secret_name(secret_key)
        secret = await self.async_client.set_secret(secret_key, secret_value)
        assert secret.properties.version is not None, (
            "Azure returned no version after upsert"
        )
        return secret.properties.version

    def remove_secret(self, secret_name: str) -> None:
        formatted: str = self.format_secret_name(secret_name)
        self.client.begin_delete_secret(formatted)

    async def aremove_secret(self, secret_name: str) -> None:
        formatted: str = self.format_secret_name(secret_name)
        await self.async_client.delete_secret(formatted)

    def list_secrets(self) -> Iterator[SecretProperties]:
        return self.client.list_properties_of_secrets()

    def alist_secrets(self) -> AsyncIterator[SecretProperties]:
        return self.async_client.list_properties_of_secrets()

    # Naming

    def format_secret_name(
        self, secret_name: str, allowed_prefix: str | None = None
    ) -> str:
        """Azure-specific secret name formatting with keyvault tag prefixes."""
        prefixes: list[str] = [
            get_settings().system_keyvault_tag,
            get_settings().model_keyvault_tag,
        ]

        for prefix in prefixes:
            if secret_name.startswith(prefix):
                if allowed_prefix and not secret_name.startswith(allowed_prefix):
                    raise ValueError(
                        f"Secret name '{secret_name}' does not start with "
                        f"the allowed prefix '{allowed_prefix}'."
                    )
                return secret_name

        return f"{get_settings().system_keyvault_tag}-{secret_name}"

    def get_service_name(self) -> str:
        return "Azure Key Vault"
