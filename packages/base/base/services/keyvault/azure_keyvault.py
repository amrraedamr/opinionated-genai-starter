import asyncio
from collections.abc import AsyncIterator, Iterator

from base.services.configuration import settings_for
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

from base.models.view_models.keyvault import SecretInfo
from base.services.common.meta import SingletonABC
from base.services.configuration.protocols import SupportsKeyVaultConfig
from base.services.error_handling import aretrying, retrying
from base.services.error_handling.keyvault.azure import AzureKeyVaultErrorHandler
from base.services.keyvault import AbstractKeyVaultService
from base.tools.resources import aclose_all, close_all


class AzureKeyVaultService(AbstractKeyVaultService, metaclass=SingletonABC):
    """Azure Key Vault client for runtime secret get/set/list.

    This is an application service, not the config system. App settings live in
    `settings` (env / `.env`). Use this when product code needs vault secrets at
    runtime (customer keys, per-tenant secrets, etc.).

    Wiring Key Vault into Settings later (optional):
      - Build a `PydanticBaseSettingsSource` that calls `alist_secrets` /
        `aget_secrets`, strips the `system_keyvault_tag` prefix, and returns
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
        self._error_handler: AzureKeyVaultErrorHandler = AzureKeyVaultErrorHandler()

    @property
    @retrying
    def client(self) -> SecretClient:
        if self._client is None:
            self._credential = DefaultAzureCredential()
            self._client = SecretClient(
                vault_url=settings_for(SupportsKeyVaultConfig).keyvault_url,
                credential=self._credential,
            )
        return self._client

    @property
    @retrying
    def async_client(self) -> AsyncSecretClient:
        if self._async_client is None:
            self._async_credential = AsyncDefaultAzureCredential()
            self._async_client = AsyncSecretClient(
                vault_url=settings_for(SupportsKeyVaultConfig).keyvault_url,
                credential=self._async_credential,
            )
        return self._async_client

    def close(self) -> None:
        close_all(self._client, self._credential)
        self._client = None
        self._credential = None

    async def aclose(self) -> None:
        await aclose_all(self._async_client, self._async_credential)
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

    @retrying
    def get_secret(self, secret_name: str) -> str | None:
        formatted: str = self.format_secret_name(secret_name)
        return self.client.get_secret(formatted).value

    @aretrying
    async def aget_secret(self, secret_name: str) -> str | None:
        formatted: str = self.format_secret_name(secret_name)
        secret = await self.async_client.get_secret(formatted)
        return secret.value

    async def aget_secrets(
        self, secret_names: list[str], concurrent: bool = True
    ) -> dict[str, str | None]:
        if not concurrent:
            return {name: await self.aget_secret(name) for name in secret_names}

        results: list[str | None] = await asyncio.gather(
            *(self.aget_secret(name) for name in secret_names)
        )
        return dict(zip(secret_names, results, strict=True))

    @retrying
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

    @aretrying
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

    @retrying
    def remove_secret(self, secret_name: str) -> None:
        formatted: str = self.format_secret_name(secret_name)
        self.client.begin_delete_secret(formatted)

    @aretrying
    async def aremove_secret(self, secret_name: str) -> None:
        formatted: str = self.format_secret_name(secret_name)
        await self.async_client.delete_secret(formatted)

    def list_secrets(self) -> Iterator[SecretInfo]:
        return (self._as_info(p) for p in self.client.list_properties_of_secrets())

    def alist_secrets(self) -> AsyncIterator[SecretInfo]:
        return (
            self._as_info(p)
            async for p in self.async_client.list_properties_of_secrets()
        )

    @staticmethod
    def _as_info(properties: SecretProperties) -> SecretInfo:
        """Azure's secret properties, as the neutral shape."""
        return SecretInfo(
            name=properties.name,
            version=properties.version,
            enabled=properties.enabled if properties.enabled is not None else True,
            created_at=properties.created_on,
            updated_at=properties.updated_on,
        )

    # Naming

    def format_secret_name(
        self, secret_name: str, allowed_prefix: str | None = None
    ) -> str:
        """Azure-specific secret name formatting with keyvault tag prefixes."""
        prefixes: list[str] = [
            settings_for(SupportsKeyVaultConfig).system_keyvault_tag,
            settings_for(SupportsKeyVaultConfig).model_keyvault_tag,
        ]

        for prefix in prefixes:
            if secret_name.startswith(prefix):
                if allowed_prefix and not secret_name.startswith(allowed_prefix):
                    raise ValueError(
                        f"Secret name '{secret_name}' does not start with "
                        f"the allowed prefix '{allowed_prefix}'."
                    )
                return secret_name

        return f"{settings_for(SupportsKeyVaultConfig).system_keyvault_tag}-{secret_name}"

    def get_service_name(self) -> str:
        return "Azure Key Vault"
