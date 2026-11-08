from base.services.configuration import settings_for
from base.tools.errors import MissingExtraError

try:
    from azure.core.exceptions import AzureError
    from azure.identity import DefaultAzureCredential
    from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
    from azure.storage.blob import BlobServiceClient
    from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "azure", "azure-storage-blob", feature="AzureBlobClient"
    ) from exc

from base.models.view_models.exceptions.blob import BlobConfigError
from base.services.common.meta import Singleton
from base.services.configuration.protocols import SupportsAzureBlobConfig
from base.tools.resources import aclose_all, close_all


class AzureBlobClient(metaclass=Singleton):
    """Azure Blob Storage client with both sync and async support.

    Designed to be instantiated once and reused across the application.
    Lazy-initializes the underlying SDK clients on first access.

    A shared key is used when one is configured; otherwise the client falls
    back to `DefaultAzureCredential`, so a deployment holding a role
    assignment needs no key. The identity is built only where there is no key.
    """

    def __init__(self) -> None:
        settings = settings_for(SupportsAzureBlobConfig)
        self._account_url: str = settings.azure_blob_account_url
        if not self._account_url:
            raise BlobConfigError("azure_blob_account_url is required")

        self._account_key: str = settings.azure_blob_account_key

        self._sync_client: BlobServiceClient | None = None
        self._async_client: AsyncBlobServiceClient | None = None
        self._credential: DefaultAzureCredential | None = None
        self._async_credential: AsyncDefaultAzureCredential | None = None

    @property
    def account_key(self) -> str:
        """The shared key, where one is configured. Empty means an identity."""
        return self._account_key

    # Client accessors (lazy init)

    # SDK-level retries are off on both: AzureBlobErrorHandler owns retry
    # policy, and two layers of backoff would multiply the wait.

    @property
    def sync_client(self) -> BlobServiceClient:
        if self._sync_client is None:
            if not self._account_key:
                self._credential = DefaultAzureCredential()
            self._sync_client = BlobServiceClient(
                account_url=self._account_url,
                credential=self._account_key or self._credential,
                retry_total=0,
            )
        return self._sync_client

    @property
    def async_client(self) -> AsyncBlobServiceClient:
        if self._async_client is None:
            if not self._account_key:
                self._async_credential = AsyncDefaultAzureCredential()
            self._async_client = AsyncBlobServiceClient(
                account_url=self._account_url,
                credential=self._account_key or self._async_credential,
                retry_total=0,
            )
        return self._async_client

    # Health & Lifecycle

    def ping(self) -> bool:
        try:
            self.sync_client.get_account_information()
            return True
        except AzureError:
            return False

    async def aping(self) -> bool:
        try:
            await self.async_client.get_account_information()
            return True
        except AzureError:
            return False

    def close(self) -> None:
        close_all(self._sync_client, self._credential)
        self._sync_client = None
        self._credential = None

    async def aclose(self) -> None:
        await aclose_all(self._async_client, self._async_credential)
        self._async_client = None
        self._async_credential = None

    async def close_all(self) -> None:
        self.close()
        await self.aclose()
