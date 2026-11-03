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

from base.services.common.meta import Singleton
from base.services.configuration import get_settings
from base.tools import resources


class AzureBlobClient(metaclass=Singleton):
    """Azure Blob Storage client with both sync and async support.

    Designed to be instantiated once and reused across the application.
    Lazy-initializes the underlying SDK clients on first access.
    """

    def __init__(self) -> None:
        self._account_url: str = get_settings().azure_blob_account_url
        if not self._account_url:
            raise ValueError("azure_blob_account_url must be configured in Settings")

        self._sync_client: BlobServiceClient | None = None
        self._async_client: AsyncBlobServiceClient | None = None
        self._credential: DefaultAzureCredential | None = None
        self._async_credential: AsyncDefaultAzureCredential | None = None

    # Client accessors (lazy init)

    @property
    def sync_client(self) -> BlobServiceClient:
        if self._sync_client is None:
            self._credential = DefaultAzureCredential()
            self._sync_client = BlobServiceClient(
                account_url=self._account_url,
                credential=self._credential,
            )
        return self._sync_client

    @property
    def async_client(self) -> AsyncBlobServiceClient:
        if self._async_client is None:
            self._async_credential = AsyncDefaultAzureCredential()
            self._async_client = AsyncBlobServiceClient(
                account_url=self._account_url,
                credential=self._async_credential,
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
        resources.close_all(self._sync_client, self._credential)
        self._sync_client = None
        self._credential = None

    async def aclose(self) -> None:
        await resources.aclose_all(self._async_client, self._async_credential)
        self._async_client = None
        self._async_credential = None

    async def close_all(self) -> None:
        self.close()
        await self.aclose()
