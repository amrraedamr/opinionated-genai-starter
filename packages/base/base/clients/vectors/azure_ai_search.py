from base.services.configuration import settings_for
from base.tools.errors import MissingExtraError

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import AzureError
    from azure.identity import DefaultAzureCredential
    from azure.identity.aio import (
        DefaultAzureCredential as AsyncDefaultAzureCredential,
    )
    from azure.search.documents import SearchClient
    from azure.search.documents.aio import SearchClient as AsyncSearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.aio import (
        SearchIndexClient as AsyncSearchIndexClient,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "azure", "azure-search-documents", feature="AzureAISearchClient"
    ) from exc

from base.services.common.meta import Singleton
from base.services.configuration.protocols import SupportsAzureSearchConfig
from base.tools.resources import aclose_all, close_all


class AzureAISearchClient(metaclass=Singleton):
    """
    Azure AI Search client with both sync and async support.
    Designed to be instantiated once and reused across the application.

    Provides cached ``SearchClient`` instances per index and shared
    ``SearchIndexClient`` instances for index management.

    An API key is used when one is configured; otherwise the client falls back
    to `DefaultAzureCredential`, so a deployment holding a role assignment
    needs no key at all. The async clients take the async credential, which is
    a separate object with a separate close.
    """

    def __init__(self) -> None:
        config = settings_for(SupportsAzureSearchConfig)
        self._endpoint: str = config.azure_search_endpoint
        if not self._endpoint:
            raise ValueError("azure_search_endpoint must be configured in Settings")

        api_key: str = config.azure_search_api_key
        self._key_credential: AzureKeyCredential | None = (
            AzureKeyCredential(api_key) if api_key else None
        )
        self._managed_credential: DefaultAzureCredential | None = None
        self._async_managed_credential: AsyncDefaultAzureCredential | None = None

        self._sync_index_client: SearchIndexClient | None = None
        self._async_index_client: AsyncSearchIndexClient | None = None
        self._sync_search_clients: dict[str, SearchClient] = {}
        self._async_search_clients: dict[str, AsyncSearchClient] = {}

    # Credentials

    @property
    def _credential(self) -> AzureKeyCredential | DefaultAzureCredential:
        """The key when there is one, else the identity this deployment runs as."""
        if self._key_credential is not None:
            return self._key_credential
        if self._managed_credential is None:
            self._managed_credential = DefaultAzureCredential()
        return self._managed_credential

    @property
    def _async_credential(self) -> AzureKeyCredential | AsyncDefaultAzureCredential:
        """The key when there is one, else the identity this deployment runs as."""
        if self._key_credential is not None:
            return self._key_credential
        if self._async_managed_credential is None:
            self._async_managed_credential = AsyncDefaultAzureCredential()
        return self._async_managed_credential

    # Index Client (for index management)

    @property
    def sync_index_client(self) -> SearchIndexClient:
        """Sync client for index management operations."""
        if self._sync_index_client is None:
            self._sync_index_client = SearchIndexClient(
                endpoint=self._endpoint, credential=self._credential
            )
        return self._sync_index_client

    @property
    def async_index_client(self) -> AsyncSearchIndexClient:
        """Async client for index management operations."""
        if self._async_index_client is None:
            self._async_index_client = AsyncSearchIndexClient(
                endpoint=self._endpoint, credential=self._async_credential
            )
        return self._async_index_client

    # Search Client (per-index, cached)

    def get_search_client(self, index_name: str) -> SearchClient:
        """Get or create a sync SearchClient for a specific index."""
        if index_name not in self._sync_search_clients:
            self._sync_search_clients[index_name] = SearchClient(
                endpoint=self._endpoint,
                index_name=index_name,
                credential=self._credential,
            )
        return self._sync_search_clients[index_name]

    def get_async_search_client(self, index_name: str) -> AsyncSearchClient:
        """Get or create an async SearchClient for a specific index."""
        if index_name not in self._async_search_clients:
            self._async_search_clients[index_name] = AsyncSearchClient(
                endpoint=self._endpoint,
                index_name=index_name,
                credential=self._async_credential,
            )
        return self._async_search_clients[index_name]

    # Health & Lifecycle

    def ping(self) -> bool:
        """Sync health check via listing indexes."""
        try:
            list(self.sync_index_client.list_index_names())
            return True
        except AzureError:
            return False

    async def aping(self) -> bool:
        """Async health check via listing indexes."""
        try:
            async for _ in self.async_index_client.list_index_names():
                break
            return True
        except AzureError:
            return False

    def close(self) -> None:
        """Close all sync clients."""
        for client in self._sync_search_clients.values():
            client.close()
        self._sync_search_clients.clear()

        if self._sync_index_client is not None:
            self._sync_index_client.close()
            self._sync_index_client = None

        close_all(self._managed_credential)
        self._managed_credential = None

    async def aclose(self) -> None:
        """Close all async clients."""
        for client in self._async_search_clients.values():
            await client.close()
        self._async_search_clients.clear()

        if self._async_index_client is not None:
            await self._async_index_client.close()
            self._async_index_client = None

        await aclose_all(self._async_managed_credential)
        self._async_managed_credential = None

    async def close_all(self) -> None:
        """Close both sync and async resources."""
        self.close()
        await self.aclose()
