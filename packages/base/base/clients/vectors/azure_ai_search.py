from base.services.configuration import settings_for
from base.tools.errors import MissingExtraError

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import AzureError
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


class AzureAISearchClient(metaclass=Singleton):
    """
    Azure AI Search client with both sync and async support.
    Designed to be instantiated once and reused across the application.

    Provides cached ``SearchClient`` instances per index and shared
    ``SearchIndexClient`` instances for index management.
    """

    def __init__(self) -> None:
        config = settings_for(SupportsAzureSearchConfig)
        self._endpoint: str = config.azure_search_endpoint
        if not self._endpoint:
            raise ValueError("azure_search_endpoint must be configured in Settings")

        api_key: str = config.azure_search_api_key
        if not api_key:
            raise ValueError("azure_search_api_key must be configured in Settings")
        self._credential: AzureKeyCredential = AzureKeyCredential(api_key)

        self._sync_index_client: SearchIndexClient | None = None
        self._async_index_client: AsyncSearchIndexClient | None = None
        self._sync_search_clients: dict[str, SearchClient] = {}
        self._async_search_clients: dict[str, AsyncSearchClient] = {}

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
                endpoint=self._endpoint, credential=self._credential
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
                credential=self._credential,
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

    async def aclose(self) -> None:
        """Close all async clients."""
        for client in self._async_search_clients.values():
            await client.close()
        self._async_search_clients.clear()

        if self._async_index_client is not None:
            await self._async_index_client.close()
            self._async_index_client = None

    async def close_all(self) -> None:
        """Close both sync and async resources."""
        self.close()
        await self.aclose()
