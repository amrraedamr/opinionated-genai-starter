from typing import Any

from base.tools.errors import MissingExtraError

try:
    from azure.core.exceptions import AzureError
    from azure.cosmos import ContainerProxy, CosmosClient, DatabaseProxy
    from azure.cosmos.aio import ContainerProxy as AsyncContainerProxy
    from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
    from azure.cosmos.aio import DatabaseProxy as AsyncDatabaseProxy
    from azure.identity import DefaultAzureCredential
    from azure.identity.aio import (
        DefaultAzureCredential as AsyncDefaultAzureCredential,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("azure", "azure-cosmos", feature="CosmosDBClient") from exc

from base.services.common.meta import Singleton
from base.services.configuration import get_settings
from base.tools import resources


class CosmosDBClient(metaclass=Singleton):
    """Azure Cosmos DB client (SQL API) with both sync and async support.

    Designed to be instantiated once and reused across the application.
    Lazy-initializes the underlying SDK clients on first access.
    """

    def __init__(self) -> None:
        self._endpoint: str = get_settings().azure_cosmos_account_uri
        if not self._endpoint:
            raise ValueError("azure_cosmos_account_uri must be configured in Settings")

        self._account_key: str | None = get_settings().azure_cosmos_account_key or None

        self._sync_client: CosmosClient | None = None
        self._async_client: AsyncCosmosClient | None = None
        self._credential: DefaultAzureCredential | None = None
        self._async_credential: AsyncDefaultAzureCredential | None = None

    def _get_common_kwargs(self) -> dict[str, Any]:
        return {}

    # Client accessors (lazy init)

    @property
    def sync_client(self) -> CosmosClient:
        if self._sync_client is None:
            self._credential = DefaultAzureCredential()
            self._sync_client = CosmosClient(
                url=self._endpoint,
                credential=self._account_key if self._account_key else self._credential,
                **self._get_common_kwargs(),
            )
        return self._sync_client

    @property
    def async_client(self) -> AsyncCosmosClient:
        if self._async_client is None:
            self._async_credential = AsyncDefaultAzureCredential()
            self._async_client = AsyncCosmosClient(
                url=self._endpoint,
                credential=self._account_key
                if self._account_key
                else self._async_credential,
                **self._get_common_kwargs(),
            )
        return self._async_client

    # Database & Container accessors

    def get_database(self, database: str) -> DatabaseProxy:
        return self.sync_client.get_database_client(database)

    def aget_database(self, database: str) -> AsyncDatabaseProxy:
        return self.async_client.get_database_client(database)

    def get_container(self, database: str, container: str) -> ContainerProxy:
        return self.get_database(database).get_container_client(container)

    def aget_container(self, database: str, container: str) -> AsyncContainerProxy:
        return self.aget_database(database).get_container_client(container)

    # Health & Lifecycle

    def ping(self) -> bool:
        """Sync health check — enumerate databases as a lightweight probe."""
        try:
            next(iter(self.sync_client.list_databases()), None)
            return True
        except AzureError:
            return False

    async def aping(self) -> bool:
        """Async health check."""
        try:
            async for _ in self.async_client.list_databases():
                break
            return True
        except AzureError:
            return False

    def close(self) -> None:
        """Close sync client and credential.

        Sync CosmosClient has no public close() — dropping the reference
        lets GC clean up the underlying requests.Session.
        """
        resources.close_all(self._credential)
        self._sync_client = None
        self._credential = None

    async def aclose(self) -> None:
        """Close async client and credential."""
        await resources.aclose_all(self._async_client, self._async_credential)
        self._async_client = None
        self._async_credential = None

    async def close_all(self) -> None:
        """Close both sync and async resources."""
        self.close()
        await self.aclose()
