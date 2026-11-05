from typing import Any

from base.services.configuration import settings_for
from base.tools.errors import MissingExtraError

try:
    from qdrant_client import AsyncQdrantClient as QdrantAsyncClient
    from qdrant_client import QdrantClient as QdrantSyncClient
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("vectors", "qdrant-client", feature="QdrantClient") from exc

from base.services.common.meta import Singleton
from base.services.configuration.protocols import SupportsQdrantConfig


class QdrantClient(metaclass=Singleton):
    """
    Qdrant vector database client with both sync and async support.
    Designed to be instantiated once and reused across the application.
    """

    def __init__(self) -> None:
        config = settings_for(SupportsQdrantConfig)
        self._url: str = config.qdrant_url
        self._api_key: str = config.qdrant_api_key
        self._grpc_port: int = config.qdrant_grpc_port
        self._prefer_grpc: bool = config.qdrant_prefer_grpc
        self._timeout: int = config.qdrant_timeout

        self._sync_client: QdrantSyncClient | None = None
        self._async_client: QdrantAsyncClient | None = None

    def _get_common_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "url": self._url,
            "timeout": self._timeout,
            "prefer_grpc": self._prefer_grpc,
            "grpc_port": self._grpc_port,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return kwargs

    # Sync

    def _ensure_sync_initialized(self) -> None:
        if self._sync_client is not None:
            return
        self._sync_client = QdrantSyncClient(**self._get_common_kwargs())

    @property
    def sync_client(self) -> QdrantSyncClient:
        """Sync Qdrant client."""
        self._ensure_sync_initialized()
        assert self._sync_client is not None
        return self._sync_client

    # Async

    def _ensure_async_initialized(self) -> None:
        if self._async_client is not None:
            return
        self._async_client = QdrantAsyncClient(**self._get_common_kwargs())

    @property
    def async_client(self) -> QdrantAsyncClient:
        """Async Qdrant client."""
        self._ensure_async_initialized()
        assert self._async_client is not None
        return self._async_client

    # Health & Lifecycle

    def ping(self) -> bool:
        """Sync health check."""
        try:
            self.sync_client.get_collections()
            return True
        except (UnexpectedResponse, ResponseHandlingException, OSError):
            return False

    async def aping(self) -> bool:
        """Async health check."""
        try:
            await self.async_client.get_collections()
            return True
        except (UnexpectedResponse, ResponseHandlingException, OSError):
            return False

    def close(self) -> None:
        """Close sync client."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        """Close async client."""
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    async def close_all(self) -> None:
        """Close both sync and async resources."""
        self.close()
        await self.aclose()
