from abc import ABC, abstractmethod
from typing import Any

from base.models.view_models.vector_search import VectorRecord, VectorSearchResult


class AbstractVectorSearchService(ABC):
    """
    Abstract base class for vector search services with both sync and async
    support.

    Subclasses must implement all methods for their specific vector database
    backend (e.g. Qdrant, Azure AI Search).
    """

    # Collection / Index Management

    @abstractmethod
    def create_collection(
        self,
        collection: str,
        vector_size: int,
        distance: str = "cosine",
        **kwargs: Any,
    ) -> None:
        """Create a collection/index for storing vectors.

        Args:
            collection: Name of the collection or index.
            vector_size: Dimensionality of the vectors.
            distance: Distance metric — ``"cosine"``, ``"dot"``, or
                ``"euclidean"``.
            **kwargs: Backend-specific options.
        """

    @abstractmethod
    async def acreate_collection(
        self,
        collection: str,
        vector_size: int,
        distance: str = "cosine",
        **kwargs: Any,
    ) -> None:
        """Create a collection/index asynchronously."""

    @abstractmethod
    def delete_collection(self, collection: str) -> None:
        """Delete a collection/index and all its data."""

    @abstractmethod
    async def adelete_collection(self, collection: str) -> None:
        """Delete a collection/index asynchronously."""

    @abstractmethod
    def collection_exists(self, collection: str) -> bool:
        """Check whether a collection/index exists."""

    @abstractmethod
    async def acollection_exists(self, collection: str) -> bool:
        """Check whether a collection/index exists asynchronously."""

    # Upsert

    @abstractmethod
    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        """Insert or update vector records.

        Implementations should handle batching internally when the number of
        records exceeds the backend's per-request limit.
        """

    @abstractmethod
    async def aupsert(self, collection: str, records: list[VectorRecord]) -> None:
        """Insert or update vector records asynchronously."""

    # Search

    @abstractmethod
    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Perform a similarity search.

        Args:
            collection: Target collection/index.
            vector: Query vector.
            limit: Maximum number of results to return.
            filters: Simple equality filters as ``{field: value}`` pairs.
                List values are treated as *any-of* conditions. Backends
                translate this into their native filter format.
            score_threshold: Minimum similarity score for results.
        """

    @abstractmethod
    async def asearch(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        """Perform a similarity search asynchronously."""

    # Get by ID

    @abstractmethod
    def get(
        self,
        collection: str,
        ids: list[str],
        with_vectors: bool = False,
    ) -> list[VectorRecord]:
        """Retrieve records by their IDs."""

    @abstractmethod
    async def aget(
        self,
        collection: str,
        ids: list[str],
        with_vectors: bool = False,
    ) -> list[VectorRecord]:
        """Retrieve records by their IDs asynchronously."""

    # Delete

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> None:
        """Delete records by their IDs."""

    @abstractmethod
    async def adelete(self, collection: str, ids: list[str]) -> None:
        """Delete records by their IDs asynchronously."""

    # Count

    @abstractmethod
    def count(self, collection: str) -> int:
        """Return the number of records in a collection."""

    @abstractmethod
    async def acount(self, collection: str) -> int:
        """Return the number of records in a collection asynchronously."""

    # Scroll / List

    @abstractmethod
    def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[VectorRecord], str | None]:
        """Iterate over records in a collection.

        Returns a tuple of ``(records, next_offset)``. Pass ``next_offset``
        back as ``offset`` to fetch the next page. ``None`` means no more
        pages.
        """

    @abstractmethod
    async def ascroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[VectorRecord], str | None]:
        """Iterate over records in a collection asynchronously."""

    # Health Check

    @abstractmethod
    def health_check(self) -> bool:
        """Verify the vector database is accessible."""

    @abstractmethod
    async def ahealth_check(self) -> bool:
        """Verify the vector database is accessible asynchronously."""

    # Metadata & Lifecycle

    @abstractmethod
    def get_service_name(self) -> str:
        """Return a human-readable name for this service."""

    @abstractmethod
    def dispose(self) -> None:
        """Release sync resources."""

    @abstractmethod
    async def adispose(self) -> None:
        """Release async resources."""
