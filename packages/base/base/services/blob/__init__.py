from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any


class BlobProperties(dict[str, Any]):
    """Lightweight wrapper around blob metadata/properties returned by backends.

    Acts as a plain dict so callers can access arbitrary keys while the most
    common properties are exposed as typed helpers.
    """

    @property
    def name(self) -> str:
        return str(self.get("name", ""))

    @property
    def size(self) -> int:
        return int(self.get("size", 0))

    @property
    def content_type(self) -> str:
        return str(self.get("content_type", "application/octet-stream"))

    @property
    def etag(self) -> str | None:
        return self.get("etag")

    @property
    def last_modified(self) -> str | None:
        return self.get("last_modified")


class AbstractBlobService(ABC):
    """Abstract base class for blob / object storage services."""

    # Container / Bucket Management

    @abstractmethod
    def create_container(self, container: str) -> None:
        """Create a container/bucket if it does not already exist."""

    @abstractmethod
    async def acreate_container(self, container: str) -> None:
        """Create a container/bucket asynchronously."""

    @abstractmethod
    def delete_container(self, container: str) -> None:
        """Delete a container/bucket and all its blobs."""

    @abstractmethod
    async def adelete_container(self, container: str) -> None:
        """Delete a container/bucket asynchronously."""

    @abstractmethod
    def container_exists(self, container: str) -> bool:
        """Check whether a container/bucket exists."""

    @abstractmethod
    async def acontainer_exists(self, container: str) -> bool:
        """Check whether a container/bucket exists asynchronously."""

    # Upload

    @abstractmethod
    def upload(
        self,
        container: str,
        blob_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        overwrite: bool = True,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Upload bytes to a blob.

        Args:
            container: Target container/bucket.
            blob_name: Destination blob key/path.
            data: Raw bytes to store.
            content_type: MIME type of the blob.
            overwrite: If ``False``, raise if the blob already exists.
            metadata: Arbitrary key-value metadata attached to the blob.
        """

    @abstractmethod
    async def aupload(
        self,
        container: str,
        blob_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        overwrite: bool = True,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Upload bytes to a blob asynchronously."""

    # Download

    @abstractmethod
    def download(self, container: str, blob_name: str) -> bytes:
        """Download a blob and return its contents as bytes."""

    @abstractmethod
    async def adownload(self, container: str, blob_name: str) -> bytes:
        """Download a blob asynchronously."""

    # Delete

    @abstractmethod
    def delete(self, container: str, blob_name: str) -> None:
        """Delete a single blob."""

    @abstractmethod
    async def adelete(self, container: str, blob_name: str) -> None:
        """Delete a single blob asynchronously."""

    # Exists

    @abstractmethod
    def exists(self, container: str, blob_name: str) -> bool:
        """Check whether a blob exists."""

    @abstractmethod
    async def aexists(self, container: str, blob_name: str) -> bool:
        """Check whether a blob exists asynchronously."""

    # Properties / Metadata

    @abstractmethod
    def get_properties(self, container: str, blob_name: str) -> BlobProperties:
        """Retrieve blob properties (size, content type, metadata, etc.)."""

    @abstractmethod
    async def aget_properties(self, container: str, blob_name: str) -> BlobProperties:
        """Retrieve blob properties asynchronously."""

    # List Blobs

    @abstractmethod
    def list_blobs(
        self, container: str, prefix: str | None = None
    ) -> Iterator[BlobProperties]:
        """Iterate over blobs in a container, optionally filtered by prefix."""

    @abstractmethod
    def alist_blobs(
        self, container: str, prefix: str | None = None
    ) -> AsyncIterator[BlobProperties]:
        """Iterate over blobs in a container asynchronously."""

    # Generate URL

    @abstractmethod
    def generate_url(
        self,
        container: str,
        blob_name: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate a time-limited URL for downloading a blob.

        For backends that don't support signed URLs this may return a
        placeholder or raise ``NotImplementedError``.
        """

    @abstractmethod
    async def agenerate_url(
        self,
        container: str,
        blob_name: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate a time-limited download URL asynchronously."""

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
