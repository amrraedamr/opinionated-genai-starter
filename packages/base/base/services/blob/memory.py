from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

from base.services.blob import AbstractBlobService, BlobProperties
from base.services.common.meta import SingletonABC
from base.services.logging.console import ConsoleLoggerService


@dataclass
class _StoredBlob:
    """Internal representation of a blob held in memory."""

    data: bytes
    content_type: str
    metadata: dict[str, str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class InMemoryBlobService(AbstractBlobService, metaclass=SingletonABC):
    """In-memory blob storage for local development and testing.

    Data is stored in plain dicts and is lost when the process exits.
    Thread safety is *not* guaranteed — use only in single-threaded /
    single-worker development environments.
    """

    def __init__(self) -> None:
        super().__init__()
        self._containers: dict[str, dict[str, _StoredBlob]] = {}
        self._logger: ConsoleLoggerService = ConsoleLoggerService()

    # Container Management

    def create_container(self, container: str) -> None:
        if container not in self._containers:
            self._containers[container] = {}
            self._logger.debug("InMemoryBlob: created container %s", container)

    async def acreate_container(self, container: str) -> None:
        self.create_container(container)

    def delete_container(self, container: str) -> None:
        self._containers.pop(container, None)
        self._logger.debug("InMemoryBlob: deleted container %s", container)

    async def adelete_container(self, container: str) -> None:
        self.delete_container(container)

    def container_exists(self, container: str) -> bool:
        return container in self._containers

    async def acontainer_exists(self, container: str) -> bool:
        return self.container_exists(container)

    # Upload

    def upload(
        self,
        container: str,
        blob_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        overwrite: bool = True,
        metadata: dict[str, str] | None = None,
    ) -> None:
        blobs = self._get_container(container)
        if not overwrite and blob_name in blobs:
            raise FileExistsError(
                f"Blob '{blob_name}' already exists in container '{container}'"
            )
        blobs[blob_name] = _StoredBlob(
            data=data,
            content_type=content_type,
            metadata=metadata or {},
        )

    async def aupload(
        self,
        container: str,
        blob_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        overwrite: bool = True,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.upload(container, blob_name, data, content_type, overwrite, metadata)

    # Download

    def download(self, container: str, blob_name: str) -> bytes:
        blob = self._get_blob(container, blob_name)
        return blob.data

    async def adownload(self, container: str, blob_name: str) -> bytes:
        return self.download(container, blob_name)

    # Delete

    def delete(self, container: str, blob_name: str) -> None:
        blobs = self._get_container(container)
        blobs.pop(blob_name, None)

    async def adelete(self, container: str, blob_name: str) -> None:
        self.delete(container, blob_name)

    # Exists

    def exists(self, container: str, blob_name: str) -> bool:
        if container not in self._containers:
            return False
        return blob_name in self._containers[container]

    async def aexists(self, container: str, blob_name: str) -> bool:
        return self.exists(container, blob_name)

    # Properties

    def get_properties(self, container: str, blob_name: str) -> BlobProperties:
        blob = self._get_blob(container, blob_name)
        return BlobProperties(
            name=blob_name,
            size=len(blob.data),
            content_type=blob.content_type,
            etag=None,
            last_modified=blob.created_at.isoformat(),
            metadata=blob.metadata,
        )

    async def aget_properties(self, container: str, blob_name: str) -> BlobProperties:
        return self.get_properties(container, blob_name)

    # List Blobs

    def list_blobs(
        self, container: str, prefix: str | None = None
    ) -> Iterator[BlobProperties]:
        blobs = self._get_container(container)
        for name, blob in blobs.items():
            if prefix and not name.startswith(prefix):
                continue
            yield BlobProperties(
                name=name,
                size=len(blob.data),
                content_type=blob.content_type,
                etag=None,
                last_modified=blob.created_at.isoformat(),
                metadata=blob.metadata,
            )

    async def alist_blobs(
        self, container: str, prefix: str | None = None
    ) -> AsyncIterator[BlobProperties]:
        blobs = self._get_container(container)
        for name, blob in blobs.items():
            if prefix and not name.startswith(prefix):
                continue
            yield BlobProperties(
                name=name,
                size=len(blob.data),
                content_type=blob.content_type,
                etag=None,
                last_modified=blob.created_at.isoformat(),
                metadata=blob.metadata,
            )

    # Generate URL

    def generate_url(
        self,
        container: str,
        blob_name: str,
        expiry_seconds: int = 3600,
    ) -> str:
        return f"memory://{container}/{blob_name}"

    async def agenerate_url(
        self,
        container: str,
        blob_name: str,
        expiry_seconds: int = 3600,
    ) -> str:
        return self.generate_url(container, blob_name, expiry_seconds)

    # Metadata & Lifecycle

    def get_service_name(self) -> str:
        return "In-Memory Blob Storage"

    def dispose(self) -> None:
        self._containers.clear()

    async def adispose(self) -> None:
        self.dispose()

    # Internals

    def _get_container(self, container: str) -> dict[str, _StoredBlob]:
        if container not in self._containers:
            raise FileNotFoundError(f"Container '{container}' does not exist")
        return self._containers[container]

    def _get_blob(self, container: str, blob_name: str) -> _StoredBlob:
        blobs = self._get_container(container)
        if blob_name not in blobs:
            raise FileNotFoundError(
                f"Blob '{blob_name}' not found in container '{container}'"
            )
        return blobs[blob_name]
