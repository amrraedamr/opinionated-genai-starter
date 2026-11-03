from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timedelta, timezone

from base.tools.errors import MissingExtraError

try:
    from azure.storage.blob import (
        BlobSasPermissions,
        ContentSettings,
        generate_blob_sas,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError(
        "azure", "azure-storage-blob", feature="AzureBlobService"
    ) from exc

from base.clients.blob.azure_blob import AzureBlobClient
from base.services.blob import AbstractBlobService, BlobProperties
from base.services.common.meta import SingletonABC


class AzureBlobService(AbstractBlobService, metaclass=SingletonABC):
    """Azure Blob Storage-backed blob service with sync and async support."""

    def __init__(self) -> None:
        super().__init__()
        self._client: AzureBlobClient = AzureBlobClient()

    # Container Management

    def create_container(self, container: str) -> None:
        container_client = self._client.sync_client.get_container_client(container)
        if not container_client.exists():
            container_client.create_container()

    async def acreate_container(self, container: str) -> None:
        container_client = self._client.async_client.get_container_client(container)
        if not await container_client.exists():
            await container_client.create_container()

    def delete_container(self, container: str) -> None:
        self._client.sync_client.delete_container(container)

    async def adelete_container(self, container: str) -> None:
        await self._client.async_client.delete_container(container)

    def container_exists(self, container: str) -> bool:
        container_client = self._client.sync_client.get_container_client(container)
        return container_client.exists()

    async def acontainer_exists(self, container: str) -> bool:
        container_client = self._client.async_client.get_container_client(container)
        return await container_client.exists()

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
        blob_client = self._client.sync_client.get_blob_client(container, blob_name)
        blob_client.upload_blob(
            data,
            overwrite=overwrite,
            content_settings=ContentSettings(content_type=content_type),
            metadata=metadata,
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
        blob_client = self._client.async_client.get_blob_client(container, blob_name)
        await blob_client.upload_blob(
            data,
            overwrite=overwrite,
            content_settings=ContentSettings(content_type=content_type),
            metadata=metadata,
        )

    # Download

    def download(self, container: str, blob_name: str) -> bytes:
        blob_client = self._client.sync_client.get_blob_client(container, blob_name)
        stream = blob_client.download_blob()
        return stream.readall()

    async def adownload(self, container: str, blob_name: str) -> bytes:
        blob_client = self._client.async_client.get_blob_client(container, blob_name)
        stream = await blob_client.download_blob()
        return await stream.readall()

    # Delete

    def delete(self, container: str, blob_name: str) -> None:
        blob_client = self._client.sync_client.get_blob_client(container, blob_name)
        blob_client.delete_blob()

    async def adelete(self, container: str, blob_name: str) -> None:
        blob_client = self._client.async_client.get_blob_client(container, blob_name)
        await blob_client.delete_blob()

    # Exists

    def exists(self, container: str, blob_name: str) -> bool:
        blob_client = self._client.sync_client.get_blob_client(container, blob_name)
        return blob_client.exists()

    async def aexists(self, container: str, blob_name: str) -> bool:
        blob_client = self._client.async_client.get_blob_client(container, blob_name)
        return await blob_client.exists()

    # Properties

    def get_properties(self, container: str, blob_name: str) -> BlobProperties:
        blob_client = self._client.sync_client.get_blob_client(container, blob_name)
        props = blob_client.get_blob_properties()
        return self._to_blob_properties(props)

    async def aget_properties(self, container: str, blob_name: str) -> BlobProperties:
        blob_client = self._client.async_client.get_blob_client(container, blob_name)
        props = await blob_client.get_blob_properties()
        return self._to_blob_properties(props)

    # List Blobs

    def list_blobs(
        self, container: str, prefix: str | None = None
    ) -> Iterator[BlobProperties]:
        container_client = self._client.sync_client.get_container_client(container)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        for blob in blobs:
            yield BlobProperties(
                name=blob.name,
                size=blob.size or 0,
                content_type=(
                    blob.content_settings.content_type
                    if blob.content_settings
                    else "application/octet-stream"
                ),
                etag=blob.etag,
                last_modified=(
                    blob.last_modified.isoformat() if blob.last_modified else None
                ),
                metadata=blob.metadata or {},
            )

    async def alist_blobs(
        self, container: str, prefix: str | None = None
    ) -> AsyncIterator[BlobProperties]:
        container_client = self._client.async_client.get_container_client(container)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        async for blob in blobs:
            yield BlobProperties(
                name=blob.name,
                size=blob.size or 0,
                content_type=(
                    blob.content_settings.content_type
                    if blob.content_settings
                    else "application/octet-stream"
                ),
                etag=blob.etag,
                last_modified=(
                    blob.last_modified.isoformat() if blob.last_modified else None
                ),
                metadata=blob.metadata or {},
            )

    # Generate URL

    def generate_url(
        self,
        container: str,
        blob_name: str,
        expiry_seconds: int = 3600,
    ) -> str:
        user_delegation_key = self._client.sync_client.get_user_delegation_key(
            key_start_time=datetime.now(timezone.utc),
            key_expiry_time=datetime.now(timezone.utc)
            + timedelta(seconds=expiry_seconds),
        )
        sas_token = generate_blob_sas(
            account_name=self._client.sync_client.account_name,  # type: ignore
            container_name=container,
            blob_name=blob_name,
            user_delegation_key=user_delegation_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(seconds=expiry_seconds),
        )
        return f"{self._client.sync_client.url}{container}/{blob_name}?{sas_token}"

    async def agenerate_url(
        self,
        container: str,
        blob_name: str,
        expiry_seconds: int = 3600,
    ) -> str:
        user_delegation_key = await self._client.async_client.get_user_delegation_key(
            key_start_time=datetime.now(timezone.utc),
            key_expiry_time=datetime.now(timezone.utc)
            + timedelta(seconds=expiry_seconds),
        )
        sas_token = generate_blob_sas(
            account_name=self._client.async_client.account_name,  # type: ignore
            container_name=container,
            blob_name=blob_name,
            user_delegation_key=user_delegation_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(seconds=expiry_seconds),
        )
        return f"{self._client.async_client.url}{container}/{blob_name}?{sas_token}"

    # Metadata & Lifecycle

    def get_service_name(self) -> str:
        return "Azure Blob Storage"

    def dispose(self) -> None:
        self._client.close()

    async def adispose(self) -> None:
        await self._client.aclose()

    # Internals

    @staticmethod
    def _to_blob_properties(props: object) -> BlobProperties:
        """Convert Azure SDK ``BlobProperties`` to our ``BlobProperties``."""
        name: str = getattr(props, "name", "")
        size: int = getattr(props, "size", 0) or 0
        content_settings = getattr(props, "content_settings", None)
        content_type: str = (
            content_settings.content_type
            if content_settings and content_settings.content_type
            else "application/octet-stream"
        )
        etag: str | None = getattr(props, "etag", None)
        last_modified = getattr(props, "last_modified", None)
        metadata: dict[str, str] = getattr(props, "metadata", {}) or {}

        return BlobProperties(
            name=name,
            size=size,
            content_type=content_type,
            etag=etag,
            last_modified=last_modified.isoformat() if last_modified else None,
            metadata=metadata,
        )
