from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

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
from base.models.view_models.blob import BlobProperties
from base.services.blob import AbstractBlobService
from base.services.common.meta import SingletonABC
from base.services.error_handling import aretrying, retrying
from base.services.error_handling.blob.azure import AzureBlobErrorHandler

_LIST_INCLUDE: list[str] = ["metadata"]
"""Azure leaves metadata out of a listing unless it is asked for."""


class AzureBlobService(AbstractBlobService, metaclass=SingletonABC):
    """Azure Blob Storage-backed blob service with sync and async support."""

    def __init__(self) -> None:
        super().__init__()
        self._client: AzureBlobClient = AzureBlobClient()
        self._error_handler: AzureBlobErrorHandler = AzureBlobErrorHandler()

    # Container Management

    @retrying
    def create_container(self, container: str) -> None:
        container_client = self._client.sync_client.get_container_client(container)
        if not container_client.exists():
            container_client.create_container()

    @aretrying
    async def acreate_container(self, container: str) -> None:
        container_client = self._client.async_client.get_container_client(container)
        if not await container_client.exists():
            await container_client.create_container()

    @retrying
    def delete_container(self, container: str) -> None:
        self._client.sync_client.delete_container(container)

    @aretrying
    async def adelete_container(self, container: str) -> None:
        await self._client.async_client.delete_container(container)

    @retrying
    def container_exists(self, container: str) -> bool:
        container_client = self._client.sync_client.get_container_client(container)
        return container_client.exists()

    @aretrying
    async def acontainer_exists(self, container: str) -> bool:
        container_client = self._client.async_client.get_container_client(container)
        return await container_client.exists()

    # Upload

    @retrying
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

    @aretrying
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

    @retrying
    def download(self, container: str, blob_name: str) -> bytes:
        blob_client = self._client.sync_client.get_blob_client(container, blob_name)
        stream = blob_client.download_blob()
        return stream.readall()

    @aretrying
    async def adownload(self, container: str, blob_name: str) -> bytes:
        blob_client = self._client.async_client.get_blob_client(container, blob_name)
        stream = await blob_client.download_blob()
        return await stream.readall()

    # Delete

    @retrying
    def delete(self, container: str, blob_name: str) -> None:
        blob_client = self._client.sync_client.get_blob_client(container, blob_name)
        blob_client.delete_blob()

    @aretrying
    async def adelete(self, container: str, blob_name: str) -> None:
        blob_client = self._client.async_client.get_blob_client(container, blob_name)
        await blob_client.delete_blob()

    # Exists

    @retrying
    def exists(self, container: str, blob_name: str) -> bool:
        blob_client = self._client.sync_client.get_blob_client(container, blob_name)
        return blob_client.exists()

    @aretrying
    async def aexists(self, container: str, blob_name: str) -> bool:
        blob_client = self._client.async_client.get_blob_client(container, blob_name)
        return await blob_client.exists()

    # Properties

    @retrying
    def get_properties(self, container: str, blob_name: str) -> BlobProperties:
        blob_client = self._client.sync_client.get_blob_client(container, blob_name)
        props = blob_client.get_blob_properties()
        return self._to_blob_properties(props)

    @aretrying
    async def aget_properties(self, container: str, blob_name: str) -> BlobProperties:
        blob_client = self._client.async_client.get_blob_client(container, blob_name)
        props = await blob_client.get_blob_properties()
        return self._to_blob_properties(props)

    # List Blobs

    def list_blobs(
        self, container: str, prefix: str | None = None
    ) -> Iterator[BlobProperties]:
        container_client = self._client.sync_client.get_container_client(container)
        try:
            blobs = container_client.list_blobs(
                name_starts_with=prefix, include=_LIST_INCLUDE
            )
            for blob in blobs:
                yield self._to_blob_properties(blob)
        except Exception as exc:
            # Retrying a half-drained pager would repeat what it already
            # yielded, so a generator translates rather than retries.
            raise self._error_handler.as_domain_error(exc) from exc

    async def alist_blobs(
        self, container: str, prefix: str | None = None
    ) -> AsyncIterator[BlobProperties]:
        container_client = self._client.async_client.get_container_client(container)
        try:
            blobs = container_client.list_blobs(
                name_starts_with=prefix, include=_LIST_INCLUDE
            )
            async for blob in blobs:
                yield self._to_blob_properties(blob)
        except Exception as exc:
            # Retrying a half-drained pager would repeat what it already
            # yielded, so a generator translates rather than retries.
            raise self._error_handler.as_domain_error(exc) from exc

    # Generate URL

    @retrying
    def generate_url(
        self,
        container: str,
        blob_name: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """A read-only SAS URL, signed by whichever credential this is holding.

        A user delegation key is minted by Entra ID, so it exists only on the
        identity path; a shared key signs the token itself.
        """
        expiry = datetime.now(timezone.utc) + timedelta(seconds=expiry_seconds)
        signing: dict[str, Any] = (
            {"account_key": self._client.account_key}
            if self._client.account_key
            else {
                "user_delegation_key": self._client.sync_client.get_user_delegation_key(
                    key_start_time=datetime.now(timezone.utc),
                    key_expiry_time=expiry,
                )
            }
        )
        sas_token = generate_blob_sas(
            account_name=self._client.sync_client.account_name,  # type: ignore
            container_name=container,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
            **signing,
        )
        return self._sas_url(
            self._client.sync_client.url, container, blob_name, sas_token
        )

    @aretrying
    async def agenerate_url(
        self,
        container: str,
        blob_name: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """A read-only SAS URL, signed by whichever credential this is holding.

        A user delegation key is minted by Entra ID, so it exists only on the
        identity path; a shared key signs the token itself.
        """
        expiry = datetime.now(timezone.utc) + timedelta(seconds=expiry_seconds)
        client = self._client.async_client
        signing: dict[str, Any] = (
            {"account_key": self._client.account_key}
            if self._client.account_key
            else {
                "user_delegation_key": await client.get_user_delegation_key(
                    key_start_time=datetime.now(timezone.utc),
                    key_expiry_time=expiry,
                )
            }
        )
        sas_token = generate_blob_sas(
            account_name=client.account_name,  # type: ignore
            container_name=container,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
            **signing,
        )
        return self._sas_url(client.url, container, blob_name, sas_token)

    # Metadata & Lifecycle

    def get_service_name(self) -> str:
        return "Azure Blob Storage"

    def dispose(self) -> None:
        self._client.close()

    async def adispose(self) -> None:
        await self._client.aclose()

    # Internals

    @staticmethod
    def _sas_url(
        account_url: str, container: str, blob_name: str, sas_token: str
    ) -> str:
        """The blob's address with a token on it.

        The token is signed over the unescaped name, but the URL carries the
        escaped one — a name holding a space, a `#` or a `?` is otherwise an
        address that reaches a different blob or none at all.
        """
        return (
            f"{account_url}{quote(container)}/{quote(blob_name, safe='~/')}?{sas_token}"
        )

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
