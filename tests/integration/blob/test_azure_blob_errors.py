"""What a real server refuses, and the family a caller catches it as.

Every failure here arrives as an `AzureError` from the SDK; none of them may
leave this layer as one. A caller catches `BlobError` and never imports azure.
"""

from collections.abc import Callable
from typing import Any

import pytest
from azure.core.exceptions import AzureError

from base.models.view_models.exceptions.blob import (
    BlobBadRequestError,
    BlobConflictError,
    BlobError,
    BlobNotFoundError,
)
from base.services.blob.azure_blob import AzureBlobService

pytestmark = pytest.mark.integration


async def test_uploading_into_a_container_that_is_not_there_is_reported_as_missing(
    blob: AzureBlobService, unique: str
) -> None:
    with pytest.raises(BlobNotFoundError):
        await blob.aupload(f"missing{unique}", "note.txt", b"bytes")


async def test_downloading_a_blob_that_is_not_there_is_reported_as_missing(
    blob: AzureBlobService, container: str
) -> None:
    with pytest.raises(BlobNotFoundError):
        await blob.adownload(container, "never.txt")


async def test_downloading_from_a_container_that_is_not_there_is_reported_as_missing(
    blob: AzureBlobService, unique: str
) -> None:
    with pytest.raises(BlobNotFoundError):
        await blob.adownload(f"missing{unique}", "note.txt")


async def test_the_properties_of_a_blob_that_is_not_there_are_reported_as_missing(
    blob: AzureBlobService, container: str
) -> None:
    with pytest.raises(BlobNotFoundError):
        await blob.aget_properties(container, "never.txt")


async def test_deleting_a_blob_that_is_not_there_is_reported_as_missing(
    blob: AzureBlobService, container: str
) -> None:
    with pytest.raises(BlobNotFoundError):
        await blob.adelete(container, "never.txt")


async def test_a_blob_name_azure_refuses_is_reported_as_a_bad_request(
    blob: AzureBlobService, container: str
) -> None:
    with pytest.raises(BlobBadRequestError):
        await blob.aupload(container, "../escape.txt", b"bytes")


async def test_an_empty_blob_name_is_refused_as_a_blob_error_not_a_value_error(
    blob: AzureBlobService, container: str
) -> None:
    """The SDK refuses this one itself, with a `ValueError` that must not escape."""
    with pytest.raises(BlobError) as caught:
        await blob.aupload(container, "", b"bytes")
    assert isinstance(caught.value.provider_error, ValueError)


async def test_a_missing_blob_is_reported_with_the_status_code_azure_sent(
    blob: AzureBlobService, container: str
) -> None:
    with pytest.raises(BlobNotFoundError) as caught:
        await blob.adownload(container, "never.txt")
    assert caught.value.status_code == 404
    assert caught.value.request_id


async def test_a_domain_error_keeps_the_sdk_error_that_caused_it(
    blob: AzureBlobService, container: str
) -> None:
    with pytest.raises(BlobNotFoundError) as caught:
        await blob.adownload(container, "never.txt")
    assert isinstance(caught.value.provider_error, AzureError)


async def test_a_conflict_is_reported_with_the_status_code_azure_sent(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"first")
    with pytest.raises(BlobConflictError) as caught:
        await blob.aupload(container, "note.txt", b"second", overwrite=False)
    assert caught.value.status_code == 409


_REFUSED: dict[str, Callable[[AzureBlobService, str, str], Any]] = {
    "create_container": lambda service, container, missing: service.create_container(
        "ab"
    ),
    "delete_container": lambda service, container, missing: service.delete_container(
        missing
    ),
    "container_exists": lambda service, container, missing: service.container_exists(
        "ab"
    ),
    "upload": lambda service, container, missing: service.upload(
        missing, "note.txt", b"bytes"
    ),
    "download": lambda service, container, missing: service.download(
        container, "never.txt"
    ),
    "delete": lambda service, container, missing: service.delete(
        container, "never.txt"
    ),
    "exists": lambda service, container, missing: service.exists("ab", "note.txt"),
    "get_properties": lambda service, container, missing: service.get_properties(
        container, "never.txt"
    ),
    "list_blobs": lambda service, container, missing: list(service.list_blobs(missing)),
}
"""Every sync call a real server is about to refuse, one way or another."""


@pytest.mark.parametrize("operation", _REFUSED)
def test_no_raw_azure_error_escapes_this_layer(
    blob: AzureBlobService, container: str, unique: str, operation: str
) -> None:
    with pytest.raises(BlobError):
        _REFUSED[operation](blob, container, f"missing{unique}")
