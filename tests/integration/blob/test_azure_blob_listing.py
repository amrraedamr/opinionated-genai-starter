"""Listing a real container: what each entry carries, and how far it walks.

Azure leaves metadata out of a listing unless the request asks for it, and the
answer arrives a page at a time, so a listing is only as true as the request
behind it. Neither generator is retry-decorated — repeating a half-drained
pager would yield what it already yielded — so both translate instead.
"""

import pytest

from base.models.view_models.exceptions.blob import (
    BlobBadRequestError,
    BlobNotFoundError,
)
from base.services.blob.azure_blob import AzureBlobService

pytestmark = pytest.mark.integration


async def test_a_new_container_lists_nothing(
    blob: AzureBlobService, container: str
) -> None:
    assert [b async for b in blob.alist_blobs(container)] == []


async def test_every_uploaded_blob_is_listed(
    blob: AzureBlobService, container: str
) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        await blob.aupload(container, name, b"bytes")
    listed = {b.name async for b in blob.alist_blobs(container)}
    assert listed == {"a.txt", "b.txt", "c.txt"}


async def test_a_prefix_narrows_the_listing(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "reports/q1.pdf", b"bytes")
    await blob.aupload(container, "reports/q2.pdf", b"bytes")
    await blob.aupload(container, "invoices/i1.pdf", b"bytes")
    listed = {b.name async for b in blob.alist_blobs(container, "reports/")}
    assert listed == {"reports/q1.pdf", "reports/q2.pdf"}


async def test_a_prefix_that_matches_nothing_lists_nothing(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "reports/q1.pdf", b"bytes")
    assert [b async for b in blob.alist_blobs(container, "nope/")] == []


async def test_a_listed_blob_carries_its_size_and_content_type(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "page.html", b"<p>hi</p>", "text/html")
    listed = [b async for b in blob.alist_blobs(container)][0]
    assert listed.size == 9
    assert listed.content_type == "text/html"
    assert listed.etag
    assert listed.last_modified


async def test_a_listed_blob_carries_the_metadata_it_was_stored_with(
    blob: AzureBlobService, container: str
) -> None:
    """Azure omits metadata from a listing unless the request asks for it."""
    await blob.aupload(container, "owned.txt", b"bytes", metadata={"owner": "ada"})
    listed = [b async for b in blob.alist_blobs(container)][0]
    assert listed.metadata == {"owner": "ada"}


async def test_a_listed_blob_stored_without_metadata_carries_an_empty_mapping(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "bare.txt", b"bytes")
    listed = [b async for b in blob.alist_blobs(container)][0]
    assert listed.metadata == {}


async def test_listing_reports_what_getting_the_properties_reports(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "one.txt", b"bytes", "text/plain", True, {"n": "1"})
    listed = [b async for b in blob.alist_blobs(container)][0]
    properties = await blob.aget_properties(container, "one.txt")
    assert (listed.name, listed.size, listed.content_type, listed.metadata) == (
        properties.name,
        properties.size,
        properties.content_type,
        properties.metadata,
    )


async def test_listing_walks_past_the_end_of_one_page(
    blob: AzureBlobService, container: str
) -> None:
    """Proved by asking the same server for two per page and counting pages."""
    names = {f"{i}.txt" for i in range(7)}
    for name in names:
        await blob.aupload(container, name, b"bytes")

    client = blob._client.async_client.get_container_client(container)
    pages = [page async for page in client.list_blobs(results_per_page=2).by_page()]
    assert len(pages) > 1

    assert {b.name async for b in blob.alist_blobs(container)} == names


# Failures


async def test_listing_a_container_that_is_not_there_is_reported_as_missing(
    blob: AzureBlobService, unique: str
) -> None:
    with pytest.raises(BlobNotFoundError):
        [b async for b in blob.alist_blobs(f"missing{unique}")]


async def test_listing_a_badly_named_container_is_reported_as_a_bad_request(
    blob: AzureBlobService,
) -> None:
    with pytest.raises(BlobBadRequestError):
        [b async for b in blob.alist_blobs("ab")]


async def test_a_listing_failure_keeps_the_sdk_error_that_caused_it(
    blob: AzureBlobService, unique: str
) -> None:
    with pytest.raises(BlobNotFoundError) as caught:
        [b async for b in blob.alist_blobs(f"missing{unique}")]
    assert caught.value.status_code == 404


# Sync counterparts


def test_the_sync_path_lists_the_same_blobs(
    blob: AzureBlobService, container: str
) -> None:
    for name in ("a.txt", "b.txt"):
        blob.upload(container, name, b"bytes")
    assert {b.name for b in blob.list_blobs(container)} == {"a.txt", "b.txt"}


def test_the_sync_path_narrows_by_prefix(
    blob: AzureBlobService, container: str
) -> None:
    blob.upload(container, "reports/q1.pdf", b"bytes")
    blob.upload(container, "invoices/i1.pdf", b"bytes")
    assert [b.name for b in blob.list_blobs(container, "reports/")] == [
        "reports/q1.pdf"
    ]


def test_the_sync_path_lists_metadata_too(
    blob: AzureBlobService, container: str
) -> None:
    blob.upload(container, "owned.txt", b"bytes", metadata={"owner": "ada"})
    assert list(blob.list_blobs(container))[0].metadata == {"owner": "ada"}


def test_the_sync_path_reports_a_missing_container_as_missing(
    blob: AzureBlobService, unique: str
) -> None:
    with pytest.raises(BlobNotFoundError):
        list(blob.list_blobs(f"missing{unique}"))


def test_the_sync_path_reports_a_badly_named_container_as_a_bad_request(
    blob: AzureBlobService,
) -> None:
    with pytest.raises(BlobBadRequestError):
        list(blob.list_blobs("ab"))


def test_calling_the_sync_listing_does_no_work_until_it_is_iterated(
    blob: AzureBlobService, unique: str
) -> None:
    generator = blob.list_blobs(f"missing{unique}")
    with pytest.raises(BlobNotFoundError):
        list(generator)
