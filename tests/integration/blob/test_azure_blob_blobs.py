"""Blobs against a real server: what goes in, what comes back, and what is kept.

Azure's put replaces a blob whole, so an overwrite that names no metadata
leaves none behind — the second upload is a new blob under an old name rather
than an edit of the old one.
"""

import io

import pytest

from base.models.view_models.exceptions.blob import BlobConflictError
from base.services.blob.azure_blob import AzureBlobService

pytestmark = pytest.mark.integration


async def test_a_blob_comes_back_with_the_bytes_it_went_in_with(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"hello")
    assert await blob.adownload(container, "note.txt") == b"hello"


async def test_an_empty_blob_round_trips(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "empty.bin", b"")
    assert await blob.adownload(container, "empty.bin") == b""


async def test_bytes_that_are_not_text_round_trip(
    blob: AzureBlobService, container: str
) -> None:
    payload = bytes(range(256))
    await blob.aupload(container, "raw.bin", payload)
    assert await blob.adownload(container, "raw.bin") == payload


async def test_a_blob_bigger_than_a_single_buffer_round_trips(
    blob: AzureBlobService, container: str
) -> None:
    payload = b"x" * (1024 * 1024)
    await blob.aupload(container, "big.bin", payload)
    assert await blob.adownload(container, "big.bin") == payload


async def test_a_stream_is_stored_as_the_bytes_it_yields(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "streamed.bin", io.BytesIO(b"from a stream"))  # type: ignore[arg-type]
    assert await blob.adownload(container, "streamed.bin") == b"from a stream"


async def test_a_nested_blob_name_round_trips(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "reports/2026/q1.pdf", b"pdf")
    assert await blob.adownload(container, "reports/2026/q1.pdf") == b"pdf"


@pytest.mark.parametrize(
    "name",
    ["with space.txt", "hash#tag.txt", "qmark?q.txt", "plus+sign.txt", "ünï.txt"],
)
async def test_a_blob_name_needing_escaping_round_trips(
    blob: AzureBlobService, container: str, name: str
) -> None:
    await blob.aupload(container, name, b"payload")
    assert await blob.adownload(container, name) == b"payload"


# Content type and metadata


async def test_the_content_type_is_stored_with_the_blob(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "page.html", b"<p>hi</p>", "text/html")
    assert (await blob.aget_properties(container, "page.html")).content_type == (
        "text/html"
    )


async def test_a_blob_uploaded_without_a_content_type_gets_the_opaque_one(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "thing.bin", b"bytes")
    assert (await blob.aget_properties(container, "thing.bin")).content_type == (
        "application/octet-stream"
    )


async def test_metadata_is_stored_with_the_blob(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(
        container, "owned.txt", b"bytes", metadata={"owner": "ada", "tier": "gold"}
    )
    properties = await blob.aget_properties(container, "owned.txt")
    assert properties.metadata == {"owner": "ada", "tier": "gold"}


async def test_a_blob_stored_without_metadata_comes_back_with_an_empty_mapping(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "bare.txt", b"bytes")
    assert (await blob.aget_properties(container, "bare.txt")).metadata == {}


async def test_an_overwrite_that_names_no_metadata_leaves_none_behind(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"first", metadata={"owner": "ada"})
    await blob.aupload(container, "note.txt", b"second")
    assert (await blob.aget_properties(container, "note.txt")).metadata == {}


async def test_properties_report_the_name_and_the_size(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "sized.txt", b"0123456789")
    properties = await blob.aget_properties(container, "sized.txt")
    assert properties.name == "sized.txt"
    assert properties.size == 10
    assert properties.etag
    assert properties.last_modified


# Overwrite


async def test_uploading_again_replaces_what_was_there(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"first")
    await blob.aupload(container, "note.txt", b"second")
    assert await blob.adownload(container, "note.txt") == b"second"


async def test_uploading_without_overwrite_onto_an_existing_blob_is_a_conflict(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"first")
    with pytest.raises(BlobConflictError):
        await blob.aupload(container, "note.txt", b"second", overwrite=False)


async def test_a_refused_upload_leaves_the_blob_it_refused_to_replace(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"first")
    with pytest.raises(BlobConflictError):
        await blob.aupload(container, "note.txt", b"second", overwrite=False)
    assert await blob.adownload(container, "note.txt") == b"first"


async def test_uploading_without_overwrite_to_a_free_name_is_allowed(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "fresh.txt", b"bytes", overwrite=False)
    assert await blob.adownload(container, "fresh.txt") == b"bytes"


# Existence and deletion


async def test_a_blob_exists_once_it_is_uploaded(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"bytes")
    assert await blob.aexists(container, "note.txt") is True


async def test_a_blob_that_was_never_uploaded_does_not_exist(
    blob: AzureBlobService, container: str
) -> None:
    assert await blob.aexists(container, "never.txt") is False


async def test_a_blob_in_a_container_that_is_not_there_does_not_exist(
    blob: AzureBlobService, unique: str
) -> None:
    assert await blob.aexists(f"missing{unique}", "note.txt") is False


async def test_a_deleted_blob_stops_existing(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"bytes")
    await blob.adelete(container, "note.txt")
    assert await blob.aexists(container, "note.txt") is False


async def test_deleting_one_blob_leaves_the_others(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "keep.txt", b"keep")
    await blob.aupload(container, "drop.txt", b"drop")
    await blob.adelete(container, "drop.txt")
    assert await blob.aexists(container, "keep.txt") is True


# Sync counterparts


def test_the_sync_path_round_trips_the_same_bytes(
    blob: AzureBlobService, container: str
) -> None:
    blob.upload(container, "note.txt", b"hello")
    assert blob.download(container, "note.txt") == b"hello"


def test_the_sync_path_stores_a_stream(blob: AzureBlobService, container: str) -> None:
    blob.upload(container, "streamed.bin", io.BytesIO(b"from a stream"))  # type: ignore[arg-type]
    assert blob.download(container, "streamed.bin") == b"from a stream"


def test_the_sync_path_stores_the_content_type_and_the_metadata(
    blob: AzureBlobService, container: str
) -> None:
    blob.upload(
        container, "page.html", b"<p>hi</p>", "text/html", metadata={"owner": "ada"}
    )
    properties = blob.get_properties(container, "page.html")
    assert properties.content_type == "text/html"
    assert properties.metadata == {"owner": "ada"}


def test_the_sync_path_refuses_an_upload_that_would_overwrite(
    blob: AzureBlobService, container: str
) -> None:
    blob.upload(container, "note.txt", b"first")
    with pytest.raises(BlobConflictError):
        blob.upload(container, "note.txt", b"second", overwrite=False)


def test_the_sync_path_deletes_a_blob(blob: AzureBlobService, container: str) -> None:
    blob.upload(container, "note.txt", b"bytes")
    assert blob.exists(container, "note.txt") is True
    blob.delete(container, "note.txt")
    assert blob.exists(container, "note.txt") is False


async def test_a_blob_written_by_the_sync_half_is_read_by_the_async_one(
    blob: AzureBlobService, container: str
) -> None:
    blob.upload(container, "shared.txt", b"bytes")
    assert await blob.adownload(container, "shared.txt") == b"bytes"
