"""`InMemoryBlobService`: the development stand-in, and where it differs.

It is the only blob backend that raises the standard library's errors rather
than the blob family, because it never talks to a provider. Pinned so a test
written against it is not surprised by the Azure one.
"""

from collections.abc import Iterator

import pytest

from base.models.view_models.blob import BlobProperties
from base.services.blob import AbstractBlobService
from base.services.blob.memory import InMemoryBlobService
from base.services.common.meta import purge_all


@pytest.fixture
def store() -> Iterator[InMemoryBlobService]:
    """A service with nothing in it, not shared with the next test."""
    purge_all()
    service = InMemoryBlobService()
    yield service
    service.dispose()
    purge_all()


def test_the_service_implements_the_neutral_contract(
    store: InMemoryBlobService,
) -> None:
    assert isinstance(store, AbstractBlobService)
    assert store.get_service_name() == "In-Memory Blob Storage"


def test_a_container_exists_only_after_it_is_created(
    store: InMemoryBlobService,
) -> None:
    assert store.container_exists("reports") is False
    store.create_container("reports")
    assert store.container_exists("reports") is True


def test_creating_a_container_twice_does_not_clear_it(
    store: InMemoryBlobService,
) -> None:
    store.create_container("reports")
    store.upload("reports", "q1.pdf", b"%PDF")
    store.create_container("reports")
    assert store.download("reports", "q1.pdf") == b"%PDF"


def test_deleting_a_container_takes_its_blobs_with_it(
    store: InMemoryBlobService,
) -> None:
    store.create_container("reports")
    store.upload("reports", "q1.pdf", b"%PDF")
    store.delete_container("reports")
    assert store.container_exists("reports") is False
    assert store.exists("reports", "q1.pdf") is False


def test_deleting_a_container_that_is_not_there_is_allowed(
    store: InMemoryBlobService,
) -> None:
    store.delete_container("never-created")


def test_an_uploaded_blob_comes_back_byte_for_byte(
    store: InMemoryBlobService,
) -> None:
    store.create_container("reports")
    store.upload("reports", "q1.pdf", b"\x00\xff binary")
    assert store.download("reports", "q1.pdf") == b"\x00\xff binary"


def test_uploading_over_a_blob_replaces_it(store: InMemoryBlobService) -> None:
    store.create_container("reports")
    store.upload("reports", "q1.pdf", b"first")
    store.upload("reports", "q1.pdf", b"second")
    assert store.download("reports", "q1.pdf") == b"second"


def test_refusing_to_overwrite_raises_rather_than_replacing(
    store: InMemoryBlobService,
) -> None:
    store.create_container("reports")
    store.upload("reports", "q1.pdf", b"first")
    with pytest.raises(FileExistsError, match="already exists"):
        store.upload("reports", "q1.pdf", b"second", overwrite=False)
    assert store.download("reports", "q1.pdf") == b"first"


def test_uploading_to_a_container_that_does_not_exist_is_refused(
    store: InMemoryBlobService,
) -> None:
    with pytest.raises(FileNotFoundError, match="Container 'missing' does not exist"):
        store.upload("missing", "q1.pdf", b"x")


def test_downloading_a_blob_that_is_not_there_is_refused(
    store: InMemoryBlobService,
) -> None:
    store.create_container("reports")
    with pytest.raises(FileNotFoundError, match="Blob 'q1.pdf' not found"):
        store.download("reports", "q1.pdf")


def test_deleting_a_blob_that_is_not_there_is_allowed(
    store: InMemoryBlobService,
) -> None:
    store.create_container("reports")
    store.delete("reports", "never-uploaded")


def test_existence_of_a_blob_in_a_missing_container_is_false_not_an_error(
    store: InMemoryBlobService,
) -> None:
    assert store.exists("missing", "q1.pdf") is False


def test_properties_describe_the_bytes_that_were_stored(
    store: InMemoryBlobService,
) -> None:
    store.create_container("reports")
    store.upload(
        "reports",
        "q1.pdf",
        b"%PDF",
        content_type="application/pdf",
        metadata={"owner": "ada"},
    )
    props = store.get_properties("reports", "q1.pdf")
    assert isinstance(props, BlobProperties)
    assert props.size == 4
    assert props.content_type == "application/pdf"
    assert props.metadata == {"owner": "ada"}
    assert props.etag is None
    assert props.last_modified is not None


def test_listing_returns_every_blob_in_the_container(
    store: InMemoryBlobService,
) -> None:
    store.create_container("reports")
    store.upload("reports", "2026/q1.pdf", b"a")
    store.upload("reports", "2025/q4.pdf", b"b")
    assert {b.name for b in store.list_blobs("reports")} == {
        "2026/q1.pdf",
        "2025/q4.pdf",
    }


def test_listing_with_a_prefix_returns_only_what_starts_with_it(
    store: InMemoryBlobService,
) -> None:
    store.create_container("reports")
    store.upload("reports", "2026/q1.pdf", b"a")
    store.upload("reports", "2025/q4.pdf", b"b")
    assert [b.name for b in store.list_blobs("reports", prefix="2026/")] == [
        "2026/q1.pdf"
    ]


def test_listing_a_missing_container_is_refused_when_iterated(
    store: InMemoryBlobService,
) -> None:
    """A generator again: nothing happens until the caller asks for an item."""
    listing = store.list_blobs("missing")
    with pytest.raises(FileNotFoundError):
        next(listing)


def test_a_signed_url_is_a_memory_scheme_placeholder(
    store: InMemoryBlobService,
) -> None:
    assert store.generate_url("reports", "q1.pdf") == "memory://reports/q1.pdf"


def test_disposing_empties_every_container(store: InMemoryBlobService) -> None:
    store.create_container("reports")
    store.upload("reports", "q1.pdf", b"x")
    store.dispose()
    assert store.container_exists("reports") is False


async def test_the_async_methods_reach_the_same_store(
    store: InMemoryBlobService,
) -> None:
    await store.acreate_container("reports")
    await store.aupload("reports", "q1.pdf", b"async bytes")
    assert await store.acontainer_exists("reports") is True
    assert await store.adownload("reports", "q1.pdf") == b"async bytes"
    assert await store.aexists("reports", "q1.pdf") is True
    assert (await store.aget_properties("reports", "q1.pdf")).size == 11
    assert [b.name async for b in store.alist_blobs("reports")] == ["q1.pdf"]
    assert await store.agenerate_url("reports", "q1.pdf") == "memory://reports/q1.pdf"
    await store.adelete("reports", "q1.pdf")
    assert await store.aexists("reports", "q1.pdf") is False
    await store.adispose()
    assert await store.acontainer_exists("reports") is False
