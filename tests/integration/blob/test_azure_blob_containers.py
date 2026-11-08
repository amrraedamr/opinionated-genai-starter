"""Containers against a real server: creating, finding and deleting them.

`create_container` asks whether the container is there before creating it, so
the method is two round trips and a name the server refuses fails on the first
of them. Azure wants 3 to 63 lowercase characters, neither end a hyphen.
"""

import pytest

from base.models.view_models.exceptions.blob import (
    BlobBadRequestError,
    BlobNotFoundError,
)
from base.services.blob.azure_blob import AzureBlobService

pytestmark = pytest.mark.integration


async def test_a_container_is_there_once_it_is_created(
    blob: AzureBlobService, unique: str
) -> None:
    name = f"t{unique}"
    await blob.acreate_container(name)
    assert await blob.acontainer_exists(name) is True
    await blob.adelete_container(name)


async def test_a_container_that_was_never_created_does_not_exist(
    blob: AzureBlobService, unique: str
) -> None:
    assert await blob.acontainer_exists(f"missing{unique}") is False


async def test_creating_a_container_that_is_already_there_is_not_an_error(
    blob: AzureBlobService, container: str
) -> None:
    await blob.acreate_container(container)
    assert await blob.acontainer_exists(container) is True


async def test_a_deleted_container_stops_existing(
    blob: AzureBlobService, unique: str
) -> None:
    name = f"t{unique}"
    await blob.acreate_container(name)
    await blob.adelete_container(name)
    assert await blob.acontainer_exists(name) is False


async def test_deleting_a_container_takes_its_blobs_with_it(
    blob: AzureBlobService, unique: str
) -> None:
    name = f"t{unique}"
    await blob.acreate_container(name)
    await blob.aupload(name, "doomed.txt", b"bytes")
    await blob.adelete_container(name)
    await blob.acreate_container(name)
    assert [b.name async for b in blob.alist_blobs(name)] == []
    await blob.adelete_container(name)


async def test_deleting_a_container_that_is_not_there_is_reported_as_missing(
    blob: AzureBlobService, unique: str
) -> None:
    with pytest.raises(BlobNotFoundError):
        await blob.adelete_container(f"missing{unique}")


@pytest.mark.parametrize("name", ["ab", "a" * 64])
async def test_a_container_name_azure_refuses_is_reported_as_a_bad_request(
    blob: AzureBlobService, name: str
) -> None:
    with pytest.raises(BlobBadRequestError):
        await blob.acreate_container(name)


async def test_asking_whether_a_badly_named_container_exists_is_a_bad_request(
    blob: AzureBlobService,
) -> None:
    with pytest.raises(BlobBadRequestError):
        await blob.acontainer_exists("ab")


# Sync counterparts


def test_the_sync_path_creates_finds_and_deletes_a_container(
    blob: AzureBlobService, unique: str
) -> None:
    name = f"t{unique}"
    blob.create_container(name)
    assert blob.container_exists(name) is True
    blob.delete_container(name)
    assert blob.container_exists(name) is False


def test_the_sync_path_tolerates_creating_a_container_twice(
    blob: AzureBlobService, unique: str
) -> None:
    name = f"t{unique}"
    blob.create_container(name)
    blob.create_container(name)
    assert blob.container_exists(name) is True
    blob.delete_container(name)


def test_the_sync_path_reports_a_missing_container_on_delete(
    blob: AzureBlobService, unique: str
) -> None:
    with pytest.raises(BlobNotFoundError):
        blob.delete_container(f"missing{unique}")


def test_the_sync_path_refuses_a_name_azure_will_not_take(
    blob: AzureBlobService,
) -> None:
    with pytest.raises(BlobBadRequestError):
        blob.create_container("ab")
