"""The SAS a shared key signs, and whether the server honours it.

A user delegation key only exists on Entra ID, so the emulator exercises the
other branch: the key signs the token itself. The token is signed over the
unescaped blob name while the URL must carry the escaped one, which is the
part a fake can never check — only a server can say whether the address
reaches the blob.
"""

import httpx
import pytest

from base.services.blob.azure_blob import AzureBlobService

pytestmark = pytest.mark.integration


async def test_a_generated_url_fetches_the_bytes_of_the_blob(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"hello")
    url = await blob.agenerate_url(container, "note.txt")
    response = httpx.get(url)
    assert response.status_code == 200
    assert response.content == b"hello"


async def test_a_generated_url_points_at_the_account_and_carries_a_signature(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"hello")
    url = await blob.agenerate_url(container, "note.txt")
    assert url.startswith(f"http://127.0.0.1:10100/devstoreaccount1/{container}/")
    assert "sig=" in url
    assert "se=" in url


@pytest.mark.parametrize(
    "name",
    [
        "with space.txt",
        "hash#tag.txt",
        "qmark?q.txt",
        "plus+sign.txt",
        "reports/2026/q1.pdf",
        "ünï.txt",
    ],
)
async def test_a_url_for_a_name_that_needs_escaping_still_fetches_the_blob(
    blob: AzureBlobService, container: str, name: str
) -> None:
    await blob.aupload(container, name, b"payload")
    response = httpx.get(await blob.agenerate_url(container, name))
    assert response.status_code == 200
    assert response.content == b"payload"


async def test_a_url_whose_expiry_has_passed_is_refused(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"hello")
    url = await blob.agenerate_url(container, "note.txt", expiry_seconds=-60)
    assert httpx.get(url).status_code == 403


async def test_a_generated_url_only_grants_a_read(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"hello")
    url = await blob.agenerate_url(container, "note.txt")
    response = httpx.put(
        url, content=b"tampered", headers={"x-ms-blob-type": "BlockBlob"}
    )
    assert response.status_code == 403
    assert await blob.adownload(container, "note.txt") == b"hello"


async def test_a_url_for_a_blob_that_is_not_there_signs_but_finds_nothing(
    blob: AzureBlobService, container: str
) -> None:
    url = await blob.agenerate_url(container, "never.txt")
    assert httpx.get(url).status_code == 404


async def test_a_url_signed_for_one_blob_does_not_open_another(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "mine.txt", b"mine")
    await blob.aupload(container, "yours.txt", b"yours")
    url = await blob.agenerate_url(container, "mine.txt")
    assert httpx.get(url.replace("mine.txt", "yours.txt")).status_code == 403


# Sync counterpart


def test_the_sync_path_generates_a_url_that_fetches_the_blob(
    blob: AzureBlobService, container: str
) -> None:
    blob.upload(container, "note.txt", b"hello")
    response = httpx.get(blob.generate_url(container, "note.txt"))
    assert response.status_code == 200
    assert response.content == b"hello"


def test_the_sync_path_escapes_the_blob_name_too(
    blob: AzureBlobService, container: str
) -> None:
    blob.upload(container, "with space.txt", b"payload")
    response = httpx.get(blob.generate_url(container, "with space.txt"))
    assert response.status_code == 200
    assert response.content == b"payload"


def test_the_sync_path_refuses_a_url_whose_expiry_has_passed(
    blob: AzureBlobService, container: str
) -> None:
    blob.upload(container, "note.txt", b"hello")
    url = blob.generate_url(container, "note.txt", expiry_seconds=-60)
    assert httpx.get(url).status_code == 403
