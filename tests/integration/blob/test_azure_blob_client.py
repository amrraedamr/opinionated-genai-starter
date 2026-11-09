"""`AzureBlobClient` against a real account: reachability and lifecycle.

A shared key is the branch an emulator can answer, and the one a fake can only
assert on: here the key actually authenticates. Closing is lazy in reverse —
the next call rebuilds the SDK client it dropped.
"""

from typing import Any

import pytest

from base.clients.blob.azure_blob import AzureBlobClient
from base.services.blob.azure_blob import AzureBlobService
from base.services.configuration.mixins.blob.azure import AzureBlobSettingsMixin
from tests.conftest import MakeSettings

from .conftest import ACCOUNT_KEY

pytestmark = pytest.mark.integration


def test_a_shared_key_reaches_the_account_without_an_identity(azurite: Any) -> None:
    client = AzureBlobClient()
    assert client.ping() is True
    assert client._credential is None
    client.close()


async def test_a_shared_key_reaches_the_account_asynchronously(azurite: Any) -> None:
    client = AzureBlobClient()
    assert await client.aping() is True
    assert client._async_credential is None
    await client.aclose()


def test_an_account_nothing_is_listening_on_pings_false(
    make_settings: MakeSettings,
) -> None:
    make_settings(
        AzureBlobSettingsMixin,
        azure_blob_account_url="http://127.0.0.1:10199/devstoreaccount1",
        azure_blob_account_key=ACCOUNT_KEY,
    )
    client = AzureBlobClient()
    assert client.ping() is False
    client.close()


async def test_an_account_nothing_is_listening_on_pings_false_asynchronously(
    make_settings: MakeSettings,
) -> None:
    make_settings(
        AzureBlobSettingsMixin,
        azure_blob_account_url="http://127.0.0.1:10199/devstoreaccount1",
        azure_blob_account_key=ACCOUNT_KEY,
    )
    client = AzureBlobClient()
    assert await client.aping() is False
    await client.aclose()


def test_the_account_name_is_the_one_in_the_account_url(azurite: Any) -> None:
    assert AzureBlobClient().sync_client.account_name == "devstoreaccount1"


async def test_the_sync_and_async_halves_are_separate_clients(azurite: Any) -> None:
    client = AzureBlobClient()
    assert client.sync_client is not client.async_client
    client.close()
    await client.aclose()


# Lifecycle


async def test_disposing_drops_the_clients_the_service_was_holding(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"bytes")
    blob.dispose()
    await blob.adispose()
    assert blob._client._sync_client is None
    assert blob._client._async_client is None


async def test_a_call_after_disposing_rebuilds_the_client_and_works(
    blob: AzureBlobService, container: str
) -> None:
    await blob.aupload(container, "note.txt", b"bytes")
    blob.dispose()
    await blob.adispose()
    assert await blob.adownload(container, "note.txt") == b"bytes"
    assert blob.download(container, "note.txt") == b"bytes"


async def test_disposing_twice_is_allowed(blob: AzureBlobService) -> None:
    blob.dispose()
    blob.dispose()
    await blob.adispose()
    await blob.adispose()


async def test_closing_the_sync_half_leaves_the_async_one_alone(azurite: Any) -> None:
    client = AzureBlobClient()
    assert client.sync_client is not None
    assert client.async_client is not None
    client.close()
    assert client._sync_client is None
    assert client._async_client is not None
    await client.aclose()


def test_the_service_names_itself(blob: AzureBlobService) -> None:
    assert blob.get_service_name() == "Azure Blob Storage"
