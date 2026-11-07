"""`AzureBlobClient`: lazy SDK clients, and the configuration it insists on.

The client is the only place `azure_blob_account_url` is read, and the only
place SDK-level retries are turned off — two layers of backoff would multiply
the wait, because the error handler already owns the retry policy.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from azure.core.exceptions import ServiceRequestError

from base.clients.blob.azure_blob import AzureBlobClient
from base.models.view_models.exceptions.blob import BlobConfigError
from base.services.common.meta import Singleton, purge_all
from base.services.configuration.mixins import AzureBlobSettingsMixin

ACCOUNT_URL = "https://account.blob.core.windows.net/"


@pytest.fixture(autouse=True)
def _purge_instances() -> Iterator[None]:
    """The client is a singleton, so one test's settings must not reach the next."""
    purge_all()
    yield
    purge_all()


def test_the_account_url_is_read_once_from_settings(make_settings: Any) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    assert AzureBlobClient()._account_url == ACCOUNT_URL


def test_an_unset_account_url_is_refused_as_a_configuration_error(
    make_settings: Any,
) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url="")
    with pytest.raises(BlobConfigError, match="azure_blob_account_url is required"):
        AzureBlobClient()


def test_no_sdk_client_is_built_until_one_is_asked_for(make_settings: Any) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    client = AzureBlobClient()
    assert client._sync_client is None
    assert client._async_client is None
    assert client._credential is None
    assert client._async_credential is None


def test_the_client_is_one_object_for_the_whole_process(make_settings: Any) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    assert AzureBlobClient() is AzureBlobClient()
    assert type(AzureBlobClient) is Singleton


def test_sdk_retries_are_off_because_the_handler_owns_them(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    built: dict[str, Any] = {}

    class FakeServiceClient:
        def __init__(self, **kwargs: Any) -> None:
            built.update(kwargs)

    monkeypatch.setattr(
        "base.clients.blob.azure_blob.BlobServiceClient", FakeServiceClient
    )
    monkeypatch.setattr(
        "base.clients.blob.azure_blob.DefaultAzureCredential", lambda: "credential"
    )

    client = AzureBlobClient()
    assert isinstance(client.sync_client, FakeServiceClient)
    assert built["retry_total"] == 0
    assert built["account_url"] == ACCOUNT_URL
    assert built["credential"] == "credential"


def test_the_sdk_client_is_built_once_and_then_reused(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    builds = {"count": 0}

    class FakeServiceClient:
        def __init__(self, **kwargs: Any) -> None:
            builds["count"] += 1

    monkeypatch.setattr(
        "base.clients.blob.azure_blob.BlobServiceClient", FakeServiceClient
    )
    monkeypatch.setattr(
        "base.clients.blob.azure_blob.DefaultAzureCredential", lambda: None
    )

    client = AzureBlobClient()
    assert client.sync_client is client.sync_client
    assert builds["count"] == 1


def test_the_async_client_is_built_with_retries_off_too(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    built: dict[str, Any] = {}

    class FakeAsyncServiceClient:
        def __init__(self, **kwargs: Any) -> None:
            built.update(kwargs)

    monkeypatch.setattr(
        "base.clients.blob.azure_blob.AsyncBlobServiceClient", FakeAsyncServiceClient
    )
    monkeypatch.setattr(
        "base.clients.blob.azure_blob.AsyncDefaultAzureCredential", lambda: None
    )

    assert isinstance(AzureBlobClient().async_client, FakeAsyncServiceClient)
    assert built["retry_total"] == 0


def test_a_reachable_account_pings_true(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    client = AzureBlobClient()

    class Reachable:
        def get_account_information(self) -> dict[str, str]:
            return {"sku_name": "Standard_LRS"}

    monkeypatch.setattr(type(client), "sync_client", property(lambda _: Reachable()))
    assert client.ping() is True


def test_an_unreachable_account_pings_false_rather_than_raising(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    client = AzureBlobClient()

    class Unreachable:
        def get_account_information(self) -> dict[str, str]:
            raise ServiceRequestError("no route to host")

    monkeypatch.setattr(type(client), "sync_client", property(lambda _: Unreachable()))
    assert client.ping() is False


async def test_an_unreachable_account_pings_false_asynchronously(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    client = AzureBlobClient()

    class Unreachable:
        async def get_account_information(self) -> dict[str, str]:
            raise ServiceRequestError("no route to host")

    monkeypatch.setattr(type(client), "async_client", property(lambda _: Unreachable()))
    assert await client.aping() is False


def test_closing_releases_the_client_and_the_credential(make_settings: Any) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    closed: list[str] = []

    class Closeable:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    client = AzureBlobClient()
    client._sync_client = Closeable("client")  # type: ignore[assignment]
    client._credential = Closeable("credential")  # type: ignore[assignment]
    client.close()
    assert closed == ["client", "credential"]
    assert client._sync_client is None
    assert client._credential is None


def test_closing_a_client_that_was_never_opened_is_allowed(make_settings: Any) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    AzureBlobClient().close()


async def test_closing_everything_releases_both_halves(make_settings: Any) -> None:
    make_settings(AzureBlobSettingsMixin, azure_blob_account_url=ACCOUNT_URL)
    closed: list[str] = []

    class SyncCloseable:
        def close(self) -> None:
            closed.append("sync")

    class AsyncCloseable:
        async def close(self) -> None:
            closed.append("async")

    client = AzureBlobClient()
    client._sync_client = SyncCloseable()  # type: ignore[assignment]
    client._async_client = AsyncCloseable()  # type: ignore[assignment]
    await client.close_all()
    assert closed == ["sync", "async"]
    assert client._sync_client is None
    assert client._async_client is None
