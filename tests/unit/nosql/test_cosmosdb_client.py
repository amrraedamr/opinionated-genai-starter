"""What the Cosmos client requires before it will build, and what it builds with.

The account URI is mandatory and the account key is not: without a key the
client falls back to `DefaultAzureCredential`, which is the managed-identity
path a deployed service uses. No Cosmos account is reachable here, so the
choice is pinned by watching which credential the SDK constructor is handed.
"""

from typing import Any

import pytest

from base.clients.nosql.cosmosdb import CosmosDBClient
from base.services.configuration.mixins import CosmosSettingsMixin


class RecordingSdk:
    """Stands in for `CosmosClient`, keeping the arguments it was built with."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed: int = 0

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self

    def get_database_client(self, database: str) -> Any:
        return {"database": database, "container": None}

    def close(self) -> None:
        self.closed += 1

    @property
    def last(self) -> dict[str, Any]:
        assert self.calls, "the SDK client was never built"
        return self.calls[-1]


class RecordingCredential:
    """Stands in for `DefaultAzureCredential`."""

    def __init__(self) -> None:
        self.built = 0
        self.closed = 0

    def __call__(self) -> "RecordingCredential":
        self.built += 1
        return self

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> RecordingSdk:
    recorder = RecordingSdk()
    monkeypatch.setattr("base.clients.nosql.cosmosdb.CosmosClient", recorder)
    monkeypatch.setattr("base.clients.nosql.cosmosdb.AsyncCosmosClient", recorder)
    return recorder


@pytest.fixture
def credential(monkeypatch: pytest.MonkeyPatch) -> RecordingCredential:
    recorder = RecordingCredential()
    monkeypatch.setattr("base.clients.nosql.cosmosdb.DefaultAzureCredential", recorder)
    monkeypatch.setattr(
        "base.clients.nosql.cosmosdb.AsyncDefaultAzureCredential", recorder
    )
    return recorder


def configure(make_settings: Any, **values: Any) -> Any:
    values.setdefault("azure_cosmos_account_uri", "https://account.documents.azure.com")
    values.setdefault("azure_cosmos_database_name", "app")
    return make_settings(CosmosSettingsMixin, **values)


def test_an_unset_account_uri_is_refused_at_construction(make_settings: Any) -> None:
    configure(make_settings, azure_cosmos_account_uri="")
    with pytest.raises(ValueError, match="azure_cosmos_account_uri"):
        CosmosDBClient()


def test_constructing_the_client_builds_no_sdk_client(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    configure(make_settings)
    CosmosDBClient()
    assert sdk.calls == []
    assert credential.built == 0


def test_an_account_key_is_used_as_the_credential(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    configure(make_settings, azure_cosmos_account_key="the-key")
    CosmosDBClient().sync_client
    assert sdk.last["credential"] == "the-key"


def test_an_account_key_builds_no_identity_at_all(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    """The identity needs aiohttp on the async side, so a keyed deployment
    must not reach for one it will never use."""
    configure(make_settings, azure_cosmos_account_key="the-key")
    CosmosDBClient().sync_client
    assert credential.built == 0


def test_no_account_key_falls_back_to_the_managed_identity(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    configure(make_settings)
    CosmosDBClient().sync_client
    assert sdk.last["credential"] is credential
    assert credential.built == 1


def test_an_empty_account_key_counts_as_no_key(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    configure(make_settings, azure_cosmos_account_key="")
    CosmosDBClient().sync_client
    assert sdk.last["credential"] is credential


def test_the_account_uri_reaches_the_sdk(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    configure(make_settings, azure_cosmos_account_uri="https://a.documents.azure.com")
    CosmosDBClient().sync_client
    assert sdk.last["url"] == "https://a.documents.azure.com"


def test_the_sync_client_is_built_once_and_reused(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    configure(make_settings)
    client = CosmosDBClient()
    assert client.sync_client is client.sync_client
    assert len(sdk.calls) == 1


def test_the_async_client_is_built_separately(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    configure(make_settings)
    client = CosmosDBClient()
    client.sync_client
    client.async_client
    assert len(sdk.calls) == 2


def test_closing_drops_the_sync_client_so_the_next_use_rebuilds_it(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    configure(make_settings)
    client = CosmosDBClient()
    client.sync_client
    client.close()
    client.sync_client
    assert len(sdk.calls) == 2
    assert credential.closed == 1


def test_closing_closes_the_sync_client_rather_than_leaving_it_to_the_collector(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    """`CosmosClient.close()` is public, and holds a transport until it runs."""
    configure(make_settings, azure_cosmos_account_key="the-key")
    client = CosmosDBClient()
    client.sync_client
    client.close()
    assert sdk.closed == 1


def test_closing_a_client_that_never_built_one_closes_nothing(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    configure(make_settings, azure_cosmos_account_key="the-key")
    CosmosDBClient().close()
    assert sdk.closed == 0


def test_the_client_is_one_instance_for_the_process(
    make_settings: Any, sdk: RecordingSdk, credential: RecordingCredential
) -> None:
    configure(make_settings)
    assert CosmosDBClient() is CosmosDBClient()
