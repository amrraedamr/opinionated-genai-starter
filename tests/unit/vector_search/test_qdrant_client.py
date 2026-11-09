"""What the Qdrant client hands the SDK, and when it builds one at all.

gRPC is preferred by default, which means a point id the HTTP API would have
tolerated is parsed strictly, so the transport choice is part of the contract
rather than a detail. The api key is sent only when there is one, so a local
container with authentication off is not handed an empty credential.
"""

from typing import Any

import pytest

from base.clients.vectors.qdrant import QdrantClient
from base.services.configuration.mixins import QdrantSettingsMixin
from tests.conftest import MakeSettings


class RecordingSdk:
    """Stands in for `QdrantClient` / `AsyncQdrantClient`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = 0

    def __call__(self, **kwargs: Any) -> "RecordingSdk":
        self.calls.append(kwargs)
        return self

    def close(self) -> None:
        self.closed += 1

    @property
    def last(self) -> dict[str, Any]:
        assert self.calls, "the SDK client was never built"
        return self.calls[-1]


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> RecordingSdk:
    recorder = RecordingSdk()
    monkeypatch.setattr("base.clients.vectors.qdrant.QdrantSyncClient", recorder)
    monkeypatch.setattr("base.clients.vectors.qdrant.QdrantAsyncClient", recorder)
    return recorder


def configure(make_settings: MakeSettings, **values: Any) -> Any:
    values.setdefault("qdrant_url", "http://localhost:6333")
    return make_settings(QdrantSettingsMixin, **values)


def test_constructing_the_client_builds_no_sdk_client(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings)
    QdrantClient()
    assert sdk.calls == []


def test_the_url_reaches_the_sdk(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings, qdrant_url="http://qdrant:6333")
    QdrantClient().sync_client
    assert sdk.last["url"] == "http://qdrant:6333"


def test_grpc_is_preferred_by_default(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings)
    QdrantClient().sync_client
    assert sdk.last["prefer_grpc"] is True
    assert sdk.last["grpc_port"] == 6334


def test_http_can_be_chosen_instead(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings, qdrant_prefer_grpc=False)
    QdrantClient().sync_client
    assert sdk.last["prefer_grpc"] is False


def test_a_custom_grpc_port_reaches_the_sdk(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings, qdrant_grpc_port=7334)
    QdrantClient().sync_client
    assert sdk.last["grpc_port"] == 7334


def test_the_timeout_reaches_the_sdk(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings, qdrant_timeout=5)
    QdrantClient().sync_client
    assert sdk.last["timeout"] == 5


def test_the_default_timeout_is_thirty_seconds(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings)
    QdrantClient().sync_client
    assert sdk.last["timeout"] == 30


def test_an_api_key_is_sent_when_there_is_one(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings, qdrant_api_key="the-key")
    QdrantClient().sync_client
    assert sdk.last["api_key"] == "the-key"


def test_no_api_key_is_sent_when_there_is_none(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    """An empty credential is worse than no credential to a local container."""
    configure(make_settings)
    QdrantClient().sync_client
    assert "api_key" not in sdk.last


def test_the_sync_client_is_built_once_and_reused(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings)
    client = QdrantClient()
    assert client.sync_client is client.sync_client
    assert len(sdk.calls) == 1


def test_the_async_client_is_built_separately(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings)
    client = QdrantClient()
    client.sync_client
    client.async_client
    assert len(sdk.calls) == 2


def test_both_clients_are_built_from_the_same_settings(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings, qdrant_api_key="k", qdrant_timeout=11)
    client = QdrantClient()
    client.sync_client
    client.async_client
    assert sdk.calls[0] == sdk.calls[1]


def test_closing_drops_the_sync_client_so_the_next_use_rebuilds_it(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings)
    client = QdrantClient()
    client.sync_client
    client.close()
    client.sync_client
    assert sdk.closed == 1
    assert len(sdk.calls) == 2


def test_closing_an_unused_client_does_nothing(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings)
    QdrantClient().close()
    assert sdk.closed == 0


def test_the_client_is_one_instance_for_the_process(
    make_settings: MakeSettings, sdk: RecordingSdk
) -> None:
    configure(make_settings)
    assert QdrantClient() is QdrantClient()
