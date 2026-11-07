"""What the Mongo client hands the driver, and when it builds one at all.

The pool sizes, the timeouts and the two retry switches are the whole of the
client's contract with pymongo, and the retry switches encode a decision made
above the driver: single-document writes are repeated by the driver, which can
recognise a repeat, and reads are repeated by the error handler instead.
"""

from typing import Any

import pytest

from base.clients.nosql.mongodb import MongoDBClient
from base.services.configuration.mixins import MongoSettingsMixin


class RecordingDriver:
    """Stands in for `MongoClient` / `AsyncMongoClient`, keeping the call."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self

    def close(self) -> None:
        return None

    @property
    def last_kwargs(self) -> dict[str, Any]:
        assert self.calls, "the driver was never constructed"
        return self.calls[-1][1]


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> RecordingDriver:
    """Both driver entry points replaced by one recorder."""
    recorder = RecordingDriver()
    monkeypatch.setattr("base.clients.nosql.mongodb.MongoClient", recorder)
    monkeypatch.setattr("base.clients.nosql.mongodb.AsyncMongoClient", recorder)
    return recorder


@pytest.fixture
def configured(make_settings: Any) -> Any:
    """Settings a Mongo client can be built from."""
    return make_settings(
        MongoSettingsMixin,
        mongodb_host="db:27017",
        mongodb_database="app",
        mongodb_min_pool_size=2,
        mongodb_max_pool_size=20,
        mongodb_connect_timeout_ms=1_111,
        mongodb_socket_timeout_ms=2_222,
        mongodb_server_selection_timeout_ms=3_333,
    )


def test_constructing_the_client_opens_no_connection(
    configured: Any, driver: RecordingDriver
) -> None:
    MongoDBClient()
    assert driver.calls == []


def test_the_pool_bounds_reach_the_driver(
    configured: Any, driver: RecordingDriver
) -> None:
    MongoDBClient().sync_client
    assert driver.last_kwargs["minPoolSize"] == 2
    assert driver.last_kwargs["maxPoolSize"] == 20


def test_every_timeout_reaches_the_driver(
    configured: Any, driver: RecordingDriver
) -> None:
    MongoDBClient().sync_client
    assert driver.last_kwargs["connectTimeoutMS"] == 1_111
    assert driver.last_kwargs["socketTimeoutMS"] == 2_222
    assert driver.last_kwargs["serverSelectionTimeoutMS"] == 3_333


def test_the_driver_keeps_its_own_write_retry(
    configured: Any, driver: RecordingDriver
) -> None:
    """Only the driver can send a transaction number, so it owns write retry."""
    MongoDBClient().sync_client
    assert driver.last_kwargs["retryWrites"] is True


def test_the_driver_does_not_retry_reads(
    configured: Any, driver: RecordingDriver
) -> None:
    """Read retry lives in the error handler, where the budget and logging are."""
    MongoDBClient().sync_client
    assert driver.last_kwargs["retryReads"] is False


def test_the_connection_string_is_the_first_positional_argument(
    configured: Any, driver: RecordingDriver
) -> None:
    MongoDBClient().sync_client
    assert driver.calls[-1][0] == ("mongodb://db:27017/",)


def test_the_sync_client_is_built_once_and_reused(
    configured: Any, driver: RecordingDriver
) -> None:
    client = MongoDBClient()
    assert client.sync_client is client.sync_client
    assert len(driver.calls) == 1


def test_the_async_client_is_built_once_and_reused(
    configured: Any, driver: RecordingDriver
) -> None:
    client = MongoDBClient()
    assert client.async_client is client.async_client
    assert len(driver.calls) == 1


def test_the_sync_and_async_clients_are_built_separately(
    configured: Any, driver: RecordingDriver
) -> None:
    client = MongoDBClient()
    client.sync_client
    client.async_client
    assert len(driver.calls) == 2


def test_closing_drops_the_sync_client_so_the_next_use_rebuilds_it(
    configured: Any, driver: RecordingDriver
) -> None:
    client = MongoDBClient()
    client.sync_client
    client.close()
    client.sync_client
    assert len(driver.calls) == 2


def test_closing_an_unused_client_does_nothing(
    configured: Any, driver: RecordingDriver
) -> None:
    MongoDBClient().close()
    assert driver.calls == []


def test_the_client_is_one_instance_for_the_process(
    configured: Any, driver: RecordingDriver
) -> None:
    assert MongoDBClient() is MongoDBClient()


def test_a_database_handle_is_taken_from_the_sync_client(
    configured: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    taken: list[str] = []

    class Handle:
        def get_database(self, database: str) -> str:
            taken.append(database)
            return f"db:{database}"

    client = MongoDBClient()
    monkeypatch.setattr(MongoDBClient, "sync_client", property(lambda _: Handle()))
    assert client.get_database("reports") == "db:reports"
    assert taken == ["reports"]
