"""Which Mongo calls are repeated after a transient failure, and which are not.

An update carries a statement the caller wrote. `$inc` applied twice is a
different document from `$inc` applied once, so the update methods go through
`classifying`, which runs the body exactly once and still raises the document
store's own error family. Reads and single-document writes the driver can
recognise as repeats go through `retrying` instead.

Only the repeat policy is pinned here; which server codes count as retryable
belongs to the error-handling tests.
"""

from typing import Any

import pytest
from pymongo.errors import AutoReconnect

from base.models.view_models.exceptions.nosql import NoSqlConnectionError
from base.services.configuration.mixins import MongoSettingsMixin
from base.services.error_handling import AbstractErrorHandler
from base.services.nosql.mongodb import MongoDBService
from tests.fakes import no_sleep

ATTEMPTS_WITH_RETRY = AbstractErrorHandler.DEFAULT_RETRIES + 1
"""Three retries plus the first attempt, from `AbstractErrorHandler`."""


class CountingDatabase:
    """A database handle whose every collection call fails transiently."""

    def __init__(self) -> None:
        self.attempts = 0
        self.updates: list[Any] = []

    def __getitem__(self, name: str) -> "CountingDatabase":
        del name
        return self

    def _fail(self, *args: Any, **kwargs: Any) -> Any:
        self.attempts += 1
        self.updates.append(args)
        raise AutoReconnect("primary stepped down")

    update_one = _fail
    update_many = _fail
    find_one = _fail
    insert_one = _fail


class AsyncCountingDatabase(CountingDatabase):
    """The async handle, failing the same way."""

    async def _afail(self, *args: Any, **kwargs: Any) -> Any:
        return CountingDatabase._fail(self, *args, **kwargs)

    update_one = _afail  # type: ignore[assignment]
    update_many = _afail  # type: ignore[assignment]
    find_one = _afail  # type: ignore[assignment]
    insert_one = _afail  # type: ignore[assignment]


@pytest.fixture
def service(make_settings: Any, monkeypatch: pytest.MonkeyPatch) -> MongoDBService:
    make_settings(MongoSettingsMixin, mongodb_host="db:27017", mongodb_database="app")
    no_sleep(monkeypatch)
    return MongoDBService()


@pytest.fixture
def failing(
    service: MongoDBService, monkeypatch: pytest.MonkeyPatch
) -> CountingDatabase:
    handle = CountingDatabase()
    monkeypatch.setattr(service, "_get_database", lambda: handle)
    return handle


@pytest.fixture
def afailing(
    service: MongoDBService, monkeypatch: pytest.MonkeyPatch
) -> AsyncCountingDatabase:
    handle = AsyncCountingDatabase()
    monkeypatch.setattr(service, "_aget_database", lambda: handle)
    return handle


async def test_an_inc_update_is_attempted_once_by_default(
    service: MongoDBService, afailing: AsyncCountingDatabase
) -> None:
    with pytest.raises(NoSqlConnectionError):
        await service.aupdate_one("counters", {"_id": "a"}, {"$inc": {"hits": 1}})
    assert afailing.attempts == 1


async def test_an_inc_update_is_repeated_only_when_the_caller_asks(
    service: MongoDBService, afailing: AsyncCountingDatabase
) -> None:
    with pytest.raises(NoSqlConnectionError):
        await service.aupdate_one(
            "counters", {"_id": "a"}, {"$inc": {"hits": 1}}, retry=True
        )
    assert afailing.attempts == ATTEMPTS_WITH_RETRY


async def test_aupdate_many_is_attempted_once_by_default(
    service: MongoDBService, afailing: AsyncCountingDatabase
) -> None:
    with pytest.raises(NoSqlConnectionError):
        await service.aupdate_many("counters", {}, {"$inc": {"hits": 1}})
    assert afailing.attempts == 1


def test_the_sync_update_is_attempted_once_by_default(
    service: MongoDBService, failing: CountingDatabase
) -> None:
    with pytest.raises(NoSqlConnectionError):
        service.update_one("counters", {"_id": "a"}, {"$inc": {"hits": 1}})
    assert failing.attempts == 1


def test_the_sync_update_is_repeated_only_when_the_caller_asks(
    service: MongoDBService, failing: CountingDatabase
) -> None:
    with pytest.raises(NoSqlConnectionError):
        service.update_one("counters", {"_id": "a"}, {"$inc": {"hits": 1}}, retry=True)
    assert failing.attempts == ATTEMPTS_WITH_RETRY


async def test_a_read_is_repeated_without_being_asked(
    service: MongoDBService, afailing: AsyncCountingDatabase
) -> None:
    with pytest.raises(NoSqlConnectionError):
        await service.afind_one("users", {"_id": "a"})
    assert afailing.attempts == ATTEMPTS_WITH_RETRY


async def test_a_single_document_insert_is_repeated_without_being_asked(
    service: MongoDBService, afailing: AsyncCountingDatabase
) -> None:
    """The driver sends a transaction number, so a repeat lands once."""
    with pytest.raises(NoSqlConnectionError):
        await service.ainsert_one("users", {"n": 1})
    assert afailing.attempts == ATTEMPTS_WITH_RETRY


async def test_the_update_document_reaches_the_driver_unchanged(
    service: MongoDBService, afailing: AsyncCountingDatabase
) -> None:
    """Nothing rewrites a `$inc` into an idempotent `$set` on the way down."""
    with pytest.raises(NoSqlConnectionError):
        await service.aupdate_one("counters", {"_id": "a"}, {"$inc": {"hits": 5}})
    assert afailing.updates[0] == ({"_id": "a"}, {"$inc": {"hits": 5}})
