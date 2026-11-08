"""`migrations.py`: the Alembic environment every service's `env.py` calls.

Running a real migration needs an Alembic context, which only `alembic` itself
builds — so the context is replaced here and what is asserted is the call the
environment would have made: which URL, which metadata, and whether it went
down the offline or the online path. The URLs themselves are pinned in
`tests/unit/sql/test_migrations.py`.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest
from sqlalchemy import MetaData

from base.services.configuration import get_settings
from base.services.configuration.mixins import SQLSettingsMixin
from base.tools import migrations
from base.tools.migrations import _arun_online, _run_offline, _run_sync, _sync_url


class FakeContext:
    """Stands in for `alembic.context`, recording how it was configured."""

    def __init__(self, offline: bool = False, config_file: str | None = None) -> None:
        self.offline = offline
        self.config = type("Config", (), {"config_file_name": config_file})()
        self.configured: list[dict[str, Any]] = []
        self.transactions = 0
        self.migrations = 0

    def is_offline_mode(self) -> bool:
        return self.offline

    def configure(self, **kwargs: Any) -> None:
        self.configured.append(kwargs)

    @contextmanager
    def begin_transaction(self) -> Generator[None]:
        self.transactions += 1
        yield

    def run_migrations(self) -> None:
        self.migrations += 1


class FakeConnection:
    """One connection, which Alembic drives through `run_sync`."""

    def __init__(self) -> None:
        self.ran: list[Any] = []

    async def run_sync(self, fn: Any, *args: Any) -> None:
        self.ran.append(fn)
        fn(self, *args)


class FakeEngine:
    def __init__(self, url: str) -> None:
        self.url = url
        self.connection = FakeConnection()
        self.disposed = 0

    def connect(self) -> Any:
        engine = self

        class Opened:
            async def __aenter__(self) -> FakeConnection:
                return engine.connection

            async def __aexit__(self, *exc: Any) -> None:
                return None

        return Opened()

    async def dispose(self) -> None:
        self.disposed += 1


@pytest.fixture
def sqlite_settings(make_settings: Any) -> Any:
    return make_settings(
        SQLSettingsMixin, sql_backend="sqlite", sqlite_db_path="database/app.db"
    )


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> FakeContext:
    fake = FakeContext()
    monkeypatch.setattr(migrations, "context", fake)
    return fake


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> list[FakeEngine]:
    built: list[FakeEngine] = []

    def create(url: str, **kwargs: Any) -> FakeEngine:
        built.append(FakeEngine(url))
        return built[-1]

    monkeypatch.setattr(migrations, "create_async_engine", create)
    return built


def test_the_offline_url_carries_the_synchronous_driver(sqlite_settings: Any) -> None:
    assert _sync_url(sqlite_settings) == "sqlite+pysqlite:///database/app.db"


def test_a_postgres_service_migrates_offline_through_psycopg2(
    make_settings: Any,
) -> None:
    from base.services.configuration.mixins import PostgresSettingsMixin

    settings = make_settings(
        SQLSettingsMixin,
        PostgresSettingsMixin,
        sql_backend="postgresql",
        pg_host="db.internal",
        pg_port="5432",
        pg_database="appdb",
        pg_username="app",
        pg_password="secret",
    )

    assert _sync_url(settings).startswith("postgresql+psycopg2://")


def test_an_offline_run_renders_the_url_rather_than_opening_it(
    context: FakeContext, sqlite_settings: Any
) -> None:
    metadata = MetaData()

    _run_offline(metadata, sqlite_settings)

    assert context.configured[0]["url"] == "sqlite+pysqlite:///database/app.db"
    assert context.configured[0]["target_metadata"] is metadata
    assert context.configured[0]["literal_binds"] is True


def test_an_offline_run_emits_its_statements_inside_one_transaction(
    context: FakeContext, sqlite_settings: Any
) -> None:
    _run_offline(MetaData(), sqlite_settings)

    assert (context.transactions, context.migrations) == (1, 1)


def test_every_run_batches_alterations_so_sqlite_can_apply_them(
    context: FakeContext, sqlite_settings: Any
) -> None:
    """SQLite cannot `ALTER`, so Alembic has to rebuild the table instead."""
    _run_offline(MetaData(), sqlite_settings)
    _run_sync(object(), MetaData())  # type: ignore[arg-type]

    assert all(call["render_as_batch"] is True for call in context.configured)


def test_a_connected_run_hands_alembic_the_connection_rather_than_a_url(
    context: FakeContext,
) -> None:
    connection = object()
    metadata = MetaData()

    _run_sync(connection, metadata)  # type: ignore[arg-type]

    assert context.configured[0]["connection"] is connection
    assert context.configured[0]["target_metadata"] is metadata
    assert "url" not in context.configured[0]


def test_a_connected_run_also_wraps_its_statements_in_a_transaction(
    context: FakeContext,
) -> None:
    _run_sync(object(), MetaData())  # type: ignore[arg-type]

    assert (context.transactions, context.migrations) == (1, 1)


async def test_an_online_run_opens_the_async_url_and_migrates_through_it(
    context: FakeContext, engines: list[FakeEngine], sqlite_settings: Any
) -> None:
    await _arun_online(MetaData(), sqlite_settings)

    assert engines[0].url == "sqlite+aiosqlite:///database/app.db"
    assert context.migrations == 1


async def test_an_online_run_disposes_of_the_engine_it_opened(
    context: FakeContext, engines: list[FakeEngine], sqlite_settings: Any
) -> None:
    await _arun_online(MetaData(), sqlite_settings)

    assert engines[0].disposed == 1


def test_running_migrations_registers_the_settings_it_was_handed(
    context: FakeContext, engines: list[FakeEngine], make_settings: Any
) -> None:
    """A service's `env.py` is its composition root, so nothing else will."""
    settings = make_settings(
        SQLSettingsMixin, sql_backend="sqlite", sqlite_db_path="database/app.db"
    )
    from base.services.configuration import reset

    reset()

    migrations.run_migrations(MetaData(), settings)

    assert get_settings() is settings


def test_offline_mode_renders_statements_instead_of_connecting(
    context: FakeContext, engines: list[FakeEngine], sqlite_settings: Any
) -> None:
    context.offline = True

    migrations.run_migrations(MetaData(), sqlite_settings)

    assert engines == []
    assert context.configured[0]["literal_binds"] is True


def test_online_mode_connects_rather_than_rendering(
    context: FakeContext, engines: list[FakeEngine], sqlite_settings: Any
) -> None:
    migrations.run_migrations(MetaData(), sqlite_settings)

    assert len(engines) == 1
    assert "connection" in context.configured[0]


def test_logging_is_configured_from_the_alembic_ini_when_there_is_one(
    context: FakeContext,
    engines: list[FakeEngine],
    sqlite_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[str] = []
    monkeypatch.setattr(migrations, "fileConfig", configured.append)
    context.config.config_file_name = "alembic.ini"

    migrations.run_migrations(MetaData(), sqlite_settings)

    assert configured == ["alembic.ini"]


def test_no_alembic_ini_means_logging_is_left_as_the_service_set_it(
    context: FakeContext,
    engines: list[FakeEngine],
    sqlite_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[str] = []
    monkeypatch.setattr(migrations, "fileConfig", configured.append)

    migrations.run_migrations(MetaData(), sqlite_settings)

    assert configured == []


def test_a_failing_migration_is_not_swallowed(
    context: FakeContext, engines: list[FakeEngine], sqlite_settings: Any
) -> None:
    def explode() -> None:
        raise RuntimeError("revision is missing")

    context.run_migrations = explode  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="revision is missing"):
        migrations.run_migrations(MetaData(), sqlite_settings)
