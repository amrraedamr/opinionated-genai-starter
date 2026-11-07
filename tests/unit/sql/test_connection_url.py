"""What `SQLClient` builds before it ever opens a connection.

The URL, the pool and the connect arguments are the whole of this client's
configuration surface, and all three are decided per backend. They are worth
pinning here because a mistake in any of them surfaces as a connection
failure in production rather than as a test failure.
"""

from __future__ import annotations

import importlib.util
import ssl
from typing import Any

import pytest

from base.clients.sql.client import SQLClient
from base.models.definitions.database import SQLBackend
from base.services.configuration.mixins import (
    MySQLSettingsMixin,
    PostgresSettingsMixin,
    SQLSettingsMixin,
)
from base.tools.errors import UnsupportedBackendError

ALL_MIXINS: tuple[type, ...] = (
    SQLSettingsMixin,
    PostgresSettingsMixin,
    MySQLSettingsMixin,
)


@pytest.fixture
def config(make_settings: Any) -> Any:
    """Settings carrying every backend's fields at once."""
    return make_settings(
        *ALL_MIXINS,
        sql_backend="postgresql",
        pg_host="db.internal",
        pg_port="6543",
        pg_database="appdb",
        pg_username="app",
        pg_password="secret",
        pg_ssl_root_cert="/certs/ca.pem",
        mysql_host="mysql.internal",
        mysql_database="appdb",
        mysql_username="app",
        mysql_password="secret",
        sqlite_db_path="database/app.db",
        db_ssl_enabled=False,
    )


def client_for(backend: SQLBackend, config: Any) -> SQLClient:
    """A client keyed per backend, so one test's client is not another's."""
    return SQLClient(backend, config, _key=backend.value)


def test_postgresql_url_names_the_driver_host_port_and_database(config: Any) -> None:
    client = client_for(SQLBackend.POSTGRESQL, config)
    assert (
        client._get_database_url("psycopg2")
        == "postgresql+psycopg2://app:secret@db.internal:6543/appdb"
    )


def test_postgresql_sync_and_async_urls_differ_only_in_driver(config: Any) -> None:
    client = client_for(SQLBackend.POSTGRESQL, config)
    sync_url = client._get_database_url(SQLBackend.POSTGRESQL.sync_driver)
    async_url = client._get_database_url(SQLBackend.POSTGRESQL.async_driver)
    assert sync_url.startswith("postgresql+psycopg2://")
    assert async_url.startswith("postgresql+asyncpg://")
    assert sync_url.split("://", 1)[1] == async_url.split("://", 1)[1]


def test_mysql_url_carries_no_port(config: Any) -> None:
    client = client_for(SQLBackend.MYSQL, config)
    assert (
        client._get_database_url("pymysql")
        == "mysql+pymysql://app:secret@mysql.internal/appdb"
    )


def test_mysql_async_url_uses_aiomysql(config: Any) -> None:
    client = client_for(SQLBackend.MYSQL, config)
    url = client._get_database_url(SQLBackend.MYSQL.async_driver)
    assert url.startswith("mysql+aiomysql://")


def test_sqlite_url_is_the_configured_path(config: Any) -> None:
    client = client_for(SQLBackend.SQLITE, config)
    assert client._get_database_url("pysqlite") == "sqlite+pysqlite:///database/app.db"
    assert (
        client._get_database_url("aiosqlite") == "sqlite+aiosqlite:///database/app.db"
    )


def test_sqlite_url_takes_an_absolute_path_as_four_slashes(make_settings: Any) -> None:
    config = make_settings(
        SQLSettingsMixin, sql_backend="sqlite", sqlite_db_path="/var/data/app.db"
    )
    client = client_for(SQLBackend.SQLITE, config)
    assert client._get_database_url("pysqlite") == "sqlite+pysqlite:////var/data/app.db"


def test_postgresql_password_is_percent_escaped(make_settings: Any) -> None:
    config = make_settings(
        *ALL_MIXINS,
        sql_backend="postgresql",
        pg_host="h",
        pg_port="5432",
        pg_database="d",
        pg_username="user name",
        pg_password="p@ss:w/rd#1",
    )
    url = client_for(SQLBackend.POSTGRESQL, config)._get_database_url("psycopg2")
    assert url == "postgresql+psycopg2://user+name:p%40ss%3Aw%2Frd%231@h:5432/d"


def test_mysql_password_is_percent_escaped(make_settings: Any) -> None:
    config = make_settings(
        *ALL_MIXINS,
        sql_backend="mysql",
        mysql_host="h",
        mysql_database="d",
        mysql_username="u/u",
        mysql_password="p@ss",
    )
    url = client_for(SQLBackend.MYSQL, config)._get_database_url("pymysql")
    assert url == "mysql+pymysql://u%2Fu:p%40ss@h/d"


def test_a_missing_host_still_builds_a_url(make_settings: Any) -> None:
    """Nothing validates the host, so an unset one reaches the driver empty."""
    config = make_settings(
        *ALL_MIXINS,
        sql_backend="postgresql",
        pg_database="d",
        pg_username="u",
        pg_password="p",
    )
    url = client_for(SQLBackend.POSTGRESQL, config)._get_database_url("psycopg2")
    assert url == "postgresql+psycopg2://u:p@:5432/d"


def test_sqlite_pool_kwargs_carry_no_pool_sizing(config: Any) -> None:
    client = client_for(SQLBackend.SQLITE, config)
    assert client._get_pool_kwargs() == {"echo": False, "echo_pool": False}


@pytest.mark.parametrize("backend", [SQLBackend.POSTGRESQL, SQLBackend.MYSQL])
def test_server_backends_pool_from_settings(
    backend: SQLBackend, make_settings: Any
) -> None:
    config = make_settings(
        *ALL_MIXINS,
        sql_backend=backend.value,
        db_pool_size=3,
        db_max_overflow=7,
        db_pool_timeout=11,
        db_pool_recycle=99,
    )
    assert client_for(backend, config)._get_pool_kwargs() == {
        "pool_size": 3,
        "max_overflow": 7,
        "pool_timeout": 11,
        "pool_recycle": 99,
        "pool_pre_ping": True,
        "pool_use_lifo": True,
        "echo": False,
        "echo_pool": False,
    }


def test_sqlite_connect_args_allow_cross_thread_use(config: Any) -> None:
    client = client_for(SQLBackend.SQLITE, config)
    assert client._get_sync_connect_args() == {"check_same_thread": False}
    assert client._get_async_connect_args() == {"check_same_thread": False}


def test_postgresql_without_ssl_sends_timeouts_only(config: Any) -> None:
    client = client_for(SQLBackend.POSTGRESQL, config)
    assert client._get_sync_connect_args() == {
        "connect_timeout": 10,
        "options": "-c statement_timeout=300000",
    }
    assert client._get_async_connect_args() == {"timeout": 10}


def test_postgresql_with_ssl_sends_sslmode_to_psycopg2_and_a_context_to_asyncpg(
    make_settings: Any,
) -> None:
    """The two drivers disagree on how SSL is asked for, and both are served."""
    config = make_settings(
        *ALL_MIXINS,
        sql_backend="postgresql",
        db_ssl_enabled=True,
        pg_ssl_root_cert="/certs/ca.pem",
    )
    client = client_for(SQLBackend.POSTGRESQL, config)
    sync_args = client._get_sync_connect_args()
    assert sync_args["sslmode"] == "require"
    assert sync_args["sslrootcert"] == "/certs/ca.pem"
    assert isinstance(client._get_async_connect_args()["ssl"], ssl.SSLContext)


def test_mysql_without_ssl_sends_timeouts_only(config: Any) -> None:
    client = client_for(SQLBackend.MYSQL, config)
    assert client._get_sync_connect_args() == {
        "connect_timeout": 10,
        "read_timeout": 300,
        "write_timeout": 300,
    }
    assert client._get_async_connect_args() == {"connect_timeout": 10}


def test_mysql_with_ssl_sends_a_context_on_both_paths(make_settings: Any) -> None:
    config = make_settings(*ALL_MIXINS, sql_backend="mysql", db_ssl_enabled=True)
    client = client_for(SQLBackend.MYSQL, config)
    assert isinstance(client._get_sync_connect_args()["ssl"], ssl.SSLContext)
    assert isinstance(client._get_async_connect_args()["ssl"], ssl.SSLContext)


def test_sqlite_ignores_the_ssl_setting(make_settings: Any) -> None:
    config = make_settings(SQLSettingsMixin, sql_backend="sqlite", db_ssl_enabled=True)
    client = client_for(SQLBackend.SQLITE, config)
    assert "ssl" not in client._get_sync_connect_args()
    assert "ssl" not in client._get_async_connect_args()


def test_an_unknown_backend_is_refused_by_every_builder(config: Any) -> None:
    """The fallthrough arms are unreachable through the enum, and still guard."""
    client = client_for(SQLBackend.SQLITE, config)
    client._backend = "oracle"  # type: ignore[assignment]
    for build in (
        lambda: client._get_database_url("x"),
        client._get_pool_kwargs,
        client._get_sync_connect_args,
        client._get_async_connect_args,
    ):
        with pytest.raises(ValueError, match="Unsupported SQL backend"):
            build()


def test_a_backend_whose_driver_is_absent_names_what_this_build_runs(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_find_spec = importlib.util.find_spec

    def without_asyncpg(name: str, package: str | None = None) -> Any:
        return None if name == "asyncpg" else real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", without_asyncpg)
    with pytest.raises(UnsupportedBackendError) as caught:
        SQLClient(SQLBackend.POSTGRESQL, config, _key="no-driver")

    error = caught.value
    assert error.backend == "postgresql"
    assert error.extra == "postgres"
    assert error.missing == ("asyncpg",)
    assert "sqlite" in error.available


def test_a_client_is_reused_per_backend_key(config: Any) -> None:
    """Two services on one backend share an engine rather than opening two."""
    first = client_for(SQLBackend.SQLITE, config)
    second = client_for(SQLBackend.SQLITE, config)
    assert first is second
    assert SQLClient(SQLBackend.MYSQL, config, _key="mysql") is not first
