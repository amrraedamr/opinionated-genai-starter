"""The Alembic wiring every service's `env.py` hands its metadata to.

Nothing here runs a migration — that needs an Alembic context. What is pinned
is the part a service gets wrong on its own: the URLs the environment derives
from the service's own settings, the guard that refuses settings without the
SQL mixin, and the type comparison that stops a custom column type looking
like a schema drift on every autogenerate.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy_utils.types.uuid import UUIDType

from base.services.configuration.mixins import (
    PostgresSettingsMixin,
    RedisSettingsMixin,
    SQLSettingsMixin,
)
from base.tools.migrations import _async_url, _client, _compare_type, run_migrations
from tests.conftest import MakeSettings


@pytest.fixture
def postgres_settings(make_settings: MakeSettings) -> Any:
    return make_settings(
        SQLSettingsMixin,
        PostgresSettingsMixin,
        sql_backend="postgresql",
        pg_host="db.internal",
        pg_port="5432",
        pg_database="appdb",
        pg_username="app",
        pg_password="secret",
    )


def test_the_environment_opens_the_backend_the_settings_name(
    postgres_settings: Any,
) -> None:
    assert _client(postgres_settings)._backend.value == "postgresql"


def test_the_async_url_carries_the_async_driver(postgres_settings: Any) -> None:
    assert _async_url(postgres_settings).startswith("postgresql+asyncpg://")


def test_the_async_url_points_at_the_service_s_own_database(
    postgres_settings: Any,
) -> None:
    assert _async_url(postgres_settings).endswith("@db.internal:5432/appdb")


def test_a_sqlite_service_migrates_its_own_file(make_settings: MakeSettings) -> None:
    settings = make_settings(
        SQLSettingsMixin, sql_backend="sqlite", sqlite_db_path="database/app.db"
    )
    assert _async_url(settings) == "sqlite+aiosqlite:///database/app.db"


def test_settings_without_the_sql_mixin_are_refused_by_name(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings(RedisSettingsMixin, redis_url="redis://x:1/0")
    with pytest.raises(TypeError, match="cannot run migrations; compose SQLMixin"):
        run_migrations(None, settings)  # type: ignore[arg-type]


def test_a_custom_uuid_column_never_reads_as_a_type_change() -> None:
    """Autogenerate otherwise proposes a diff for `UUIDType` on every run."""
    assert _compare_type(None, None, None, None, UUIDType()) is False


def test_an_ordinary_column_type_is_left_to_alembic() -> None:
    from sqlalchemy import String

    assert _compare_type(None, None, None, None, String(50)) is None
