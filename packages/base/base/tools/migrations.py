"""Alembic environment wiring shared by every service.

A service's `alembic/env.py` imports its ORM modules so they attach to its
declarative metadata, then hands that metadata here. The database a migration
reaches is whichever one the calling service's settings point at, so each
service keeps its own schema and its own revision history.
"""

import asyncio
from logging.config import fileConfig
from typing import Any

from sqlalchemy import MetaData, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy_utils.types.uuid import UUIDType

from alembic import context
from base.clients.sql.client import SQLClient
from base.models.definitions.database import SQLBackend
from base.services.configuration import CoreSettings, configure
from base.services.configuration.protocols import SupportsSQLConfig


def _client(settings: SupportsSQLConfig) -> SQLClient:
    backend = SQLBackend(settings.sql_backend)
    return SQLClient(backend, settings, _key=backend.value)


def _sync_url(settings: SupportsSQLConfig) -> str:
    client = _client(settings)
    return client._get_database_url(client._backend.sync_driver)


def _async_url(settings: SupportsSQLConfig) -> str:
    client = _client(settings)
    return client._get_database_url(client._backend.async_driver)


def _compare_type(
    _ctx: Any,
    _inspected_column: Any,
    _metadata_column: Any,
    _inspected_type: Any,
    metadata_type: Any,
) -> bool | None:
    """Skip false-positive type diffs for custom types like UUIDType on SQLite."""
    if isinstance(metadata_type, UUIDType):
        return False
    return None


def _run_offline(metadata: MetaData, settings: SupportsSQLConfig) -> None:
    context.configure(
        url=_sync_url(settings),
        target_metadata=metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=_compare_type,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync(connection: Connection, metadata: MetaData) -> None:
    context.configure(
        connection=connection,
        target_metadata=metadata,
        render_as_batch=True,
        compare_type=_compare_type,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _arun_online(metadata: MetaData, settings: SupportsSQLConfig) -> None:
    connectable = create_async_engine(_async_url(settings), poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(_run_sync, metadata)
    await connectable.dispose()


def run_migrations(metadata: MetaData, settings: CoreSettings) -> None:
    """Register `settings` and run this service's migrations against its database."""
    if not isinstance(settings, SupportsSQLConfig):
        raise TypeError(
            f"{type(settings).__name__} cannot run migrations; compose SQLMixin "
            "so the settings satisfy SupportsSQLConfig."
        )

    configure(settings)

    if context.config.config_file_name is not None:
        fileConfig(context.config.config_file_name)

    if context.is_offline_mode():
        _run_offline(metadata, settings)
    else:
        asyncio.run(_arun_online(metadata, settings))
