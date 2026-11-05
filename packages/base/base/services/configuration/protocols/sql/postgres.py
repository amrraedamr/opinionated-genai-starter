from typing import Protocol, runtime_checkable

from base.services.configuration.protocols.sql import SupportsSQLConfig


@runtime_checkable
class SupportsPostgresConfig(SupportsSQLConfig, Protocol):
    """Satisfied by `SQLSettingsMixin` plus `PostgresSettingsMixin`."""

    pg_host: str
    pg_port: str
    pg_database: str
    pg_ssl_root_cert: str
    pg_username: str
    pg_password: str
