from typing import Protocol, runtime_checkable

from base.services.configuration.protocols.sql import SupportsSQLConfig


@runtime_checkable
class SupportsMySQLConfig(SupportsSQLConfig, Protocol):
    """Satisfied by `SQLSettingsMixin` plus `MySQLSettingsMixin`."""

    mysql_host: str
    mysql_database: str
    mysql_username: str
    mysql_password: str
