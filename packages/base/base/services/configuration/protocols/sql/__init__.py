from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsSQLConfig(Protocol):
    """Satisfied by `SQLSettingsMixin`."""

    sql_backend: str
    sqlite_db_path: str
    db_pool_size: int
    db_max_overflow: int
    db_pool_timeout: int
    db_pool_recycle: int
    db_ssl_enabled: bool
