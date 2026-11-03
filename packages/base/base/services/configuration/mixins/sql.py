from typing import Literal

from pydantic import BaseModel


class SQLSettingsMixin(BaseModel):
    sql_backend: Literal["sqlite", "mysql", "postgresql"] = "sqlite"
    sqlite_db_path: str = "database/app.db"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 3600
    db_ssl_enabled: bool = True
