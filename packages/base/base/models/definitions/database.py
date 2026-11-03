from enum import StrEnum


class SQLBackend(StrEnum):
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"

    @property
    def sync_driver(self) -> str:
        match self:
            case SQLBackend.SQLITE:
                return "pysqlite"
            case SQLBackend.MYSQL:
                return "pymysql"
            case SQLBackend.POSTGRESQL:
                return "psycopg2"

    @property
    def async_driver(self) -> str:
        match self:
            case SQLBackend.SQLITE:
                return "aiosqlite"
            case SQLBackend.MYSQL:
                return "aiomysql"
            case SQLBackend.POSTGRESQL:
                return "asyncpg"


class Locale(StrEnum):
    EN = "en"
    AR = "ar"
    OTHER = "other"


class SortingOrder(StrEnum):
    ASC = "asc"
    DESC = "desc"
