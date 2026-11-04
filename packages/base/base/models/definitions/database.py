import importlib.util
from enum import StrEnum

# SQLAlchemy's driver name for SQLite is not its import name.
_DRIVER_IMPORTS: dict[str, str] = {"pysqlite": "sqlite3"}


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

    @property
    def extra(self) -> str:
        """`base` extra that installs this backend's drivers."""
        match self:
            case SQLBackend.SQLITE:
                return "sqlite"
            case SQLBackend.MYSQL:
                return "mysql"
            case SQLBackend.POSTGRESQL:
                return "postgres"

    @property
    def distribution(self) -> str:
        """Distributions that extra pulls in, for an error message to name."""
        match self:
            case SQLBackend.SQLITE:
                return "aiosqlite"
            case SQLBackend.MYSQL:
                return "aiomysql / pymysql"
            case SQLBackend.POSTGRESQL:
                return "asyncpg / psycopg2-binary"

    @property
    def required_imports(self) -> tuple[str, ...]:
        """Import names of the sync and async drivers this backend needs."""
        return tuple(
            _DRIVER_IMPORTS.get(driver, driver)
            for driver in (self.sync_driver, self.async_driver)
        )

    @property
    def missing_imports(self) -> tuple[str, ...]:
        """Drivers this backend needs that the running build does not install."""
        return tuple(
            name
            for name in self.required_imports
            if importlib.util.find_spec(name) is None
        )

    @property
    def is_installed(self) -> bool:
        """Whether this build carries the drivers to open this backend."""
        return not self.missing_imports


def installed_backends() -> tuple[SQLBackend, ...]:
    """Backends the running build can actually connect with.

    Probes the environment rather than reading a manifest, so it answers for a
    built image as well as a development checkout.
    """
    return tuple(backend for backend in SQLBackend if backend.is_installed)


class Locale(StrEnum):
    EN = "en"
    AR = "ar"
    OTHER = "other"


class SortingOrder(StrEnum):
    ASC = "asc"
    DESC = "desc"
