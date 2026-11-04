import ssl
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, cast
from urllib.parse import quote_plus

from base.tools.errors import MissingExtraError, UnsupportedBackendError

try:
    from sqlalchemy import QueuePool, create_engine
    from sqlalchemy.engine import Engine
    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.pool import AsyncAdaptedQueuePool
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("sql", "sqlalchemy", feature="SQLClient") from exc

from base.models.definitions.database import SQLBackend, installed_backends
from base.services.common.meta import KeyedSingleton
from base.services.configuration import get_settings
from base.services.configuration.protocols import SupportsSQLConfig


class SQLClient(metaclass=KeyedSingleton):
    def __init__(
        self, backend: SQLBackend, config: SupportsSQLConfig | None = None
    ) -> None:
        self._backend: SQLBackend = backend
        self._config: SupportsSQLConfig = config or cast(
            SupportsSQLConfig, get_settings()
        )
        self._require_driver()

        self._sync_engine: Engine | None = None
        self._async_engine: AsyncEngine | None = None
        self._sync_session_maker: sessionmaker[Session] | None = None
        self._async_session_maker: async_sessionmaker[AsyncSession] | None = None

    def _require_driver(self) -> None:
        """Fail naming the backends this build runs, not with a driver error."""
        missing = self._backend.missing_imports
        if not missing:
            return
        raise UnsupportedBackendError(
            backend=self._backend.value,
            extra=self._backend.extra,
            distribution=self._backend.distribution,
            missing=missing,
            available=[backend.value for backend in installed_backends()],
        )

    def _get_database_url(self, driver: str) -> str:
        match self._backend:
            case SQLBackend.SQLITE:
                return f"sqlite+{driver}:///{self._config.sqlite_db_path}"
            case SQLBackend.MYSQL:
                username: str = quote_plus(self._config.mysql_username)
                password: str = quote_plus(self._config.mysql_password)
                return (
                    f"mysql+{driver}://{username}:{password}"
                    f"@{self._config.mysql_host}/{self._config.mysql_database}"
                )
            case SQLBackend.POSTGRESQL:
                username = quote_plus(self._config.pg_username)
                password = quote_plus(self._config.pg_password)
                return (
                    f"postgresql+{driver}://{username}:{password}"
                    f"@{self._config.pg_host}:{self._config.pg_port}/{self._config.pg_database}"
                )
            case _:
                raise ValueError(f"Unsupported SQL backend: {self._backend!r}")

    def _get_pool_kwargs(self) -> dict[str, Any]:
        match self._backend:
            case SQLBackend.SQLITE:
                return {"echo": False, "echo_pool": False}
            case SQLBackend.MYSQL | SQLBackend.POSTGRESQL:
                return {
                    "pool_size": self._config.db_pool_size,
                    "max_overflow": self._config.db_max_overflow,
                    "pool_timeout": self._config.db_pool_timeout,
                    "pool_recycle": self._config.db_pool_recycle,
                    "pool_pre_ping": True,
                    "pool_use_lifo": True,
                    "echo": False,
                    "echo_pool": False,
                }
            case _:
                raise ValueError(f"Unsupported SQL backend: {self._backend!r}")

    def _get_sync_connect_args(self) -> dict[str, Any]:
        match self._backend:
            case SQLBackend.SQLITE:
                return {"check_same_thread": False}
            case SQLBackend.MYSQL:
                args: dict[str, Any] = {
                    "connect_timeout": 10,
                    "read_timeout": 300,
                    "write_timeout": 300,
                }
                if self._config.db_ssl_enabled:
                    args["ssl"] = ssl.create_default_context()
                return args
            case SQLBackend.POSTGRESQL:
                args = {
                    "connect_timeout": 10,
                    "options": "-c statement_timeout=300000",
                }
                if self._config.db_ssl_enabled:
                    args["sslmode"] = "require"
                    args["sslrootcert"] = self._config.pg_ssl_root_cert
                return args
            case _:
                raise ValueError(f"Unsupported SQL backend: {self._backend!r}")

    def _get_async_connect_args(self) -> dict[str, Any]:
        match self._backend:
            case SQLBackend.SQLITE:
                return {"check_same_thread": False}
            case SQLBackend.MYSQL:
                args: dict[str, Any] = {"connect_timeout": 10}
                if self._config.db_ssl_enabled:
                    args["ssl"] = ssl.create_default_context()
                return args
            case SQLBackend.POSTGRESQL:
                args = {"timeout": 10}
                if self._config.db_ssl_enabled:
                    args["ssl"] = ssl.create_default_context()
                return args
            case _:
                raise ValueError(f"Unsupported SQL backend: {self._backend!r}")

    def _ensure_sync_initialized(self) -> None:
        if self._sync_engine is not None:
            return

        self._sync_engine = create_engine(
            url=self._get_database_url(self._backend.sync_driver),
            poolclass=QueuePool,
            connect_args=self._get_sync_connect_args(),
            **self._get_pool_kwargs(),
        )
        self._sync_session_maker = sessionmaker(
            bind=self._sync_engine,
            autoflush=False,
            expire_on_commit=False,
        )

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        self._ensure_sync_initialized()
        assert self._sync_session_maker is not None
        with self._sync_session_maker.begin() as session:
            yield session

    def _ensure_async_initialized(self) -> None:
        if self._async_engine is not None:
            return

        async_pool_class = (
            None if self._backend == SQLBackend.SQLITE else AsyncAdaptedQueuePool
        )
        self._async_engine = create_async_engine(
            url=self._get_database_url(self._backend.async_driver),
            poolclass=async_pool_class,
            connect_args=self._get_async_connect_args(),
            **self._get_pool_kwargs(),
        )
        self._async_session_maker = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            autoflush=False,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def asession(self) -> AsyncGenerator[AsyncSession, None]:
        self._ensure_async_initialized()
        assert self._async_session_maker is not None
        async with self._async_session_maker.begin() as session:
            yield session

    def close(self) -> None:
        if self._sync_engine is not None:
            self._sync_engine.dispose()
            self._sync_engine = None
            self._sync_session_maker = None

    async def aclose(self) -> None:
        if self._async_engine is not None:
            await self._async_engine.dispose()
            self._async_engine = None
            self._async_session_maker = None

    async def close_all(self) -> None:
        self.close()
        await self.aclose()
