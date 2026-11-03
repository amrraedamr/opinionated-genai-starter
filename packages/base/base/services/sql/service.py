from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import cast

from base.tools.errors import MissingExtraError

try:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("sql", "sqlalchemy", feature="SQLService") from exc

from base.clients.sql.client import SQLClient
from base.models.definitions.database import SQLBackend
from base.services.common.meta import SingletonABC
from base.services.configuration import get_settings
from base.services.configuration.protocols import SupportsSQLConfig
from base.services.sql import AbstractSQLService


class SQLService(AbstractSQLService, metaclass=SingletonABC):
    def __init__(self, config: SupportsSQLConfig | None = None) -> None:
        self._config: SupportsSQLConfig = config or cast(
            SupportsSQLConfig, get_settings()
        )
        self._backend: SQLBackend = SQLBackend(self._config.sql_backend)
        self._client: SQLClient = SQLClient(
            self._backend, self._config, _key=self._backend.value
        )

    @contextmanager
    def _get_session(self) -> Generator[Session, None, None]:
        with self._client.session() as session:
            yield session

    @asynccontextmanager
    async def _aget_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self._client.asession() as session:
            yield session

    def get_service_name(self) -> str:
        return self._backend.value.capitalize()

    def dispose(self) -> None:
        self._client.close()

    async def adispose(self) -> None:
        await self._client.aclose()
