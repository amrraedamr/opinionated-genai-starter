from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
from typing import Any, TypeVar, final

from base.tools.errors import MissingExtraError

try:
    from sqlalchemy import Select, asc, desc, func, select, text
    from sqlalchemy.engine import Result
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session
    from sqlalchemy.sql.expression import UnaryExpression
except ModuleNotFoundError as exc:  # pragma: no cover
    raise MissingExtraError("sql", "sqlalchemy", feature="AbstractSQLService") from exc

from base.models.view_models.database import (
    PaginatedResult,
    PaginationParams,
    SortingOrder,
    SortingParams,
)
from base.services.error_handling import (
    aclassifying,
    aretrying,
    classifying,
    retrying,
)
from base.services.error_handling.sql import SQLErrorHandler
from base.tools.errors import ServiceHealthCheckError
from base.tools.pagination import apaginate, paginate

T = TypeVar("T")
R = TypeVar("R")
RowT = TypeVar("RowT")


class AbstractSQLService(ABC):
    """
    Abstract base class for SQL services with both sync and async support.

    Subclasses only need to implement session management and metadata.
    All query, execute, pagination, upsert, and health check logic is handled here.
    """

    _error_handler: SQLErrorHandler = SQLErrorHandler()
    """Shared by every subclass: the policy belongs to the backend, not the
    instance, and the handler holds no per-call state."""

    # Abstract — subclasses must implement

    @abstractmethod
    def _get_session(self) -> AbstractContextManager[Session]: ...

    @abstractmethod
    def _aget_session(self) -> AbstractAsyncContextManager[AsyncSession]: ...

    @abstractmethod
    def get_service_name(self) -> str: ...

    @abstractmethod
    def dispose(self) -> None: ...

    @abstractmethod
    async def adispose(self) -> None: ...

    # Transactions callers drive themselves

    @final
    @contextmanager
    def transaction(self) -> Generator[Session, None, None]:
        """A unit of work for callers that must control the boundary.

        Use a store for ordinary CRUD. This is for work a store cannot express
        — several writes that must commit together, or row locking.

        Nothing here is retried: the statements are the caller's, and a failed
        session cannot be resumed, so repeating the block is a decision only
        the caller can make. Failures are still reported as this backend's
        family, so one `except` serves whichever method raised.
        """
        try:
            with self._get_session() as session:
                yield session
        except SQLAlchemyError as exc:
            raise self._error_handler.as_domain_error(exc) from exc

    @final
    @asynccontextmanager
    async def atransaction(self) -> AsyncGenerator[AsyncSession, None]:
        """A unit of work for callers that must control the boundary.

        Use a store for ordinary CRUD. This is for work a store cannot express
        — several writes that must commit together, or row locking.

        Nothing here is retried: the statements are the caller's, and a failed
        session cannot be resumed, so repeating the block is a decision only
        the caller can make. Failures are still reported as this backend's
        family, so one `except` serves whichever method raised.
        """
        try:
            async with self._aget_session() as session:
                yield session
        except SQLAlchemyError as exc:
            raise self._error_handler.as_domain_error(exc) from exc

    # Query

    @retrying
    def query(
        self,
        sql_query: Select[tuple[RowT]],
        result_processor: Callable[[Result[tuple[RowT]]], R],
        expunge: bool = True,
    ) -> R:
        """Performs select queries."""
        with self._get_session() as session:
            executed = session.execute(sql_query).unique()
            result = result_processor(executed)
            if expunge:
                session.expunge_all()
            return result

    @aretrying
    async def aquery(
        self,
        sql_query: Select[tuple[RowT]],
        result_processor: Callable[[Result[tuple[RowT]]], R],
        expunge: bool = True,
    ) -> R:
        """Performs select queries asynchronously."""
        async with self._aget_session() as session:
            executed = (await session.execute(sql_query)).unique()
            result = result_processor(executed)
            if expunge:
                session.expunge_all()
            return result

    # Execute

    @classifying
    def execute(self, execute_query: Any) -> Any:
        """Performs update, delete queries."""
        with self._get_session() as session:
            return session.execute(execute_query)

    @aclassifying
    async def aexecute(self, execute_query: Any) -> Any:
        """Performs update, delete queries asynchronously."""
        async with self._aget_session() as session:
            return await session.execute(execute_query)

    # Transactions

    @classifying
    def execute_in_transaction(
        self,
        statements: list[Any] | None = None,
        rows_to_add: list[T] | None = None,
        flush: bool = True,
        expunge: bool = True,
        refresh: bool = True,
    ) -> list[T]:
        """Execute delete/update statements and/or row inserts in a single
        transaction. All operations commit together or not at all. Returns
        the row list after flush and refresh so callers get storage-assigned
        fields (IDs, timestamps) populated."""
        rows = rows_to_add or []
        with self._get_session() as session:
            if statements:
                for stmt in statements:
                    session.execute(stmt)
            if rows:
                session.add_all(rows)
                if flush:
                    session.flush()
                if refresh:
                    for row in rows:
                        session.refresh(row)
                if expunge:
                    session.expunge_all()
        return rows

    @aclassifying
    async def aexecute_in_transaction(
        self,
        statements: list[Any] | None = None,
        rows_to_add: list[T] | None = None,
        flush: bool = True,
        expunge: bool = True,
        refresh: bool = True,
    ) -> list[T]:
        """Execute delete/update statements and/or row inserts in a single
        transaction asynchronously. All operations commit together or not at
        all. Returns the row list after flush and refresh so callers get
        storage-assigned fields (IDs, timestamps) populated."""
        rows = rows_to_add or []
        async with self._aget_session() as session:
            if statements:
                for stmt in statements:
                    await session.execute(stmt)
            if rows:
                session.add_all(rows)
                if flush:
                    await session.flush()
                if refresh:
                    for row in rows:
                        await session.refresh(row)
                if expunge:
                    session.expunge_all()
        return rows

    # Bulk Operations

    @classifying
    def update_many(self, update_query: Any) -> None:
        """Performs bulk update queries."""
        with self._get_session() as session:
            session.execute(update_query)

    @aclassifying
    async def aupdate_many(self, update_query: Any) -> None:
        """Performs bulk update queries asynchronously."""
        async with self._aget_session() as session:
            await session.execute(update_query)

    @classifying
    def delete_many(self, delete_query: Any) -> None:
        """Performs bulk delete queries."""
        with self._get_session() as session:
            session.execute(delete_query)

    @aclassifying
    async def adelete_many(self, delete_query: Any) -> None:
        """Performs bulk delete queries asynchronously."""
        async with self._aget_session() as session:
            await session.execute(delete_query)

    # Pagination & Sorting

    @retrying
    def paginate(
        self,
        sql_query: Any,
        page_params: PaginationParams,
        expunge: bool = True,
        return_all: bool = False,
        has_subquery: bool = False,
    ) -> PaginatedResult:
        """Performs select queries with pagination."""
        count_query = select(func.count()).select_from(sql_query.subquery())
        total: int = self.query(count_query, lambda res: res.scalar()) or 0

        def fetch(skip: int, limit: int) -> list[Any]:
            rows: list[Any] = self.query(
                sql_query.offset(skip).limit(limit),
                lambda res: list(res.fetchall()),
                expunge=expunge,
            )
            return self._extract_paginated_data(rows, return_all, has_subquery)

        return paginate(total, page_params, fetch)

    @aretrying
    async def apaginate(
        self,
        sql_query: Any,
        page_params: PaginationParams,
        expunge: bool = True,
        return_all: bool = False,
        has_subquery: bool = False,
    ) -> PaginatedResult:
        """Performs select queries with pagination asynchronously."""
        count_query = select(func.count()).select_from(sql_query.subquery())
        total: int = await self.aquery(count_query, lambda res: res.scalar()) or 0

        async def afetch(skip: int, limit: int) -> list[Any]:
            rows: list[Any] = await self.aquery(
                sql_query.offset(skip).limit(limit),
                lambda res: list(res.fetchall()),
                expunge=expunge,
            )
            return self._extract_paginated_data(rows, return_all, has_subquery)

        return await apaginate(total, page_params, afetch)

    @staticmethod
    def _extract_paginated_data(
        result: list[Any], return_all: bool, has_subquery: bool
    ) -> list[Any]:
        if return_all:
            return result
        if has_subquery:
            return [item for row in result for item in row]
        return [row[0] for row in result]

    @staticmethod
    def sorting(sorting_params: SortingParams) -> UnaryExpression | None:
        """Builds sorting clause (sync — no I/O)."""
        if not sorting_params.sort_by or not sorting_params.sort_order:
            return None
        return (
            asc(sorting_params.sort_by)
            if sorting_params.sort_order == SortingOrder.ASC
            else desc(sorting_params.sort_by)
        )

    # Upsert

    @retrying
    def upsert_row(
        self,
        model_row_object: T,
        flush: bool = True,
        expunge: bool = True,
        refresh: bool = True,
    ) -> T:
        """Insert OR Replace a model row object within the database."""
        with self._get_session() as session:
            session.add(model_row_object)
            if flush:
                session.flush()
            if refresh:
                session.refresh(model_row_object)
            if expunge:
                session.expunge(model_row_object)
        return model_row_object

    @aretrying
    async def aupsert_row(
        self,
        model_row_object: T,
        flush: bool = True,
        expunge: bool = True,
        refresh: bool = True,
    ) -> T:
        """Insert OR Replace a model row object asynchronously."""
        async with self._aget_session() as session:
            session.add(model_row_object)
            if flush:
                await session.flush()
            if refresh:
                await session.refresh(model_row_object)
            if expunge:
                session.expunge(model_row_object)
        return model_row_object

    @retrying
    def upsert_rows(
        self,
        model_row_objects: list[T],
        flush: bool = True,
        expunge: bool = True,
        refresh: bool = True,
    ) -> list[T]:
        """Insert OR Replace model row objects within the database."""
        with self._get_session() as session:
            session.add_all(model_row_objects)
            if flush:
                session.flush()
            if refresh:
                for obj in model_row_objects:
                    session.refresh(obj)
            if expunge:
                session.expunge_all()
        return model_row_objects

    @aretrying
    async def aupsert_rows(
        self,
        model_row_objects: list[T],
        flush: bool = True,
        expunge: bool = True,
        refresh: bool = True,
    ) -> list[T]:
        """Insert OR Replace model row objects asynchronously."""
        async with self._aget_session() as session:
            session.add_all(model_row_objects)
            if flush:
                await session.flush()
            if refresh:
                for obj in model_row_objects:
                    await session.refresh(obj)
            if expunge:
                session.expunge_all()
        return model_row_objects

    @retrying
    def bulk_upsert_rows(
        self,
        model_row_objects: list[Any],
        flush: bool = True,
        expunge: bool = True,
        refresh: bool = True,
    ) -> None:
        """Insert OR Replace model row objects in bulk (no identity tracking)."""
        with self._get_session() as session:
            session.add_all(model_row_objects)
            if flush:
                session.flush()
            if refresh:
                for obj in model_row_objects:
                    session.refresh(obj)
            if expunge:
                session.expunge_all()

    @aretrying
    async def abulk_upsert_rows(
        self,
        model_row_objects: list[Any],
        flush: bool = True,
        expunge: bool = True,
        refresh: bool = True,
    ) -> None:
        """Insert OR Replace model row objects in bulk asynchronously."""
        async with self._aget_session() as session:
            session.add_all(model_row_objects)
            if flush:
                await session.flush()
            if refresh:
                for obj in model_row_objects:
                    await session.refresh(obj)
            if expunge:
                session.expunge_all()

    # Health Check

    def health_check(self) -> bool:
        """Performs a lightweight query to verify database accessibility."""
        try:
            with self._get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as e:
            raise ServiceHealthCheckError(
                f"{self.get_service_name()} health check failed: {e}"
            ) from e

    async def ahealth_check(self) -> bool:
        """Performs a lightweight query to verify database accessibility asynchronously."""
        try:
            async with self._aget_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as e:
            raise ServiceHealthCheckError(
                f"{self.get_service_name()} health check failed: {e}"
            ) from e
