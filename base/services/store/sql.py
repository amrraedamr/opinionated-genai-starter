"""Shared SQL implementation of the `AbstractStore` contract.

`AbstractStore` stays a pure contract. This is the one concrete implementation
of it for a soft-deleted SQLAlchemy table, so a service store declares what is
genuinely its own — its model, the columns `edit` must not touch, the columns
`search` matches — and inherits the rest.
"""

from functools import cached_property
from typing import Any, ClassVar, TypeVar

from sqlalchemy import ColumnElement, Select, inspect, or_, select
from sqlalchemy.exc import IntegrityError

from base.models.orm import TimestampMixin
from base.models.view_models.database import FilteredPaginationParams, PaginatedResult
from base.services.sql.service import SQLService
from base.services.store import AbstractStore
from base.tools.database import SQLResultFormats

TEntity = TypeVar("TEntity", bound=TimestampMixin)


class SQLStore(AbstractStore[TEntity]):
    """`AbstractStore` over one soft-deleted SQLAlchemy table.

    Every row this store returns is an active one: soft-deleted rows are
    filtered out of every read, and `delete` soft-deletes rather than removing.

    Entities must also carry `SerializableMixin`, which every declarative root
    in this workspace provides, for `to_dict` and `update_from_dict`.
    """

    #: Columns `edit` never overwrites, whatever the caller sends.
    protected_fields: ClassVar[frozenset[str]] = frozenset(
        {"id", "created_at", "updated_at", "deleted_at"}
    )
    #: Columns `get_all_paged` matches `FilteredPaginationParams.search` against.
    search_fields: ClassVar[tuple[str, ...]] = ()

    def __init__(self, model: type[TEntity]) -> None:
        self._model: type[TEntity] = model
        self._sql_service: SQLService = SQLService()
        self._base_select_query: Select[tuple[TEntity]] = select(model)

    def _duplicate_error_types(self) -> tuple[type[BaseException], ...]:
        return (IntegrityError,)

    # Query building

    def _active(self, *conditions: ColumnElement[bool]) -> Select[tuple[TEntity]]:
        """The base query, narrowed to rows that are not soft-deleted."""
        return self._base_select_query.where(
            *conditions, self._model.deleted_at.is_(None)
        )

    def _column(self, name: str) -> Any:
        return getattr(self._model, name)

    def _filtered(
        self,
        filter_page_params: FilteredPaginationParams,
        *conditions: ColumnElement[bool],
    ) -> Select[tuple[TEntity]]:
        """An active-row query narrowed by the requested date range and search."""
        query = self._active(*conditions).where(
            self._model.created_at >= filter_page_params.from_date,
            self._model.created_at <= filter_page_params.to_date,
        )
        if filter_page_params.search and self.search_fields:
            term = f"%{filter_page_params.search}%"
            query = query.where(
                or_(*(self._column(name).ilike(term) for name in self.search_fields))
            )
        return query

    # Query running

    def _one(self, query: Select[tuple[TEntity]]) -> TEntity | None:
        return self._sql_service.query(query, SQLResultFormats.one_or_none_scalar)

    async def _aone(self, query: Select[tuple[TEntity]]) -> TEntity | None:
        return await self._sql_service.aquery(
            query, SQLResultFormats.one_or_none_scalar
        )

    def _many(self, query: Select[tuple[TEntity]]) -> list[TEntity]:
        return self._sql_service.query(query, SQLResultFormats.all_scalar_list)

    async def _amany(self, query: Select[tuple[TEntity]]) -> list[TEntity]:
        return await self._sql_service.aquery(query, SQLResultFormats.all_scalar_list)

    # Get single

    def _get(self, value: Any, column_name: str = "id") -> TEntity | None:
        return self._one(self._active(self._column(column_name) == value))

    async def _aget(self, value: Any, column_name: str = "id") -> TEntity | None:
        return await self._aone(self._active(self._column(column_name) == value))

    # Get many

    def _get_many(self, keys: list[Any], column_name: str = "id") -> list[TEntity]:
        return self._many(self._active(self._column(column_name).in_(keys)))

    async def _aget_many(
        self, keys: list[Any], column_name: str = "id"
    ) -> list[TEntity]:
        return await self._amany(self._active(self._column(column_name).in_(keys)))

    # Add

    def _add(self, entity: TEntity) -> TEntity:
        return self._sql_service.upsert_row(entity)

    async def _aadd(self, entity: TEntity) -> TEntity:
        return await self._sql_service.aupsert_row(entity)

    def _add_many(self, entities: list[TEntity]) -> list[TEntity]:
        return self._sql_service.upsert_rows(entities)

    async def _aadd_many(self, entities: list[TEntity]) -> list[TEntity]:
        return await self._sql_service.aupsert_rows(entities)

    # Edit

    @cached_property
    def _column_names(self) -> frozenset[str]:
        """Mapped column attributes, so relationships never reach `update_from_dict`."""
        return frozenset(attr.key for attr in inspect(self._model).mapper.column_attrs)

    def _sanitize_update_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v
            for k, v in data.items()
            if k in self._column_names and k not in self.protected_fields
        }

    def _apply(self, existing: TEntity, entity: TEntity) -> TEntity:
        existing.update_from_dict(self._sanitize_update_data(entity.to_dict()))
        return existing

    def _edit(
        self, value: Any, entity: TEntity, column_name: str = "id"
    ) -> TEntity | None:
        existing = self._get(value, column_name)
        if existing is None:
            return None
        return self._sql_service.upsert_row(self._apply(existing, entity))

    async def _aedit(
        self, value: Any, entity: TEntity, column_name: str = "id"
    ) -> TEntity | None:
        existing = await self._aget(value, column_name)
        if existing is None:
            return None
        return await self._sql_service.aupsert_row(self._apply(existing, entity))

    # Delete

    def _cascade(self, entity: TEntity) -> list[Any]:
        """Rows to soft-delete alongside `entity`. Empty unless a store owns children."""
        return []

    async def _acascade(self, entity: TEntity) -> list[Any]:
        """Rows to soft-delete alongside `entity`. Empty unless a store owns children."""
        return []

    def _soft_delete(self, entity: TEntity) -> bool:
        """Soft-delete a row the caller already loaded, plus anything it cascades to."""
        children = self._cascade(entity)
        entity.soft_delete()
        for child in children:
            child.soft_delete()
        self._sql_service.upsert_rows([entity, *children])
        return True

    async def _asoft_delete(self, entity: TEntity) -> bool:
        """Soft-delete a row the caller already loaded, plus anything it cascades to."""
        children = await self._acascade(entity)
        entity.soft_delete()
        for child in children:
            child.soft_delete()
        await self._sql_service.aupsert_rows([entity, *children])
        return True

    def _delete(self, value: Any, column_name: str = "id") -> bool:
        entity = self._get(value, column_name)
        return False if entity is None else self._soft_delete(entity)

    async def _adelete(self, value: Any, column_name: str = "id") -> bool:
        entity = await self._aget(value, column_name)
        return False if entity is None else await self._asoft_delete(entity)

    # Get all

    def _get_all(self, **kwargs: Any) -> list[TEntity]:
        return self._many(self._active())

    async def _aget_all(self, **kwargs: Any) -> list[TEntity]:
        return await self._amany(self._active())

    # Get all paged

    def _get_all_paged(
        self, filter_page_params: FilteredPaginationParams, **kwargs: Any
    ) -> PaginatedResult[TEntity]:
        return self._sql_service.paginate(
            self._filtered(filter_page_params), filter_page_params
        )

    async def _aget_all_paged(
        self, filter_page_params: FilteredPaginationParams, **kwargs: Any
    ) -> PaginatedResult[TEntity]:
        return await self._sql_service.apaginate(
            self._filtered(filter_page_params), filter_page_params
        )
