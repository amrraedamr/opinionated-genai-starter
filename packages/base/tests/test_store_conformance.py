"""Run the conformance suite against an in-memory AbstractStore.

Proves the suite exercises a real implementation. A SQL or Mongo store passes
the same class by swapping `make_store`.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

from base.models.view_models.database import FilteredPaginationParams, PaginatedResult
from base.services.store import AbstractStore
from base.testing.conformance import AbstractStoreConformance


@dataclass
class Widget:
    name: str = "widget"
    id: uuid.UUID = field(default_factory=uuid.uuid4)


class InMemoryWidgetStore(AbstractStore[Widget]):
    def __init__(self) -> None:
        self._rows: dict[uuid.UUID, Widget] = {}

    def _match(self, value: Any, column_name: str) -> Widget | None:
        for row in self._rows.values():
            if getattr(row, column_name, None) == value:
                return row
        return None

    def _get(self, value: Any, column_name: str = "id") -> Widget | None:
        return self._match(value, column_name)

    async def _aget(self, value: Any, column_name: str = "id") -> Widget | None:
        return self._get(value, column_name)

    def _get_many(self, keys: list[Any], column_name: str = "id") -> list[Widget]:
        found = [self._match(key, column_name) for key in keys]
        return [row for row in found if row is not None]

    async def _aget_many(self, keys: list[Any], column_name: str = "id") -> list[Widget]:
        return self._get_many(keys, column_name)

    def _add(self, entity: Widget) -> Widget:
        self._rows[entity.id] = entity
        return entity

    async def _aadd(self, entity: Widget) -> Widget:
        return self._add(entity)

    def _add_many(self, entities: list[Widget]) -> list[Widget]:
        return [self._add(entity) for entity in entities]

    async def _aadd_many(self, entities: list[Widget]) -> list[Widget]:
        return self._add_many(entities)

    def _edit(self, value: Any, entity: Widget, column_name: str = "id") -> Widget | None:
        existing = self._match(value, column_name)
        if existing is None:
            return None
        existing.name = entity.name
        return existing

    async def _aedit(
        self, value: Any, entity: Widget, column_name: str = "id"
    ) -> Widget | None:
        return self._edit(value, entity, column_name)

    def _delete(self, value: Any, column_name: str = "id") -> bool:
        existing = self._match(value, column_name)
        if existing is None:
            return False
        del self._rows[existing.id]
        return True

    async def _adelete(self, value: Any, column_name: str = "id") -> bool:
        return self._delete(value, column_name)

    def _get_all(self, **kwargs: Any) -> list[Widget]:
        return list(self._rows.values())

    async def _aget_all(self, **kwargs: Any) -> list[Widget]:
        return self._get_all(**kwargs)

    def _get_all_paged(
        self, filter_page_params: FilteredPaginationParams, **kwargs: Any
    ) -> PaginatedResult[Widget]:
        rows = self._get_all()
        size = filter_page_params.size
        page = filter_page_params.page
        start = (page - 1) * size
        total = len(rows)
        return PaginatedResult(
            data=rows[start : start + size],
            page=page,
            size=size,
            total=total,
            total_pages=(total + size - 1) // size if size else 0,
        )

    async def _aget_all_paged(
        self, filter_page_params: FilteredPaginationParams, **kwargs: Any
    ) -> PaginatedResult[Widget]:
        return self._get_all_paged(filter_page_params, **kwargs)


class TestInMemoryWidgetStore(AbstractStoreConformance):
    def make_store(self) -> AbstractStore[Widget]:
        return InMemoryWidgetStore()

    def make_entity(self, **overrides: Any) -> Widget:
        return Widget(**overrides)
