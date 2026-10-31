"""Conformance suite every `AbstractStore` implementation must pass.

Subclass `AbstractStoreConformance`, implement `make_store` and `make_entity`,
and the backend is held to the same contract as every other one. That is what
makes swapping SQL for Mongo a verified claim rather than an intention.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from base.models.definitions.errors import DatabaseErrors
from base.models.view_models.database import FilteredPaginationParams
from base.services.store import AbstractStore
from base.tools.errors import ErrorsException


class AbstractStoreConformance:
    """Contract tests shared by every AbstractStore implementation."""

    def make_store(self) -> AbstractStore[Any]:
        raise NotImplementedError

    def make_entity(self, **overrides: Any) -> Any:
        raise NotImplementedError

    def identity(self, entity: Any) -> Any:
        return entity.id

    @pytest.fixture
    def store(self) -> AbstractStore[Any]:
        return self.make_store()

    # Add and get

    async def test_aadd_then_aget_returns_the_entity(self, store):
        entity = await store.aadd(self.make_entity())
        found = await store.aget(self.identity(entity))
        assert self.identity(found) == self.identity(entity)

    async def test_aget_missing_raises_record_not_found(self, store):
        with pytest.raises(ErrorsException) as excinfo:
            await store.aget(uuid.uuid4())
        assert excinfo.value.error_enum is DatabaseErrors.RECORD_NOT_FOUND

    async def test_aadd_many_preserves_order(self, store):
        entities = [self.make_entity() for _ in range(3)]
        stored = await store.aadd_many(entities)
        assert [self.identity(e) for e in stored] == [self.identity(e) for e in entities]

    async def test_aget_many_returns_only_requested(self, store):
        kept = await store.aadd(self.make_entity())
        await store.aadd(self.make_entity())
        found = await store.aget_many([self.identity(kept)])
        assert [self.identity(e) for e in found] == [self.identity(kept)]

    # Edit

    async def test_aedit_updates_in_place(self, store):
        entity = await store.aadd(self.make_entity())
        updated = await store.aedit(self.identity(entity), self.make_entity())
        assert updated is not None
        assert self.identity(await store.aget(self.identity(entity))) == self.identity(entity)

    async def test_aedit_missing_returns_none(self, store):
        assert await store.aedit(uuid.uuid4(), self.make_entity()) is None

    # Delete

    async def test_adelete_removes_entity(self, store):
        entity = await store.aadd(self.make_entity())
        assert await store.adelete(self.identity(entity)) is True
        with pytest.raises(ErrorsException):
            await store.aget(self.identity(entity))

    async def test_adelete_missing_returns_false(self, store):
        assert await store.adelete(uuid.uuid4()) is False

    async def test_adelete_is_not_observable_twice(self, store):
        entity = await store.aadd(self.make_entity())
        assert await store.adelete(self.identity(entity)) is True
        assert await store.adelete(self.identity(entity)) is False

    # Listing

    async def test_aget_all_returns_everything_added(self, store):
        added = [await store.aadd(self.make_entity()) for _ in range(3)]
        found = await store.aget_all()
        assert {self.identity(e) for e in added} <= {self.identity(e) for e in found}

    async def test_aget_all_paged_reports_totals(self, store):
        for _ in range(3):
            await store.aadd(self.make_entity())
        page = await store.aget_all_paged(FilteredPaginationParams(page=1, size=2))
        assert page.page == 1
        assert page.size == 2
        assert page.total >= 3
        assert len(page.data) <= 2

    # Sync counterparts

    def test_add_then_get_returns_the_entity(self, store):
        entity = store.add(self.make_entity())
        assert self.identity(store.get(self.identity(entity))) == self.identity(entity)

    def test_get_missing_raises_record_not_found(self, store):
        with pytest.raises(ErrorsException):
            store.get(uuid.uuid4())
