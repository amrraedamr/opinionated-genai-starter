"""Fake Mongo collections and databases for the shaping tests.

A fake here answers like the driver rather than like a mock: a cursor chains
`sort`, `skip` and `limit` and is read once, writes return a result object
with the counts the service reads back, so the service's own code runs.
"""

from __future__ import annotations

from typing import Any


class FakeCursor:
    """A find cursor that records the chain and yields fixed documents."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.sorted_by: list[tuple[str, int]] | None = None
        self.skipped: int | None = None
        self.limited: int | None = None

    def sort(self, sort: list[tuple[str, int]]) -> FakeCursor:
        self.sorted_by = sort
        return self

    def skip(self, skip: int) -> FakeCursor:
        self.skipped = skip
        return self

    def limit(self, limit: int) -> FakeCursor:
        self.limited = limit
        return self

    def __iter__(self) -> Any:
        return iter(self.documents)

    async def to_list(self, length: int | None = None) -> list[dict[str, Any]]:
        del length
        return self.documents


class FakeWriteResult:
    """The counts and ids the service reads off a write."""

    def __init__(
        self,
        inserted_id: Any = "generated-id",
        inserted_ids: list[Any] | None = None,
        modified_count: int = 1,
        deleted_count: int = 1,
    ) -> None:
        self.inserted_id = inserted_id
        self.inserted_ids = inserted_ids or []
        self.modified_count = modified_count
        self.deleted_count = deleted_count


class FakeCollection:
    """Records every call the service makes, and answers with canned data."""

    def __init__(
        self, name: str, documents: list[dict[str, Any]] | None = None
    ) -> None:
        self.name = name
        self.documents: list[dict[str, Any]] = documents or []
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.cursors: list[FakeCursor] = []
        self.search_indexes: list[str] = []
        self.update_error: Exception | None = None
        self.drop_never_completes: bool = False
        self.result = FakeWriteResult()
        self.distinct_values: list[Any] = ["a", "b"]
        self.count_value: int = 0

    def _record(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((method, args, kwargs))

    def call(self, method: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """The arguments of the last call to `method`."""
        for name, args, kwargs in reversed(self.calls):
            if name == method:
                return args, kwargs
        raise AssertionError(f"{method} was never called on {self.name}")

    def called(self, method: str) -> int:
        return sum(1 for name, _, _ in self.calls if name == method)

    # Reads

    def find_one(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        self._record("find_one", *args, **kwargs)
        return self.documents[0] if self.documents else None

    def find(self, *args: Any, **kwargs: Any) -> FakeCursor:
        self._record("find", *args, **kwargs)
        cursor = FakeCursor(self.documents)
        self.cursors.append(cursor)
        return cursor

    def count_documents(self, *args: Any, **kwargs: Any) -> int:
        self._record("count_documents", *args, **kwargs)
        return self.count_value

    def distinct(self, *args: Any, **kwargs: Any) -> list[Any]:
        self._record("distinct", *args, **kwargs)
        return self.distinct_values

    def aggregate(self, *args: Any, **kwargs: Any) -> Any:
        self._record("aggregate", *args, **kwargs)
        cursor = FakeCursor(self.documents)
        self.cursors.append(cursor)
        return cursor

    # Writes

    def insert_one(self, *args: Any, **kwargs: Any) -> FakeWriteResult:
        self._record("insert_one", *args, **kwargs)
        return self.result

    def insert_many(self, *args: Any, **kwargs: Any) -> FakeWriteResult:
        self._record("insert_many", *args, **kwargs)
        return self.result

    def update_one(self, *args: Any, **kwargs: Any) -> FakeWriteResult:
        self._record("update_one", *args, **kwargs)
        return self.result

    def update_many(self, *args: Any, **kwargs: Any) -> FakeWriteResult:
        self._record("update_many", *args, **kwargs)
        return self.result

    def replace_one(self, *args: Any, **kwargs: Any) -> FakeWriteResult:
        self._record("replace_one", *args, **kwargs)
        return self.result

    def delete_one(self, *args: Any, **kwargs: Any) -> FakeWriteResult:
        self._record("delete_one", *args, **kwargs)
        return self.result

    def delete_many(self, *args: Any, **kwargs: Any) -> FakeWriteResult:
        self._record("delete_many", *args, **kwargs)
        return self.result

    # Search indexes

    def list_search_indexes(self, name: str | None = None, **kwargs: Any) -> list[Any]:
        self._record("list_search_indexes", name=name, **kwargs)
        return [{"name": name}] if name in self.search_indexes else []

    def create_search_index(self, model: Any) -> str:
        self._record("create_search_index", model)
        self.search_indexes.append(model.document["name"])
        return str(model.document["name"])

    def update_search_index(self, name: str, definition: dict[str, Any]) -> None:
        self._record("update_search_index", name, definition)
        if self.update_error is not None:
            raise self.update_error

    def drop_search_index(self, name: str) -> None:
        self._record("drop_search_index", name)
        if not self.drop_never_completes and name in self.search_indexes:
            self.search_indexes.remove(name)


class AsyncFakeCollection(FakeCollection):
    """The same recorder, with the driver's async signatures."""

    async def find_one(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:  # type: ignore[override]
        return FakeCollection.find_one(self, *args, **kwargs)

    async def count_documents(self, *args: Any, **kwargs: Any) -> int:  # type: ignore[override]
        return FakeCollection.count_documents(self, *args, **kwargs)

    async def distinct(self, *args: Any, **kwargs: Any) -> list[Any]:  # type: ignore[override]
        return FakeCollection.distinct(self, *args, **kwargs)

    async def aggregate(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return FakeCollection.aggregate(self, *args, **kwargs)

    async def insert_one(self, *args: Any, **kwargs: Any) -> FakeWriteResult:  # type: ignore[override]
        return FakeCollection.insert_one(self, *args, **kwargs)

    async def insert_many(self, *args: Any, **kwargs: Any) -> FakeWriteResult:  # type: ignore[override]
        return FakeCollection.insert_many(self, *args, **kwargs)

    async def update_one(self, *args: Any, **kwargs: Any) -> FakeWriteResult:  # type: ignore[override]
        return FakeCollection.update_one(self, *args, **kwargs)

    async def update_many(self, *args: Any, **kwargs: Any) -> FakeWriteResult:  # type: ignore[override]
        return FakeCollection.update_many(self, *args, **kwargs)

    async def replace_one(self, *args: Any, **kwargs: Any) -> FakeWriteResult:  # type: ignore[override]
        return FakeCollection.replace_one(self, *args, **kwargs)

    async def delete_one(self, *args: Any, **kwargs: Any) -> FakeWriteResult:  # type: ignore[override]
        return FakeCollection.delete_one(self, *args, **kwargs)

    async def delete_many(self, *args: Any, **kwargs: Any) -> FakeWriteResult:  # type: ignore[override]
        return FakeCollection.delete_many(self, *args, **kwargs)

    async def list_search_indexes(  # type: ignore[override]
        self, name: str | None = None, **kwargs: Any
    ) -> FakeCursor:
        existing = FakeCollection.list_search_indexes(self, name=name, **kwargs)
        return FakeCursor(existing)

    async def create_search_index(self, model: Any) -> str:  # type: ignore[override]
        return FakeCollection.create_search_index(self, model)

    async def update_search_index(  # type: ignore[override]
        self, name: str, definition: dict[str, Any]
    ) -> None:
        FakeCollection.update_search_index(self, name, definition)

    async def drop_search_index(self, name: str) -> None:  # type: ignore[override]
        FakeCollection.drop_search_index(self, name)


class FakeDatabase:
    """Hands out one fake collection per name, and answers `ping`."""

    def __init__(self, collection_type: type[FakeCollection] = FakeCollection) -> None:
        self._type = collection_type
        self.collections: dict[str, FakeCollection] = {}
        self.commands: list[str] = []

    def __getitem__(self, name: str) -> FakeCollection:
        if name not in self.collections:
            self.collections[name] = self._type(name)
        return self.collections[name]

    def command(self, name: str) -> dict[str, Any]:
        self.commands.append(name)
        return {"ok": 1}


class AsyncFakeDatabase(FakeDatabase):
    """The async database handle."""

    def __init__(self) -> None:
        super().__init__(AsyncFakeCollection)

    async def command(self, name: str) -> dict[str, Any]:  # type: ignore[override]
        self.commands.append(name)
        return {"ok": 1}
