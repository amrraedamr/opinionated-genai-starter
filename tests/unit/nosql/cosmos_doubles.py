"""Fake Cosmos containers for the query-shaping tests.

No Cosmos account is reachable from this repository, so the whole of the
Cosmos adapter is pinned by reading the SQL it builds and the item calls it
makes. The container answers like the SDK's `ContainerProxy`: `query_items`
is an iterable of results, the item calls return a body.
"""

from __future__ import annotations

from typing import Any

from azure.cosmos import exceptions


class FakeContainer:
    """Records every call and answers with whatever the test queued."""

    def __init__(self, results: list[Any] | None = None) -> None:
        self.results: list[Any] = results if results is not None else []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.missing: bool = False
        self.upsert_error: Exception | None = None

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))

    def call(self, method: str) -> dict[str, Any]:
        for name, kwargs in reversed(self.calls):
            if name == method:
                return kwargs
        raise AssertionError(f"{method} was never called")

    def called(self, method: str) -> int:
        return sum(1 for name, _ in self.calls if name == method)

    @property
    def queries(self) -> list[str]:
        return [kwargs["query"] for name, kwargs in self.calls if name == "query_items"]

    def query_items(self, **kwargs: Any) -> Any:
        self._record("query_items", **kwargs)
        return iter(list(self.results))

    def read_item(self, **kwargs: Any) -> dict[str, Any]:
        self._record("read_item", **kwargs)
        if self.missing:
            raise exceptions.CosmosResourceNotFoundError(message="no such item")
        return {"id": kwargs["item"]}

    def upsert_item(self, **kwargs: Any) -> dict[str, Any]:
        self._record("upsert_item", **kwargs)
        if self.upsert_error is not None:
            raise self.upsert_error
        return dict(kwargs["body"])

    def delete_item(self, **kwargs: Any) -> None:
        self._record("delete_item", **kwargs)


class AsyncFakeContainer(FakeContainer):
    """The same recorder with the async SDK's signatures."""

    def query_items(self, **kwargs: Any) -> Any:
        self._record("query_items", **kwargs)
        results = list(self.results)

        async def iterator() -> Any:
            for item in results:
                yield item

        return iterator()

    async def read_item(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return FakeContainer.read_item(self, **kwargs)

    async def upsert_item(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return FakeContainer.upsert_item(self, **kwargs)

    async def delete_item(self, **kwargs: Any) -> None:  # type: ignore[override]
        FakeContainer.delete_item(self, **kwargs)
