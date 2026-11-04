from typing import Any, Protocol, runtime_checkable

from base.models.view_models.vector_search import VectorSearchResult


@runtime_checkable
class SupportsLexicalSearch(Protocol):
    """A backend that can also search the stored text, not just the vector.

    Kept off `AbstractVectorSearchService` so backends without full-text are
    not forced to fake it. Callers check with `isinstance` and fall back to
    dense-only retrieval.
    """

    async def alexical_search(
        self,
        collection: str,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]: ...
