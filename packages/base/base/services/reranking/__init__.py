"""Reordering retrieved passages by relevance to the query."""

from abc import ABC, abstractmethod
from typing import final

from base.models.view_models.rerank import RerankResult


class AbstractRerankService(ABC):
    """A cross-encoder that scores each passage against the query.

    Retrieval ranks by vector proximity, which is cheap and approximate. A
    reranker reads query and passage together, so it costs more per document
    but is far more accurate — the usual shape is to over-fetch, rerank, and
    keep the top few.
    """

    @abstractmethod
    def _rerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> list[RerankResult]: ...

    @final
    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        """Reorder `documents`, best first. Empty input never calls a provider."""
        if not query.strip() or not documents:
            return []
        return self._rerank(query, documents, top_n)

    @abstractmethod
    async def _arerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> list[RerankResult]: ...

    @final
    async def arerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        """Reorder `documents`, best first. Empty input never calls a provider."""
        if not query.strip() or not documents:
            return []
        return await self._arerank(query, documents, top_n)

    @abstractmethod
    def get_service_name(self) -> str: ...
