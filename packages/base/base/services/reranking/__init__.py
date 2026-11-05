"""Reordering retrieved passages by relevance to the query."""

import time
from abc import ABC, abstractmethod
from typing import final

from base.models.view_models.rerank import RerankBatch, RerankResult
from base.services.usage import UsageTracker


class AbstractRerankService(ABC):
    """A cross-encoder that scores each passage against the query.

    Retrieval ranks by vector proximity, which is cheap and approximate. A
    reranker reads query and passage together, so it costs more per document
    but is far more accurate — the usual shape is to over-fetch, rerank, and
    keep the top few.

    Recording lives here rather than in each adapter: it has to happen once per
    call, and an adapter retries inside itself.
    """

    @property
    @abstractmethod
    def model(self) -> str:
        """The reranking model, for the usage record."""

    @abstractmethod
    def _rerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> RerankBatch: ...

    @final
    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        """Reorder `documents`, best first. Empty input never calls a provider."""
        if not query.strip() or not documents:
            return []
        started = time.perf_counter()
        error: str | None = None
        batch: RerankBatch | None = None
        try:
            batch = self._rerank(query, documents, top_n)
            return batch.results
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record(batch, started, error)

    @abstractmethod
    async def _arerank(
        self, query: str, documents: list[str], top_n: int | None
    ) -> RerankBatch: ...

    @final
    async def arerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        """Reorder `documents`, best first. Empty input never calls a provider."""
        if not query.strip() or not documents:
            return []
        started = time.perf_counter()
        error: str | None = None
        batch: RerankBatch | None = None
        try:
            batch = await self._arerank(query, documents, top_n)
            return batch.results
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record(batch, started, error)

    def _record(
        self, batch: RerankBatch | None, started: float, error: str | None
    ) -> None:
        """Account for one rerank request, billed or failed.

        Reranking is billed per search unit rather than per token, so a record
        usually carries no cost. `UsageRecord.cost_usd` stays `None` when no
        rate is known, which is the point: an unpriced call is visible rather
        than silently free.
        """
        usage = batch.usage if batch else None
        UsageTracker().record(
            model=self.model,
            label=self.get_service_name(),
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            cached_input_tokens=usage.cached_input_tokens if usage else 0,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=error,
        )

    @abstractmethod
    def get_service_name(self) -> str: ...
